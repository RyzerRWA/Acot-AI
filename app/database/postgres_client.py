"""
PostgreSQL client for the ACOT AI retrieval and embedding pipeline.

This is a small drop-in replacement for the parts of the previous REST
client that the pipeline uses, so the retrievers and ingestion code do
not need to change:

    client.table("projects").select("id, name").ilike("name", "%x%").limit(5).execute().data
    client.table("projects").update({...}).eq("id", 1).execute()
    client.table("document_chunks").insert({...}).execute()
    client.rpc("match_projects", {...}).execute().data

Queries go straight to PostgreSQL through psycopg, using DATABASE_URL.
Rows are returned as JSON-style dicts (numbers stay numbers, timestamps
are ISO strings), which matches the previous client shape.
"""

import atexit
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_pools: Dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


# =========================================================
# HELPERS
# =========================================================

def _identifier(name: str) -> sql.Identifier:
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return sql.Identifier(name)


def _is_number_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            for item in value
        )
    )


def _placeholder(value: Any):
    """
    Return (SQL fragment, adapted value) for a bound parameter.

    Lists of numbers are cast to real[] because the embedding columns
    and the match_* functions use real[] (psycopg sends float8[]).
    """

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        return sql.SQL("%s"), Jsonb(value)

    if _is_number_list(value):
        return sql.SQL("%s::real[]"), [float(item) for item in value]

    return sql.SQL("%s"), value


def _get_pool(database_url: str) -> ConnectionPool:
    with _pools_lock:
        pool = _pools.get(database_url)

        if pool is None:
            pool = ConnectionPool(
                conninfo=database_url,
                min_size=1,
                max_size=10,
                kwargs={
                    "row_factory": dict_row,
                    "autocommit": True,
                    "connect_timeout": 15,
                },
                open=False,
            )
            pool.open(wait=True, timeout=30)
            _pools[database_url] = pool

        return pool


def close_pools() -> None:
    with _pools_lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:
                pass
        _pools.clear()


atexit.register(close_pools)


# =========================================================
# RESPONSE
# =========================================================

class PostgresResponse:
    """Mimics the previous response object (`.data`)."""

    def __init__(self, data: Any):
        self.data = data
        self.count = len(data) if isinstance(data, list) else None


# =========================================================
# QUERY BUILDER
# =========================================================

class PostgresQuery:

    def __init__(
        self,
        client: "PostgresClient",
        schema: str,
        table: str
    ):
        self._client = client
        self._schema = schema
        self._table = table

        self._operation = "select"
        self._columns: List[str] = ["*"]
        self._payload: Any = None
        self._filters: List[tuple] = []
        self._limit: Optional[int] = None
        self._single = False

    # -----------------------------------------------------
    # Operations
    # -----------------------------------------------------

    def select(self, columns: str = "*"):
        self._operation = "select"

        parsed = [
            part.strip()
            for part in str(columns).split(",")
            if part.strip()
        ]

        self._columns = parsed or ["*"]
        return self

    def update(self, payload: Dict[str, Any]):
        self._operation = "update"
        self._payload = payload
        return self

    def insert(self, payload: Any):
        self._operation = "insert"
        self._payload = payload
        return self

    # -----------------------------------------------------
    # Filters
    # -----------------------------------------------------

    def eq(self, column: str, value: Any):
        self._filters.append(("eq", column, value))
        return self

    def ilike(self, column: str, pattern: str):
        self._filters.append(("ilike", column, pattern))
        return self

    def is_(self, column: str, value: Any):
        self._filters.append(("is", column, value))
        return self

    def limit(self, count: int):
        self._limit = int(count)
        return self

    def single(self):
        self._single = True
        return self

    # -----------------------------------------------------
    # SQL building
    # -----------------------------------------------------

    def _where(self):
        clauses = []
        params: List[Any] = []

        for kind, column, value in self._filters:

            col = _identifier(column)

            if kind == "eq":
                clauses.append(sql.SQL("{} = %s").format(col))
                params.append(value)

            elif kind == "ilike":
                clauses.append(sql.SQL("{} ILIKE %s").format(col))
                params.append(value)

            elif kind == "is":
                keyword = str(value).strip().lower()

                if keyword == "null":
                    clauses.append(sql.SQL("{} IS NULL").format(col))
                elif keyword == "true":
                    clauses.append(sql.SQL("{} IS TRUE").format(col))
                elif keyword == "false":
                    clauses.append(sql.SQL("{} IS FALSE").format(col))
                else:
                    raise ValueError(
                        f"Unsupported is_ value: {value!r}"
                    )

        if not clauses:
            return sql.SQL(""), params

        return (
            sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses),
            params,
        )

    def _target(self):
        return sql.SQL("{}.{}").format(
            _identifier(self._schema),
            _identifier(self._table),
        )

    def _build_select(self):

        if self._columns == ["*"]:
            column_sql = sql.SQL("*")
        else:
            column_sql = sql.SQL(", ").join(
                _identifier(column) for column in self._columns
            )

        where_sql, params = self._where()

        inner = (
            sql.SQL("SELECT ")
            + column_sql
            + sql.SQL(" FROM ")
            + self._target()
            + where_sql
        )

        if self._limit is not None:
            inner += sql.SQL(" LIMIT {}").format(
                sql.Literal(self._limit)
            )

        query = (
            sql.SQL("SELECT to_jsonb(t) AS row FROM (")
            + inner
            + sql.SQL(") t")
        )

        return query, params

    def _build_update(self):

        if not isinstance(self._payload, dict) or not self._payload:
            raise ValueError("update() needs a non-empty dict.")

        if not self._filters:
            raise ValueError(
                "update() without a filter is not allowed."
            )

        assignments = []
        params: List[Any] = []

        for column, value in self._payload.items():
            fragment, adapted = _placeholder(value)
            assignments.append(
                sql.SQL("{} = ").format(_identifier(column)) + fragment
            )
            params.append(adapted)

        where_sql, where_params = self._where()

        query = (
            sql.SQL("UPDATE ")
            + self._target()
            + sql.SQL(" AS t SET ")
            + sql.SQL(", ").join(assignments)
            + where_sql
            + sql.SQL(" RETURNING to_jsonb(t) AS row")
        )

        return query, params + where_params

    def _build_insert(self):

        rows = (
            self._payload
            if isinstance(self._payload, list)
            else [self._payload]
        )

        if not rows or not all(
            isinstance(row, dict) and row for row in rows
        ):
            raise ValueError(
                "insert() needs a non-empty dict or list of dicts."
            )

        columns = list(rows[0].keys())

        for row in rows:
            if list(row.keys()) != columns:
                raise ValueError(
                    "All inserted rows must have the same columns."
                )

        value_groups = []
        params: List[Any] = []

        for row in rows:
            fragments = []

            for column in columns:
                fragment, adapted = _placeholder(row[column])
                fragments.append(fragment)
                params.append(adapted)

            value_groups.append(
                sql.SQL("(")
                + sql.SQL(", ").join(fragments)
                + sql.SQL(")")
            )

        query = (
            sql.SQL("INSERT INTO ")
            + self._target()
            + sql.SQL(" AS t (")
            + sql.SQL(", ").join(_identifier(c) for c in columns)
            + sql.SQL(") VALUES ")
            + sql.SQL(", ").join(value_groups)
            + sql.SQL(" RETURNING to_jsonb(t) AS row")
        )

        return query, params

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------

    def execute(self) -> PostgresResponse:

        if self._operation == "select":
            query, params = self._build_select()
        elif self._operation == "update":
            query, params = self._build_update()
        elif self._operation == "insert":
            query, params = self._build_insert()
        else:
            raise ValueError(f"Unsupported operation: {self._operation}")

        with self._client.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = [item["row"] for item in cursor.fetchall()]

        if self._single:
            if len(rows) != 1:
                raise LookupError(
                    f"Expected exactly one row, got {len(rows)}."
                )
            return PostgresResponse(rows[0])

        return PostgresResponse(rows)


# =========================================================
# RPC
# =========================================================

class PostgresRPC:

    def __init__(
        self,
        client: "PostgresClient",
        schema: str,
        function_name: str,
        params: Optional[Dict[str, Any]]
    ):
        self._client = client
        self._schema = schema
        self._function = function_name
        self._params = params or {}

    def execute(self) -> PostgresResponse:

        arguments = []
        values: List[Any] = []

        for name, value in self._params.items():
            fragment, adapted = _placeholder(value)
            arguments.append(
                sql.SQL("{} => ").format(_identifier(name)) + fragment
            )
            values.append(adapted)

        query = (
            sql.SQL("SELECT * FROM {}.{}(").format(
                _identifier(self._schema),
                _identifier(self._function),
            )
            + sql.SQL(", ").join(arguments)
            + sql.SQL(")")
        )

        with self._client.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, values)
                rows = cursor.fetchall()

        data = []

        for row in rows:
            # SETOF jsonb / scalar functions return one column named
            # after the function; unwrap it like PostgREST does.
            if len(row) == 1 and self._function in row:
                data.append(row[self._function])
            else:
                data.append(row)

        return PostgresResponse(data)


# =========================================================
# CLIENT
# =========================================================

class PostgresClient:

    def __init__(
        self,
        database_url: Optional[str] = None,
        schema: str = "public"
    ):
        url = (database_url or os.getenv("DATABASE_URL") or "").strip()

        if not url:
            raise ValueError(
                "DATABASE_URL is not configured. "
                f"Checked: {PROJECT_ROOT / '.env'}"
            )

        self._database_url = url
        self._schema = schema
        self._pool: Optional[ConnectionPool] = None

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = _get_pool(self._database_url)
        return self._pool

    def schema(self, name: str) -> "PostgresClient":
        return PostgresClient(self._database_url, schema=name)

    def table(self, name: str) -> PostgresQuery:
        return PostgresQuery(self, self._schema, name)

    def rpc(
        self,
        function_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> PostgresRPC:
        return PostgresRPC(self, self._schema, function_name, params)


def get_postgres_client() -> PostgresClient:
    return PostgresClient()
