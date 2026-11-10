"""Search one embedded Prop-AI table.

    python scripts/search_propai.py "dubai marina"
    python scripts/search_propai.py "azizi riviera" --table raw
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.propai_vector_search import search_propai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--table", default="places", choices=("places", "raw", "status", "brokers"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    for row in search_propai(args.query, args.table, args.limit):
        score = row.get("score")
        label = row.get("label") or row.get("name") or row.get("name_en") or row.get("row_key")
        extra = row.get("area") or row.get("agency_name") or row.get("developer") or ""
        print(f"{score:.3f}  {row.get('level') or args.table:16}  {label}  | {extra}")


if __name__ == "__main__":
    main()
