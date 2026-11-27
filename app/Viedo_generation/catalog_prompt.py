"""Resolve the agent's edited list against the property catalog."""

import re

from app.database.postgres import fetch


class PromptError(ValueError):
    pass


def resolve_catalog_selection(
    prompt: str | None,
    project_keys: list[str] | None = None,
    communities: list[str] | None = None,
) -> list[dict]:
    """Use submitted names or keys as the selection and resolve all data in the catalog."""
    if prompt is None:
        prompt = ", ".join([*(project_keys or []), *(communities or [])])
    return catalog_candidates(prompt)


def catalog_candidates(prompt: str) -> list[dict]:
    terms = {}
    for part in re.split(r"[,;\n]+", prompt):
        name = re.sub(
            r"\s+\((?:studio|\d+(?:\s*-\s*\d+)?\s*bed)\)$",
            "", part.strip(), flags=re.IGNORECASE,
        ).strip()
        if name:
            terms.setdefault(name.lower(), name)
    if not terms or len(terms) > 20:
        raise PromptError("Enter between 1 and 20 project or community names, separated by commas.")

    candidates = {}
    for normalized, name in terms.items():
        # Exact project names/keys take priority over expanding a community.
        matches = fetch(
            """
            select key, name, area, master_project
            from public.propai_projects
            where not is_plot and (lower(btrim(key)) = %s or lower(btrim(name)) = %s)
            order by (lower(btrim(key)) = %s) desc, key
            limit 6
            """,
            (normalized, normalized, normalized),
        )
        if matches and matches[0]["key"].strip().lower() == normalized:
            matches = matches[:1]
        elif len(matches) > 1:
            options = ", ".join(row["key"] for row in matches[:5])
            raise PromptError(f'"{name}" matches several projects. Use an exact project key: {options}.')
        explicit_project = bool(matches)
        if not matches:
            matches = fetch(
                """
                select key, name, area, master_project
                from public.propai_projects p
                where not is_plot
                  and (lower(btrim(area)) = %s or lower(btrim(master_project)) = %s)
                  and exists (
                      select 1 from public.propai_project_images i
                      where i.project_key = p.key
                      group by i.project_key having count(*) >= 4
                  )
                order by image_count desc nulls last, key
                limit 30
                """,
                (normalized, normalized),
            )
            if not matches:
                raise PromptError(f'No catalog project or community with available photos matches "{name}".')
        for row in matches:
            candidate = candidates.setdefault(row["key"], {
                "key": row["key"],
                "name": row["name"],
                "community": row.get("area"),
                "explicit_project": False,
            })
            candidate["explicit_project"] |= explicit_project
    return list(candidates.values())
