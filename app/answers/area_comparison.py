"""
"Compare Dubai Marina and Business Bay": a side-by-side table of two areas, built from the database.

No language model is involved, so the answer costs no tokens. The numbers come from the same queries the
agent's market_stats and area_stats tools run. The profile rows (community type, location, lifestyle) are
general knowledge about well-known areas and are marked as such; an area without a profile simply gets no
profile rows. The conclusion is written from the numbers and never claims a yield or a return.
"""
from __future__ import annotations

import re

from app.database.postgres import fetch
from app.listings.tools import ALIASES, area_stats, market_stats

# Short, safe descriptions of well-known areas. General knowledge, not ACOT data.
PROFILES = {
    "Dubai Marina": ("Waterfront residential towers", "Coastal, beside JBR and the Marina promenade",
                     "Waterfront dining, beach and leisure", "Mostly residential apartments"),
    "Business Bay": ("Urban mixed-use district", "Central Dubai, next to Downtown and the canal",
                     "Business and city living", "Residential, office and hotel towers"),
    "Downtown Dubai": ("Premium urban centre", "Central, around Burj Khalifa and Dubai Mall",
                       "Landmark city living, shopping and dining", "Residential, hotels and retail"),
    "Jumeirah Village Circle": ("Mid-market residential community", "Inland, central-west Dubai",
                                "Quieter community living with parks", "Mostly apartments and townhouses"),
    "Jumeirah Lake Towers": ("Towers around man-made lakes", "Next to Dubai Marina on Sheikh Zayed Road",
                             "Lakeside cafes and easy metro access", "Residential and commercial towers"),
    "Jumeirah Beach Residence": ("Beachfront residential", "On the beach, beside Dubai Marina",
                                 "Beach, promenade and leisure", "Mostly residential apartments"),
    "Palm Jumeirah": ("Luxury island community", "Man-made island off the coast",
                      "Beach and resort-style living", "Villas, apartments and hotels"),
    "Dubai Hills Estate": ("Master-planned family community", "Central-inland, off Al Khail Road",
                           "Golf, parks and family living", "Villas, townhouses and apartments"),
    "Dubai Creek Harbour": ("Waterfront master plan", "On the creek, near Ras Al Khor",
                            "Waterfront living with city-skyline views", "Mostly apartments"),
    "Dubai South": ("Master-planned area near the airport", "South-west Dubai, near Al Maktoum Airport",
                    "Newer, quieter, growing community", "Apartments and townhouses"),
    "Dubai Silicon Oasis": ("Technology and residential community", "Inland, east Dubai",
                            "Practical, family-friendly living", "Mostly apartments"),
    "Al Furjan": ("Residential community", "Inland, south-west Dubai",
                  "Quiet family living", "Townhouses, villas and apartments"),
    "Dubai Sports City": ("Sports-themed residential community", "Inland, near Motor City",
                          "Sports facilities and a lower-density feel", "Mostly apartments"),
    "Motor City": ("Residential community", "Inland, near the Autodrome",
                   "Quiet community living", "Apartments and townhouses"),
    "Arjan": ("Mid-market residential area", "Inland, central-west Dubai",
              "Practical residential living", "Mostly apartments"),
}
_PROFILE_LOWER = {name.lower(): name for name in PROFILES}

_COMPARE = (
    re.compile(r"^\s*(?:please\s+)?(?:can you\s+)?compar(?:e|ing|ison of)\s+(.+?)\s+(?:and|with|to|vs\.?|versus|&)\s+(.+?)\s*[?.!]*$", re.I),
    re.compile(r"^\s*(?:what(?:'s| is)\s+)?(?:the\s+)?difference between\s+(.+?)\s+and\s+(.+?)\s*[?.!]*$", re.I),
    re.compile(r"^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[?.!]*$", re.I),
)
_TRAILING = re.compile(r"\s+(?:for|in terms of|regarding|as|when it comes to|on)\s+.*$", re.I)

_names: dict[str, str] | None = None


def _known_areas() -> dict[str, str]:
    """lower-case name -> proper name, for every community that has projects or rentals."""
    global _names
    if _names is None:
        rows = fetch("select btrim(community) as n from projects where community is not null union "
                     "select btrim(community) from listings where community is not null")
        _names = {r["n"].lower(): r["n"] for r in rows if r.get("n")}
    return _names


def _resolve(text: str) -> str | None:
    text = re.sub(r"\s+", " ", _TRAILING.sub("", text or "")).strip(" ?.!,'\"")
    text = re.sub(r"^the\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(?:area|community|neighbourhood|neighborhood)$", "", text, flags=re.I)
    known = _known_areas()
    for candidate in (text, re.sub(r",?\s*dubai$", "", text, flags=re.I)):
        key = candidate.lower().strip()
        if not key or len(key) > 60:
            continue
        alias = ALIASES.get(key)
        for name in (alias.lower() if alias else None, key, f"dubai {key}"):
            if name and name in known:
                return known[name]
    return None


def parse_area_comparison(question: str) -> tuple[str, str] | None:
    """The two area names when the question compares two known areas, else None (the agent handles it)."""
    for pattern in _COMPARE:
        match = pattern.match(question or "")
        if match:
            first, second = _resolve(match.group(1)), _resolve(match.group(2))
            if first and second and first != second:
                return first, second
            return None
    return None


# ----------------------------------------------------------------------------- numbers

def _aed(value) -> str:
    return f"AED {int(round(value)):,}"


def _facts(area: str) -> dict:
    projects = market_stats({"communities": [area]}).output
    rentals = area_stats({"areas": [area]}).output
    prow = (projects.get("rows") or [{}])[0]
    rrow = (rentals.get("rows") or [{}])[0]
    years = projects.get("handover_years") or []
    top_year = max(years, key=lambda y: y["projects"]) if years else None
    beds = prow.get("bedrooms_offered") or []
    return {
        "projects": int(prow.get("projects") or 0),
        "min_price": prow.get("min_starting_price_aed"),
        "median_price": prow.get("median_starting_price_aed"),
        "beds": beds if len(beds) == 2 else None,
        "top_year": top_year,
        "listings": int(rrow.get("listings") or 0),
        "rent_median": rrow.get("median_annual_rent_aed"),
        "rent_range": rrow.get("typical_range_annual_aed"),
        "rent_psf": rrow.get("avg_rent_per_sqft_per_year_aed"),
    }


def _beds_text(beds) -> str:
    if not beds:
        return "n/a"
    low, high = beds
    if low == high:
        return "Studios" if low == 0 else f"{low} bedroom" + ("s" if low != 1 else "")
    return f"Studios to {high} bedrooms" if low == 0 else f"{low} to {high} bedrooms"


def _cells(f: dict) -> list[tuple[str, str]]:
    price = "n/a"
    if f["min_price"]:
        price = f"from {_aed(f['min_price'])}"
        if f["median_price"] and f["projects"] >= 3:
            price += f" (median {_aed(f['median_price'])})"
    handover = "n/a"
    if f["top_year"] and f["projects"]:
        handover = f"mostly {f['top_year']['year']} ({f['top_year']['projects']} of {f['projects']})" \
            if f["projects"] > 1 else str(f["top_year"]["year"])
    rent = "n/a"
    if f["rent_median"]:
        rent = f"median {_aed(f['rent_median'])}"
        if f["rent_range"]:
            rent += f", typically {_aed(f['rent_range'][0])} to {_aed(f['rent_range'][1])}"
    return [
        ("Projects for sale (off-plan)", str(f["projects"])),
        ("Starting prices", price),
        ("Bedrooms offered", _beds_text(f["beds"])),
        ("Handover", handover),
        ("Rental listings", str(f["listings"])),
        ("Yearly rent", rent),
        ("Rent per sqft / year", _aed(f["rent_psf"]) if f["rent_psf"] else "n/a"),
    ]


def _conclusion(a: str, b: str, fa: dict, fb: dict) -> list[str]:
    pa, pb = PROFILES.get(a), PROFILES.get(b)
    lines = []
    if pa and pb:
        lines.append(f"{a} is known for {pa[2].lower()}, while {b} is known for {pb[2].lower()}.")
    wins: dict[str, list[str]] = {a: [], b: []}
    if fa["projects"] != fb["projects"]:
        wins[a if fa["projects"] > fb["projects"] else b].append(
            f"more projects for sale ({max(fa['projects'], fb['projects'])} against "
            f"{min(fa['projects'], fb['projects'])})")
    if fa["min_price"] and fb["min_price"] and fa["min_price"] != fb["min_price"]:
        wins[a if fa["min_price"] < fb["min_price"] else b].append("the lower starting price")
    if fa["rent_median"] and fb["rent_median"] and fa["rent_median"] != fb["rent_median"]:
        wins[a if fa["rent_median"] < fb["rent_median"] else b].append("the lower typical rent")
    if fa["listings"] != fb["listings"]:
        wins[a if fa["listings"] > fb["listings"] else b].append("more rental listings")
    parts = [f"{name} has {' and '.join(items)}" for name, items in wins.items() if items]
    if parts:
        lines.append("In our data, " + "; ".join(parts) + ".")
    thin = [n for n, f in ((a, fa), (b, fb)) if f["projects"] < 3 and f["listings"] < 5]
    if thin:
        lines.append(f"Our data on {' and '.join(thin)} is thin, so read the numbers as a rough guide.")
    lines.append("The better choice depends on your budget, whether you want to buy off-plan or rent, "
                 "and your investment strategy. ACOT has no yield or price-growth data, so this is not a "
                 "return forecast.")
    return lines


def build_area_comparison(a: str, b: str) -> str:
    fa, fb = _facts(a), _facts(b)
    rows = list(zip(_cells(fa), _cells(fb)))
    pa, pb = PROFILES.get(a), PROFILES.get(b)

    out = [f"{a} vs {b}", "", f"| Factor | {a} | {b} |", "| --- | --- | --- |"]
    if pa and pb:
        for index, label in enumerate(("Community type", "Location", "Lifestyle", "Property focus")):
            out.append(f"| {label} | {pa[index]} | {pb[index]} |")
    for (label, left), (_, right) in rows:
        out.append(f"| {label} | {left} | {right} |")
    out.append("")
    if pa and pb:
        out.append("Community type, location, lifestyle and property focus are general guidance. "
                   "Everything else comes from the ACOT data.")
        out.append("")
    out.append("Summary:")
    out.extend(_conclusion(a, b, fa, fb))
    out.append("")
    out.append(f"Would you like to see the projects or rentals in {a} or {b}, or narrow by budget or bedrooms?")
    return "\n".join(out)
