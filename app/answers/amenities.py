"""Property Finder amenity codes, as stored in `projects.amenities`.

Keep in step with AMENITIES in ACOT-BACKEND/src/listings/project-detail.ts.
Passing the raw codes to the model made it guess ("SP" became "spa").
Brochure text often writes the same thing as "PR (Private Parking)". The chat
and the spoken reply should say the name only.
"""

import re

# Two-letter code, then the readable name in parentheses.
CODE_THEN_NAME = re.compile(r"\b[A-Z]{2}\s*\(([^)]+)\)")

AMENITY_LABELS = {
    "SP": "Shared pool",
    "SY": "Shared gym",
    "SE": "Security",
    "PR": "Children's play area",
    "PG": "Private garden",
}


def amenity_labels(codes):
    """Readable names for the codes we know; unknown codes are dropped, not guessed."""
    if not codes:
        return []
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",")]
    labels = []
    for code in codes:
        label = AMENITY_LABELS.get(str(code).strip().upper())
        if label and label not in labels:
            labels.append(label)
    return labels


def strip_amenity_codes(text):
    """'PR (Private Parking)' becomes 'Private Parking'. Other text is left as is."""
    if not text:
        return text
    return CODE_THEN_NAME.sub(r"\1", str(text))
