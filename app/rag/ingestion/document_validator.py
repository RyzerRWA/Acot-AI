class DocumentValidator:
    """
    Validates extracted document content before
    sending it to the embedding pipeline.
    """

    MIN_CHARACTERS = 500
    MIN_WORDS = 100

    IMPORTANT_TERMS = [
        "project",
        "developer",
        "location",
        "community",
        "amenities",
        "apartment",
        "bedroom",
        "price",
        "handover",
        "area",
        "floor",
        "property",
        "residence",
    ]

    @classmethod
    def validate(cls, text: str, project_name: str = None):
        """
        Validate extracted document text.

        Returns:
            {
                "valid": bool,
                "reason": str,
                "text_length": int,
                "word_count": int,
                "matched_terms": list
            }
        """

        if not text:
            return {
                "valid": False,
                "reason": "Empty document",
                "text_length": 0,
                "word_count": 0,
                "matched_terms": []
            }

        text = text.strip()

        text_length = len(text)
        words = text.split()
        word_count = len(words)

        # -----------------------------------------------------
        # Minimum content check
        # -----------------------------------------------------

        if text_length < cls.MIN_CHARACTERS:
            return {
                "valid": False,
                "reason": (
                    f"Document too short "
                    f"({text_length} characters)"
                ),
                "text_length": text_length,
                "word_count": word_count,
                "matched_terms": []
            }

        if word_count < cls.MIN_WORDS:
            return {
                "valid": False,
                "reason": (
                    f"Too few words "
                    f"({word_count} words)"
                ),
                "text_length": text_length,
                "word_count": word_count,
                "matched_terms": []
            }

        # -----------------------------------------------------
        # Real-estate content check
        # -----------------------------------------------------

        text_lower = text.lower()

        matched_terms = [
            term
            for term in cls.IMPORTANT_TERMS
            if term in text_lower
        ]

        if len(matched_terms) < 2:
            return {
                "valid": False,
                "reason": "Document does not appear to contain useful real-estate information",
                "text_length": text_length,
                "word_count": word_count,
                "matched_terms": matched_terms
            }

        # -----------------------------------------------------
        # Project name check
        # -----------------------------------------------------

        project_match = None

        if project_name:
            project_match = (
                project_name.lower() in text_lower
            )

        return {
            "valid": True,
            "reason": "Document contains sufficient useful content",
            "text_length": text_length,
            "word_count": word_count,
            "matched_terms": matched_terms,
            "project_name_found": project_match
        }