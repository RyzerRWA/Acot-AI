class RelevanceFilter:

    def filter_documents(
        self,
        question: str,
        results: list
    ):

        question = question.lower()

        # -----------------------------
        # QUESTION TOPICS
        # -----------------------------

        investment_keywords = [
            "investment",
            "roi",
            "return",
            "market",
            "growth",
            "trend",
            "opportunity"
        ]

        building_keywords = [
            "building",
            "safety",
            "code",
            "construction",
            "requirements"
        ]


        # -----------------------------
        # DETECT QUESTION TYPE
        # -----------------------------

        is_investment_question = any(
            keyword in question
            for keyword in investment_keywords
        )

        is_building_question = any(
            keyword in question
            for keyword in building_keywords
        )


        filtered_results = []


        # -----------------------------
        # FILTER RESULTS
        # -----------------------------

        for item in results:

            document = item["document"]

            metadata = document.get(
                "metadata",
                {}
            )

            title = metadata.get(
                "document_title",
                ""
            ).lower()

            scope = metadata.get(
                "scope",
                ""
            ).lower()


            document_text = (
                title + " " + scope
            )


            # -------------------------
            # INVESTMENT QUESTION
            # -------------------------

            if is_investment_question:

                # Skip building-related docs
                if any(
                    keyword in document_text
                    for keyword in building_keywords
                ):

                    continue

            # -------------------------
            # BUILDING QUESTION
            # -------------------------

            if is_building_question:

                # Prefer building documents
                if not any(
                    keyword in document_text
                    for keyword in building_keywords
                ):

                    continue

            filtered_results.append(item)

        return filtered_results