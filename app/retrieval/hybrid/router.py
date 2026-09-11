class HybridRouter:

    def route(self, question: str):

        question_lower = question.lower().strip()

        # =====================================================
        # KEYWORDS
        # =====================================================

        # -----------------------------------------------------
        # STRUCTURED DATA KEYWORDS
        # -----------------------------------------------------

        structured_keywords = [
            "community",
            "communities",

            "property",
            "properties",

            "project",
            "projects",

            "price",
            "prices",

            "rent",
            "rental",

            "bedroom",
            "bedrooms",

            "area",
            "size",

            "landmark",
            "landmarks",

            "district",

            "developer",
            "developers",

            "handover",
        ]

        # -----------------------------------------------------
        # DOCUMENT / BROCHURE KEYWORDS
        # -----------------------------------------------------

        document_keywords = [
            # Regulations / government documents
            "building code",
            "building codes",
            "requirements",
            "regulation",
            "regulations",
            "law",
            "laws",
            "safety",
            "security",
            "security features",
            "guidelines",
            "government",

            # Project brochure information
            "amenities",
            "amenity",
            "facilities",
            "facility",
            "features",

            "interior",
            "interiors",
            "finishes",

            "kitchen",
            "appliances",

            "views",
            "view",

            "floor plan",
            "floor plans",

            "layout",
            "layouts",

            "dimensions",

            "clubhouse",
            "pool",
            "gym",
            "cinema",
            "lobby",
            "surveillance",
        ]

        # -----------------------------------------------------
        # HYBRID KEYWORDS
        # -----------------------------------------------------

        hybrid_keywords = [
            "investment",
            "good investment",
            "market",
            "market performance",
            "market analysis",
            "trend",
            "growth",
            "roi",
            "return",
            "opportunity",
            "forecast",
        ]

        # =====================================================
        # DETECT WHETHER QUERY IS PROJECT-SPECIFIC
        # =====================================================

        project_indicators = [
            "project",
            "property",
            "development",
            "villa",
            "apartment",
            "residence",
            "residences",
            "tower",
            "building",
        ]

        has_project_indicator = any(
            keyword in question_lower
            for keyword in project_indicators
        )

        # =====================================================
        # CHECK KEYWORD MATCHES
        # =====================================================

        has_structured = any(
            keyword in question_lower
            for keyword in structured_keywords
        )

        has_document = any(
            keyword in question_lower
            for keyword in document_keywords
        )

        has_hybrid = any(
            keyword in question_lower
            for keyword in hybrid_keywords
        )

        # =====================================================
        # HYBRID CHECK
        # =====================================================

        # Investment / market questions always need
        # structured + document/context information.

        if has_hybrid:

            return {
                "route": "hybrid",

                "question_type": (
                    "investment"
                    if "investment" in question_lower
                    else "general"
                ),

                "structured": True,
                "documents": True,
                "hybrid": True
            }

        # =====================================================
        # STRUCTURED + DOCUMENT = HYBRID
        # =====================================================

        # Example:
        #
        # "Tell me the price and amenities of Sky Edition"
        #
        # price     -> PostgreSQL
        # amenities -> brochure / pgvector
        #
        # Therefore this must be HYBRID.

        if has_structured and has_document:

            return {
                "route": "hybrid",
                "question_type": "hybrid_query",
                "structured": True,
                "documents": True,
                "hybrid": True
            }

        # =====================================================
        # PROJECT-SPECIFIC INFORMATION + DOCUMENT
        # =====================================================

        # Example:
        #
        # "Tell me about Sky Edition including its amenities"
        #
        # The project itself may not contain a structured keyword,
        # but the question is project-specific and asks for
        # brochure information.

        if has_document and has_project_indicator:

            return {
                "route": "documents",
                "question_type": "document_query",
                "structured": False,
                "documents": True,
                "hybrid": False
            }

        # =====================================================
        # DOCUMENT CHECK
        # =====================================================

        if has_document:

            return {
                "route": "documents",
                "question_type": "document_query",
                "structured": False,
                "documents": True,
                "hybrid": False
            }

        # =====================================================
        # STRUCTURED CHECK
        # =====================================================

        if has_structured:

            return {
                "route": "structured",
                "question_type": "structured_query",
                "structured": True,
                "documents": False,
                "hybrid": False
            }

        # =====================================================
        # DEFAULT
        # =====================================================

        return {
            "route": "hybrid",
            "question_type": "general",
            "structured": True,
            "documents": True,
            "hybrid": True
        }