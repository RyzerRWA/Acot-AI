class InvestmentAnalyzer:

    def analyze(self, community_summary: dict):
        """
        Analyze structured community data and generate
        deterministic investment signals.

        Investment risk and data confidence are calculated
        separately.

        Important:
        This analysis is based only on the available dataset
        and should not be considered financial advice.
        """

        # =============================================
        # GET BASIC DATA
        # =============================================

        records_found = community_summary.get(
            "records_found",
            0
        )

        rental_yield_data = community_summary.get(
            "rental_yield_percent",
            {}
        )

        property_statuses = community_summary.get(
            "property_statuses",
            []
        )

        # NEW:
        # Actual distribution from Structured Aggregator
        property_status_distribution = community_summary.get(
            "property_status_distribution",
            {}
        )

        property_types = community_summary.get(
            "property_types",
            []
        )

        market_positions = community_summary.get(
            "market_positions",
            []
        )

        # =============================================
        # RENTAL YIELD ANALYSIS
        # =============================================

        yield_analysis = self._analyze_rental_yield(
            rental_yield_data
        )

        # =============================================
        # DEVELOPMENT EXPOSURE ANALYSIS
        #
        # Uses actual property status distribution.
        # =============================================

        development_analysis = (
            self._analyze_development_exposure(
                property_status_distribution,
                records_found
            )
        )

        # =============================================
        # DATA CONFIDENCE
        #
        # Confidence DOES NOT affect investment score.
        # =============================================

        confidence_analysis = (
            self._calculate_confidence(
                records_found
            )
        )

        # =============================================
        # INVESTMENT SCORE
        # =============================================

        scoring = self._calculate_score(
            yield_analysis=yield_analysis,
            development_analysis=development_analysis
        )

        # =============================================
        # POSITIVE SIGNALS
        # =============================================

        positive_signals = []

        if yield_analysis["assessment"] == "strong":

            positive_signals.append({
                "factor": "rental_yield",
                "assessment": "strong",
                "value": yield_analysis[
                    "average_yield_percent"
                ],
                "message": yield_analysis[
                    "message"
                ]
            })

        elif yield_analysis["assessment"] == "moderate":

            positive_signals.append({
                "factor": "rental_yield",
                "assessment": "moderate",
                "value": yield_analysis[
                    "average_yield_percent"
                ],
                "message": yield_analysis[
                    "message"
                ]
            })

        # Development exposure positive signal

        if development_analysis["risk_level"] == "low":

            positive_signals.append({
                "factor": "development_exposure",
                "assessment": "low",
                "value": development_analysis[
                    "development_exposure_percent"
                ],
                "message": development_analysis[
                    "message"
                ]
            })

        # =============================================
        # INVESTMENT RISKS
        # =============================================

        investment_risks = []

        if development_analysis["risk_level"] in [
            "high",
            "moderate"
        ]:

            investment_risks.append({
                "factor": "development_exposure",
                "risk_level": development_analysis[
                    "risk_level"
                ],
                "development_exposure_percent": (
                    development_analysis[
                        "development_exposure_percent"
                    ]
                ),
                "message": development_analysis[
                    "message"
                ]
            })

        # Small sample risk

        if records_found < 10:

            investment_risks.append({
                "factor": "sample_size",
                "risk_level": "data_limitation",
                "message": (
                    "The analysis is based on a limited "
                    "number of available records."
                )
            })

        # =============================================
        # MARKET CONTEXT
        #
        # Does not automatically affect score.
        # =============================================

        market_context = {

            "property_types": property_types,

            "market_positions": market_positions,

            "amenities": community_summary.get(
                "amenities",
                []
            ),

            "landmarks": community_summary.get(
                "landmarks",
                []
            )
        }

        # =============================================
        # DATA LIMITATIONS
        # =============================================

        limitations = self._data_limitations(
            records_found
        )

        # =============================================
        # ACOT SCORE + AI RECOMMENDATION
        # =============================================

        acot_score = self._calculate_acot_score(
            scoring.get("score", 0),
            yield_analysis=yield_analysis,
            development_analysis=development_analysis
        )

        ai_recommendation = self._build_ai_recommendation(
            verdict=scoring.get("verdict", "Neutral"),
            confidence=confidence_analysis.get("level", "Low"),
            yield_analysis=yield_analysis,
            development_analysis=development_analysis
        )

        # =============================================
        # FINAL RESULT
        # =============================================

        return {

            # -----------------------------------------
            # BASIC INFORMATION
            # -----------------------------------------

            "community": community_summary.get(
                "community"
            ),

            "district": community_summary.get(
                "district"
            ),

            "records_found": records_found,

            # -----------------------------------------
            # INVESTMENT VERDICT
            # -----------------------------------------

            "investment_verdict": scoring[
                "verdict"
            ],

            "investment_score": scoring[
                "score"
            ],

            # Frontend-ready 0-100 ACOT score.
            # This is a normalized representation of the existing
            # investment score; no new market assumptions are added.
            "acot_score": acot_score,

            # Recommendation generated from the deterministic ACOT
            # investment signals, without an additional Gemini call.
            "ai_recommendation": ai_recommendation,

            "score_breakdown": scoring[
                "score_breakdown"
            ],

            # -----------------------------------------
            # CONFIDENCE
            # -----------------------------------------

            "confidence": confidence_analysis[
                "level"
            ],

            "confidence_reason": confidence_analysis[
                "message"
            ],

            # -----------------------------------------
            # ANALYSIS
            # -----------------------------------------

            "rental_yield_analysis": yield_analysis,

            "development_analysis": development_analysis,

            # -----------------------------------------
            # SIGNALS
            # -----------------------------------------

            "positive_signals": positive_signals,

            "investment_risks": investment_risks,

            # -----------------------------------------
            # MARKET CONTEXT
            # -----------------------------------------

            "market_context": market_context,

            # -----------------------------------------
            # LIMITATIONS
            # -----------------------------------------

            "data_limitations": limitations
        }

    # =================================================
    # ACOT SCORE
    # =================================================

    def _calculate_acot_score(
        self,
        investment_score,
        yield_analysis,
        development_analysis
    ):
        """
        Normalize the existing ACOT investment score to 0-100.

        Existing investment_score range:
            -3 .. +3

        Normalized ACOT score:
            0 .. 100

        The normalization does not introduce any new market data.
        """

        if not isinstance(investment_score, (int, float)):
            return None

        has_signal = (
            bool(yield_analysis.get("available"))
            or bool(development_analysis.get("statuses"))
        )

        if not has_signal:
            return None

        score = max(-3.0, min(3.0, float(investment_score)))
        normalized = ((score + 3.0) / 6.0) * 100.0

        return {
            "value": int(round(normalized)),
            "max": 100,
            "source_score": score,
            "formula": "normalized from existing -3..+3 investment score",
        }

    # =================================================
    # AI RECOMMENDATION
    # =================================================

    def _build_ai_recommendation(
        self,
        verdict,
        confidence,
        yield_analysis,
        development_analysis
    ):
        """
        Build a concise recommendation from the existing ACOT signals.

        This intentionally avoids a second LLM/Gemini call, so adding the
        recommendation does not increase API usage or latency.
        """

        verdict_text = str(verdict or "Neutral")
        confidence_text = str(confidence or "Low")

        if verdict_text == "Attractive":
            recommendation = (
                "Available investment signals are positive. "
                "Validate pricing, rental assumptions, and current market data "
                "before making a final decision."
            )

        elif verdict_text == "Moderately Attractive":
            recommendation = (
                "Available signals are moderately positive. "
                "Further validation of rental performance, pricing, and project "
                "status is recommended."
            )

        elif verdict_text == "Moderately Risky":
            recommendation = (
                "Available signals indicate caution. Review development exposure "
                "and validate rental and market assumptions before proceeding."
            )

        elif verdict_text == "High Risk":
            recommendation = (
                "Available signals indicate elevated risk. Additional property, "
                "rental, and market due diligence is recommended."
            )

        else:
            recommendation = (
                "Available evidence is mixed or limited. Gather additional rental "
                "and market data before relying on the current assessment."
            )

        if confidence_text == "Low":
            recommendation += (
                " Confidence is low because the available sample is limited."
            )

        return {
            "text": recommendation,
            "verdict": verdict_text,
            "confidence": confidence_text,
        }


    # =================================================
    # RENTAL YIELD ANALYSIS
    # =================================================

    def _analyze_rental_yield(
        self,
        rental_yield_data
    ):

        average_yield = rental_yield_data.get(
            "average"
        )

        # ---------------------------------------------
        # NO RENTAL YIELD DATA
        # ---------------------------------------------

        if average_yield is None:

            return {

                "available": False,

                "average_yield_percent": None,

                "assessment": "unknown",

                "score": 0,

                "message": (
                    "Rental yield data is not available "
                    "for the current sample."
                )
            }

        # ---------------------------------------------
        # STRONG RENTAL YIELD
        # ---------------------------------------------

        if average_yield >= 7:

            return {

                "available": True,

                "average_yield_percent": average_yield,

                "assessment": "strong",

                "score": 3,

                "message": (
                    f"The available sample shows an "
                    f"estimated gross rental yield of "
                    f"{average_yield}%."
                )
            }

        # ---------------------------------------------
        # MODERATE RENTAL YIELD
        # ---------------------------------------------

        elif average_yield >= 5:

            return {

                "available": True,

                "average_yield_percent": average_yield,

                "assessment": "moderate",

                "score": 2,

                "message": (
                    f"The available sample shows an "
                    f"estimated gross rental yield of "
                    f"{average_yield}%."
                )
            }

        # ---------------------------------------------
        # LOW RENTAL YIELD
        # ---------------------------------------------

        else:

            return {

                "available": True,

                "average_yield_percent": average_yield,

                "assessment": "low",

                "score": 0,

                "message": (
                    f"The available sample shows an "
                    f"estimated gross rental yield of "
                    f"{average_yield}%."
                )
            }

    # =================================================
    # DEVELOPMENT EXPOSURE ANALYSIS
    # =================================================

    def _analyze_development_exposure(
        self,
        property_status_distribution,
        records_found
    ):
        """
        Analyze property development exposure using the
        actual distribution of retrieved records.

        Example:

        {
            "ready": 10,
            "off_plan": 2,
            "under_construction": 3
        }

        Important:
        Percentages describe the current retrieved sample,
        NOT the entire real-world community market.
        """

        # ---------------------------------------------
        # NO DATA
        # ---------------------------------------------

        if not property_status_distribution:

            return {

                "statuses": [],

                "status_distribution": {},

                "ready_percent": 0.0,

                "off_plan_percent": 0.0,

                "under_construction_percent": 0.0,

                "development_exposure_percent": 0.0,

                "risk_level": "unknown",

                "score": 0,

                "message": (
                    "Property development status data "
                    "is not available for the current "
                    "sample."
                )
            }

        # ---------------------------------------------
        # NORMALIZE DISTRIBUTION
        # ---------------------------------------------

        normalized_distribution = {}

        for status, count in property_status_distribution.items():

            normalized_status = str(
                status
            ).lower().strip()

            try:

                normalized_count = int(count)

            except (
                ValueError,
                TypeError
            ):

                normalized_count = 0

            normalized_distribution[
                normalized_status
            ] = normalized_count

        # ---------------------------------------------
        # CALCULATE TOTAL
        # ---------------------------------------------

        total_properties = sum(
            normalized_distribution.values()
        )

        # Fallback

        if total_properties <= 0:

            total_properties = records_found

        if total_properties <= 0:

            return {

                "statuses": [],

                "status_distribution": normalized_distribution,

                "ready_percent": 0.0,

                "off_plan_percent": 0.0,

                "under_construction_percent": 0.0,

                "development_exposure_percent": 0.0,

                "risk_level": "unknown",

                "score": 0,

                "message": (
                    "Property status counts are not "
                    "sufficient for development analysis."
                )
            }

        # ---------------------------------------------
        # GET STATUS COUNTS
        # ---------------------------------------------

        ready_count = normalized_distribution.get(
            "ready",
            0
        )

        off_plan_count = normalized_distribution.get(
            "off_plan",
            0
        )

        under_construction_count = (
            normalized_distribution.get(
                "under_construction",
                0
            )
        )

        # ---------------------------------------------
        # CALCULATE PERCENTAGES
        # ---------------------------------------------

        ready_percent = round(
            (ready_count / total_properties) * 100,
            2
        )

        off_plan_percent = round(
            (off_plan_count / total_properties) * 100,
            2
        )

        under_construction_percent = round(
            (
                under_construction_count
                / total_properties
            ) * 100,
            2
        )

        development_exposure_percent = round(
            (
                (
                    off_plan_count
                    + under_construction_count
                )
                / total_properties
            ) * 100,
            2
        )

        statuses = list(
            normalized_distribution.keys()
        )

        # ---------------------------------------------
        # HIGH DEVELOPMENT EXPOSURE
        #
        # More than 75% of retrieved sample
        # ---------------------------------------------

        if development_exposure_percent > 75:

            risk_level = "high"

            score = -3

        # ---------------------------------------------
        # HIGH-MODERATE EXPOSURE
        #
        # 51% to 75%
        # ---------------------------------------------

        elif development_exposure_percent > 50:

            risk_level = "high"

            score = -2

        # ---------------------------------------------
        # MODERATE EXPOSURE
        #
        # 26% to 50%
        # ---------------------------------------------

        elif development_exposure_percent > 25:

            risk_level = "moderate"

            score = -1

        # ---------------------------------------------
        # LOW EXPOSURE
        #
        # 0% to 25%
        # ---------------------------------------------

        else:

            risk_level = "low"

            score = 0

        # ---------------------------------------------
        # CREATE MESSAGE
        # ---------------------------------------------

        message = (

            f"Within the current retrieved sample, "
            f"{development_exposure_percent}% of records "
            f"are development-stage properties "
            f"({off_plan_percent}% off-plan and "
            f"{under_construction_percent}% "
            f"under construction)."
        )

        return {

            "statuses": statuses,

            "status_distribution": (
                normalized_distribution
            ),

            "sample_size": total_properties,

            "ready_percent": ready_percent,

            "off_plan_percent": off_plan_percent,

            "under_construction_percent": (
                under_construction_percent
            ),

            "development_exposure_percent": (
                development_exposure_percent
            ),

            "risk_level": risk_level,

            "score": score,

            "message": message
        }

    # =================================================
    # DATA CONFIDENCE
    #
    # CONFIDENCE DOES NOT AFFECT INVESTMENT SCORE
    # =================================================

    def _calculate_confidence(
        self,
        records_found
    ):

        # ---------------------------------------------
        # HIGH CONFIDENCE
        # ---------------------------------------------

        if records_found >= 30:

            return {

                "level": "High",

                "message": (
                    f"The analysis is based on "
                    f"{records_found} available "
                    f"records."
                )
            }

        # ---------------------------------------------
        # MEDIUM CONFIDENCE
        # ---------------------------------------------

        elif records_found >= 10:

            return {

                "level": "Medium",

                "message": (
                    f"The analysis is based on "
                    f"{records_found} available "
                    f"records, providing moderate "
                    f"sample coverage."
                )
            }

        # ---------------------------------------------
        # LOW CONFIDENCE
        # ---------------------------------------------

        else:

            return {

                "level": "Low",

                "message": (
                    f"The analysis is based on only "
                    f"{records_found} available "
                    f"records, limiting confidence "
                    f"in the assessment."
                )
            }

    # =================================================
    # INVESTMENT SCORE
    #
    # CONFIDENCE IS NOT INCLUDED
    # =================================================

    def _calculate_score(

        self,

        yield_analysis,

        development_analysis
    ):

        # ---------------------------------------------
        # RENTAL YIELD SCORE
        # ---------------------------------------------

        yield_score = yield_analysis.get(
            "score",
            0
        )

        # ---------------------------------------------
        # DEVELOPMENT EXPOSURE SCORE
        # ---------------------------------------------

        development_score = development_analysis.get(
            "score",
            0
        )

        # ---------------------------------------------
        # TOTAL SCORE
        # ---------------------------------------------

        total_score = (

            yield_score

            +

            development_score
        )

        # ---------------------------------------------
        # INVESTMENT VERDICT
        # ---------------------------------------------

        if total_score >= 3:

            verdict = "Attractive"

        elif total_score >= 1:

            verdict = "Moderately Attractive"

        elif total_score <= -3:

            verdict = "High Risk"

        elif total_score <= -1:

            verdict = "Moderately Risky"

        else:

            verdict = "Neutral"

        return {

            "score": total_score,

            "verdict": verdict,

            "score_breakdown": {

                "rental_yield": yield_score,

                "development_exposure": (
                    development_score
                )
            }
        }

    # =================================================
    # DATA LIMITATIONS
    # =================================================

    def _data_limitations(
        self,
        records_found
    ):

        limitations = []

        # ---------------------------------------------
        # SAMPLE SIZE
        # ---------------------------------------------

        if records_found < 10:

            limitations.append(
                "Limited number of available "
                "property records."
            )

        # ---------------------------------------------
        # SYNTHETIC DATA
        # ---------------------------------------------

        limitations.append(
            "Current analysis uses synthetic "
            "benchmark data for demonstration."
        )

        # ---------------------------------------------
        # HISTORICAL PERFORMANCE
        # ---------------------------------------------

        limitations.append(
            "Historical price performance is not "
            "available in the current analysis."
        )

        # ---------------------------------------------
        # LIVE DATA
        # ---------------------------------------------

        limitations.append(
            "Live transaction data is not "
            "available in the current analysis."
        )

        # ---------------------------------------------
        # NET YIELD LIMITATION
        # ---------------------------------------------

        limitations.append(
            "Rental yield is a gross estimate and "
            "does not include service charges, "
            "maintenance, vacancy, financing, or "
            "other ownership costs."
        )

        return limitations