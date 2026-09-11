import io
import json

from bs4 import BeautifulSoup
from pypdf import PdfReader


class ContentExtractor:

    def extract(
        self,
        content: bytes,
        content_type: str,
        source_type: str
    ) -> str:

        if not content:
            raise ValueError("Content is empty")

        print("\n================================")
        print("CONTENT EXTRACTION")
        print("================================")

        print(f"Source Type: {source_type}")
        print(f"Content Type: {content_type}")

        # =====================================================
        # PDF
        # =====================================================

        if source_type == "pdf":

            return self._extract_pdf(content)

        # =====================================================
        # JSON / API
        # =====================================================

        if source_type in ("json", "api"):

            return self._extract_json(content)

        # =====================================================
        # HTML
        # =====================================================

        if source_type == "html":

            return self._extract_html(content)

        # =====================================================
        # FALLBACK
        # =====================================================

        return self._extract_fallback(content)

    # =========================================================
    # PDF EXTRACTION
    # =========================================================

    def _extract_pdf(self, content: bytes) -> str:

        print("Extracting PDF text...")

        pdf_file = io.BytesIO(content)

        reader = PdfReader(pdf_file)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text() or ""

                if text.strip():

                    pages.append(
                        f"\n--- PAGE {page_number} ---\n"
                        f"{text}"
                    )

            except Exception as e:

                print(
                    f"Warning: Could not extract "
                    f"page {page_number}: {e}"
                )

        result = "\n".join(pages).strip()

        print(
            f"Extracted {len(result)} characters "
            f"from {len(reader.pages)} PDF pages."
        )

        return result

    # =========================================================
    # JSON / API EXTRACTION
    # =========================================================

    def _extract_json(self, content: bytes) -> str:

        print("Extracting JSON/API content...")

        try:

            text = content.decode(
                "utf-8",
                errors="replace"
            )

            data = json.loads(text)

            formatted = json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )

            print(
                f"Extracted {len(formatted)} characters "
                f"from JSON/API response."
            )

            return formatted

        except json.JSONDecodeError:

            print(
                "Response is not valid JSON. "
                "Using raw text fallback."
            )

            return content.decode(
                "utf-8",
                errors="replace"
            )

    # =========================================================
    # HTML EXTRACTION
    # =========================================================

    def _extract_html(self, content: bytes) -> str:

        print("Extracting HTML text...")

        html = content.decode(
            "utf-8",
            errors="replace"
        )

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # Remove elements that usually contain
        # navigation, scripts, styles, etc.

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer"
            ]
        ):

            element.decompose()

        text = soup.get_text(
            separator="\n"
        )

        lines = []

        for line in text.splitlines():

            cleaned = " ".join(
                line.split()
            )

            if cleaned:

                lines.append(cleaned)

        result = "\n".join(lines)

        print(
            f"Extracted {len(result)} characters "
            f"from HTML."
        )

        return result

    # =========================================================
    # FALLBACK
    # =========================================================

    def _extract_fallback(
        self,
        content: bytes
    ) -> str:

        print("Using raw text fallback...")

        return content.decode(
            "utf-8",
            errors="replace"
        )