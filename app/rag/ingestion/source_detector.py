from urllib.parse import urlparse


class SourceDetector:

    @staticmethod
    def detect(url: str) -> str:

        if not url:
            return "unknown"

        url = url.strip().lower()

        # PDF
        if url.endswith(".pdf"):
            return "pdf"

        # JSON API
        if url.endswith(".json"):
            return "json"

        # Common API indicators
        if "/api/" in url:
            return "api"

        if "api." in url:
            return "api"

        # Otherwise treat as web page
        parsed = urlparse(url)

        if parsed.scheme in ("http", "https"):
            return "html"

        return "unknown"