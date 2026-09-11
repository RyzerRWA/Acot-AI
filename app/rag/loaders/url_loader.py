import requests
from bs4 import BeautifulSoup


def load_url_content(url: str) -> str:
    """
    Fetch a web page and extract readable text.
    """

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        return text

    except requests.RequestException as error:

        print(
            f"Error loading URL: {url}"
        )

        print(error)

        return ""
