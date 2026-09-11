import requests


class URLFetcher:

    def __init__(self, timeout: int = 30):

        self.timeout = timeout

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/139.0 Safari/537.36"
            )
        }

    def fetch(self, url: str):

        if not url:
            raise ValueError("URL cannot be empty")

        print("\n================================")
        print("FETCHING URL")
        print("================================")

        print(f"URL: {url}")

        try:

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )

            response.raise_for_status()

            print(f"Status Code: {response.status_code}")
            print(
                f"Content Type: "
                f"{response.headers.get('Content-Type')}"
            )

            print(
                f"Final URL: "
                f"{response.url}"
            )

            return {
                "url": url,
                "final_url": response.url,
                "status_code": response.status_code,
                "content_type": response.headers.get(
                    "Content-Type",
                    ""
                ),
                "content": response.content
            }

        except requests.exceptions.Timeout:

            raise RuntimeError(
                f"Request timed out while fetching: {url}"
            )

        except requests.exceptions.HTTPError as e:

            raise RuntimeError(
                f"HTTP error while fetching {url}: {e}"
            )

        except requests.exceptions.RequestException as e:

            raise RuntimeError(
                f"Failed to fetch {url}: {e}"
            )