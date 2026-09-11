import re


class TextCleaner:

    def clean(self, text: str) -> str:

        if not text:
            return ""

        print("\n================================")
        print("TEXT CLEANING")
        print("================================")

        original_length = len(text)

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove excessive blank lines
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Remove leading/trailing spaces from lines
        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        text = "\n".join(lines)

        # Remove excessive whitespace again
        text = re.sub(r" {2,}", " ", text)

        # Final cleanup
        text = text.strip()

        print(f"Original characters: {original_length}")
        print(f"Cleaned characters: {len(text)}")

        return text