import re


class TextChunker:
    """
    Structure-aware text chunker.

    Tries to preserve paragraphs and sentences instead of
    cutting the document at arbitrary character positions.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # =========================================================
    # SENTENCE SPLITTER
    # =========================================================

    def _split_sentences(self, text: str):
        """
        Split text approximately at sentence boundaries.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =========================================================
    # MAIN CHUNKING
    # =========================================================

    def chunk(self, text: str):

        if not text or not text.strip():
            return []

        print("\n================================")
        print("TEXT CHUNKING")
        print("================================")

        print(f"Chunk size: {self.chunk_size}")
        print(f"Chunk overlap: {self.chunk_overlap}")

        # -----------------------------------------------------
        # Split document into paragraphs
        # -----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n+",
            text
        )

        paragraphs = [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

        chunks = []
        current_chunk = ""

        # -----------------------------------------------------
        # Build chunks from paragraphs
        # -----------------------------------------------------

        for paragraph in paragraphs:

            # Paragraph fits in current chunk
            if (
                len(current_chunk)
                + len(paragraph)
                + 2
                <= self.chunk_size
            ):

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

                continue

            # -------------------------------------------------
            # Save current chunk
            # -------------------------------------------------

            if current_chunk:

                chunks.append(
                    current_chunk.strip()
                )

            # -------------------------------------------------
            # Paragraph itself is too large
            # -------------------------------------------------

            if len(paragraph) > self.chunk_size:

                sentences = self._split_sentences(
                    paragraph
                )

                current_chunk = ""

                for sentence in sentences:

                    # Sentence fits
                    if (
                        len(current_chunk)
                        + len(sentence)
                        + 1
                        <= self.chunk_size
                    ):

                        if current_chunk:
                            current_chunk += " "

                        current_chunk += sentence

                    else:

                        if current_chunk:

                            chunks.append(
                                current_chunk.strip()
                            )

                        current_chunk = sentence

            else:

                current_chunk = paragraph

        # -----------------------------------------------------
        # Add final chunk
        # -----------------------------------------------------

        if current_chunk:

            chunks.append(
                current_chunk.strip()
            )

        # -----------------------------------------------------
        # Add overlap
        # -----------------------------------------------------

        final_chunks = []

        for index, chunk in enumerate(chunks):

            if index == 0:

                final_chunks.append(chunk)

                continue

            previous_chunk = chunks[index - 1]

            overlap_text = previous_chunk[
                -self.chunk_overlap:
            ]

            combined_chunk = (
                overlap_text
                + "\n"
                + chunk
            )

            final_chunks.append(
                combined_chunk.strip()
            )

        print(
            f"Created {len(final_chunks)} chunks "
            f"from {len(text)} characters."
        )

        return final_chunks