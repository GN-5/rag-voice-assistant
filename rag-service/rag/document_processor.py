from pathlib import Path
import fitz
import docx


class DocumentProcessor:
    def extract(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".docx":
            return self._extract_docx(path)
        elif suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _extract_pdf(self, path: Path) -> str:
        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return self._normalize("\n".join(pages))
    
    def _extract_docx(self, path: Path) -> str:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        return self._normalize("\n".join(paragraphs))
    
    def _normalize(self, text: str) -> str:
        import unicodedata, re
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
