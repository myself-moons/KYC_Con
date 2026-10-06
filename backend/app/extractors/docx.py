"""Extract DOCX paragraphs and tables in body order as a logical page."""

from io import BytesIO

from docx import Document as WordDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import ExtractionErrorCode
from app.extractors.base import DocumentExtractor, ExtractionResult
from app.models.document import Document, DocumentStatus
from app.models.extraction import ExtractionError, PageResult


class DOCXExtractor(DocumentExtractor):
    """Read text content from Word body elements without inventing page layout."""

    def extract(self, document: Document, content: bytes) -> ExtractionResult:
        """Extract all top-level paragraphs and tables in their source order."""
        if not content:
            return self._failed(
                document.document_id,
                ExtractionErrorCode.NO_TEXT_DETECTED,
                "The DOCX document is empty.",
            )
        try:
            word_document = WordDocument(BytesIO(content))
        except Exception:
            return self._failed(
                document.document_id,
                ExtractionErrorCode.DOCX_PARSE_FAILED,
                "The DOCX document is corrupted or cannot be parsed.",
            )

        body_text: list[str] = []
        for child in word_document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, word_document).text.strip()
                if paragraph:
                    body_text.append(paragraph)
            elif isinstance(child, CT_Tbl):
                table = Table(child, word_document)
                for row in table.rows:
                    cell_text = [cell.text.strip() for cell in row.cells]
                    if any(cell_text):
                        body_text.append("\t".join(cell_text))

        text = "\n".join(body_text).strip()
        if not text:
            return self._failed(
                document.document_id,
                ExtractionErrorCode.NO_TEXT_DETECTED,
                "The DOCX document contains no extractable text.",
            )

        page = PageResult(
            document_id=document.document_id,
            page_number=1,
            text=text,
            needs_ocr=False,
            extraction_method="docx",
            word_boxes=[],
        )
        return ExtractionResult(
            document_id=document.document_id,
            status=DocumentStatus.SUCCESS,
            pages=[page],
            extraction_method="docx",
        )

    @staticmethod
    def _failed(
        document_id: str, code: ExtractionErrorCode, message: str
    ) -> ExtractionResult:
        return ExtractionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            error=ExtractionError(
                code=code,
                message=message,
                document_id=document_id,
                retryable=False,
            ),
            extraction_method="docx",
        )
