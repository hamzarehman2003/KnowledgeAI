from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.dependencies import reset_lexical_index
from app.core.config import get_settings
from app.documents.extraction import EncryptedPdfError, PdfExtractionError, extract_pdf_pages
from app.documents.validation import InvalidDocumentError, validate_pdf_upload
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingService
from app.rag.ingestion import build_document_chunks
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

router = APIRouter(prefix="/documents", tags=["documents"])


class UploadDocumentResponse(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    status: str


class ExtractedPageResponse(BaseModel):
    page_number: int
    text: str


class ExtractDocumentResponse(BaseModel):
    document_id: str
    extracted_page_count: int
    extracted_pages: list[ExtractedPageResponse]


class ChunkResponse(BaseModel):
    page_number: int
    chunk_index: int
    text: str
    token_count: int


class ChunkDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    chunks: list[ChunkResponse]


class IndexDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    embedding_model: str
    collection: str
    status: str


@router.post("/upload", response_model=UploadDocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> UploadDocumentResponse:
    """Accept and persist one validated PDF.

    Text extraction, chunking, and vector indexing are deliberately separate
    future stages; this endpoint is responsible only for receiving the file.
    """
    settings = get_settings()
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read(max_size_bytes + 1)

    try:
        validated = validate_pdf_upload(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            max_size_mb=settings.max_upload_size_mb,
        )
    except InvalidDocumentError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    finally:
        await file.close()

    document_id = str(uuid4())
    upload_directory = Path(settings.upload_directory)
    upload_directory.mkdir(parents=True, exist_ok=True)
    (upload_directory / f"{document_id}.pdf").write_bytes(validated.content)

    return UploadDocumentResponse(
        document_id=document_id,
        filename=validated.filename,
        size_bytes=len(validated.content),
        status="uploaded",
    )


@router.get("/{document_id}/extract", response_model=ExtractDocumentResponse)
def extract_document(document_id: UUID) -> ExtractDocumentResponse:
    """Extract page-aware text from one previously uploaded PDF."""
    settings = get_settings()
    pdf_path = Path(settings.upload_directory) / f"{document_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        pages = extract_pdf_pages(pdf_path)
    except EncryptedPdfError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except PdfExtractionError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    return ExtractDocumentResponse(
        document_id=str(document_id),
        extracted_page_count=len(pages),
        extracted_pages=[
            ExtractedPageResponse(page_number=page.page_number, text=page.text) for page in pages
        ],
    )


@router.get("/{document_id}/chunks", response_model=ChunkDocumentResponse)
def chunk_document(document_id: UUID) -> ChunkDocumentResponse:
    """Normalize and split a previously uploaded PDF into inspectable chunks."""
    settings = get_settings()
    pdf_path = Path(settings.upload_directory) / f"{document_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        chunks = build_document_chunks(
            pdf_path,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    except (EncryptedPdfError, PdfExtractionError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return ChunkDocumentResponse(
        document_id=str(document_id),
        chunk_count=len(chunks),
        chunks=[
            ChunkResponse(
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ],
    )


@router.post("/{document_id}/index", response_model=IndexDocumentResponse)
def index_document(document_id: UUID) -> IndexDocumentResponse:
    """Embed one document's chunks and replace its vectors in ChromaDB."""
    settings = get_settings()
    pdf_path = Path(settings.upload_directory) / f"{document_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        chunks = build_document_chunks(
            pdf_path,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    except (EncryptedPdfError, PdfExtractionError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    try:
        embedder = OllamaEmbeddingService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_store = ChromaVectorStore(
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection_name=settings.chroma_collection_name,
        )
        vector_store.replace_document_chunks(
            document_id=str(document_id),
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=settings.ollama_embedding_model,
        )
    except (EmbeddingError, VectorStoreError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding or vector storage service is unavailable.",
        ) from error

    # New vectors are searchable immediately; the lexical index must follow.
    reset_lexical_index()

    return IndexDocumentResponse(
        document_id=str(document_id),
        chunk_count=len(chunks),
        embedding_model=settings.ollama_embedding_model,
        collection=settings.chroma_collection_name,
        status="indexed",
    )
