"""Citation-grounded hybrid RAG service."""

from .config import Settings, settings
from .pipeline import Answer, Citation, Document, RAGPipeline

__version__ = "1.0.0"
__all__ = ["Answer", "Citation", "Document", "RAGPipeline", "Settings", "settings"]
