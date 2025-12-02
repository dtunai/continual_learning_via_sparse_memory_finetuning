"""data loaders for continual learning tasks."""

from .triviaqa_loader import TriviaQAFactDataset, create_triviaqa_dataloader
from .simpleqa_loader import SimpleQADocumentDataset, create_simpleqa_dataloader
from .background_corpus_loader import (
    BackgroundCorpusDataset,
    create_background_dataloader,
)

__all__ = [
    "TriviaQAFactDataset",
    "create_triviaqa_dataloader",
    "SimpleQADocumentDataset",
    "create_simpleqa_dataloader",
    "BackgroundCorpusDataset",
    "create_background_dataloader",
]
