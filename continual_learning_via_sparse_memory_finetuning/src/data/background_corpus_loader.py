"""
background corpus dataset

loads data from dclm/fineweb/wikitext for collecting memory index statistics
"""

from logging import getLogger
from typing import Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = getLogger(__name__)


class BackgroundCorpusDataset(Dataset):
    """
    dataset for background corpus (dclm, fineweb, etc).
    used to collect memory index statistics for tf-idf.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        corpus_name: str = "dclm",
        max_seq_length: int = 512,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.corpus_name = corpus_name
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.num_samples = num_samples

        logger.info(f"loading {corpus_name}...")

        if corpus_name == "dclm":
            try:
                self.dataset = load_dataset(
                    "mlfoundations/dclm-baseline-1.0",
                    split="train",
                    streaming=True,
                )
                if num_samples:
                    self.dataset = self.dataset.take(num_samples)
            except:
                logger.warning("dclm not available, using fallback")
                self.dataset = self._create_fallback(num_samples)

        elif corpus_name == "fineweb":
            try:
                self.dataset = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    split="train",
                    streaming=True,
                )
                if num_samples:
                    self.dataset = self.dataset.take(num_samples)
            except:
                logger.warning("fineweb not available, using fallback")
                self.dataset = self._create_fallback(num_samples)

        else:
            logger.warning(f"unknown corpus {corpus_name}, using fallback")
            self.dataset = self._create_fallback(num_samples)

        logger.info(f"loaded {corpus_name}")

    def _create_fallback(self, num_samples: Optional[int]):
        """fallback to wikitext or dummy data."""
        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
            return dataset
        except:
            logger.warning("creating dummy data")
            return [
                {"text": f"background sample {i}. " * 50}
                for i in range(num_samples or 1000)
            ]

    def __len__(self) -> int:
        if self.num_samples:
            return self.num_samples
        try:
            return len(self.dataset)
        except:
            return 1000000

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            if isinstance(self.dataset, list):
                sample = self.dataset[idx]
            else:
                for i, item in enumerate(self.dataset):
                    if i == idx:
                        sample = item
                        break
        except:
            sample = {"text": "background sample."}

        text = sample.get("text", "") or sample.get("content", "")

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def create_background_dataloader(
    tokenizer: PreTrainedTokenizer,
    corpus_name: str = "dclm",
    batch_size: int = 64,
    max_seq_length: int = 512,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """create dataloader for background corpus."""
    dataset = BackgroundCorpusDataset(
        tokenizer=tokenizer,
        corpus_name=corpus_name,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        seed=seed,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
