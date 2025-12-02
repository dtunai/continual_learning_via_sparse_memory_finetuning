"""
simpleqa document learning dataset

loads wikipedia documents for simpleqa questions, chunks into paragraphs,
generates augmentations. each batch contains augmentations of one chunk
"""

from logging import getLogger
from typing import Optional, List, Dict
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = getLogger(__name__)


class SimpleQADocumentDataset(Dataset):
    """
    dataset for learning from simpleqa wikipedia documents.
    each item is one chunk with synthetic augmentations.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_questions: int = 100,
        augmentations_per_chunk: int = 64,
        max_seq_length: int = 512,
        chunk_size: int = 500,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_questions = num_questions
        self.augmentations_per_chunk = augmentations_per_chunk
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.seed = seed

        logger.info("loading simpleqa...")
        try:
            dataset = load_dataset("openai/simple-qa", split="train")
        except:
            logger.warning("simpleqa not available, using placeholder")
            dataset = []
            for i in range(num_questions):
                dataset.append(
                    {
                        "question": f"placeholder question {i}",
                        "answer": f"placeholder answer {i}",
                        "wikipedia_url": f"https://en.wikipedia.org/wiki/Placeholder_{i}",
                        "wikipedia_text": "placeholder wikipedia text. " * 100,
                    }
                )

        rng = random.Random(seed)
        if len(dataset) > num_questions:
            indices = rng.sample(range(len(dataset)), num_questions)
            self.questions = [dataset[i] for i in indices]
        else:
            self.questions = list(dataset)

        # create chunks
        self.chunks = []
        for q_idx, q_data in enumerate(self.questions):
            for chunk in self._chunk_document(q_data):
                self.chunks.append(
                    {
                        "question_id": q_idx,
                        "chunk_text": chunk,
                        "question": q_data.get("question", ""),
                        "answer": q_data.get("answer", ""),
                    }
                )

        logger.info(
            f"loaded {len(self.questions)} questions, {len(self.chunks)} chunks"
        )

    def _chunk_document(self, q_data: Dict) -> List[str]:
        """split document into chunks by paragraph or size."""
        if "wikipedia_text" in q_data:
            text = q_data["wikipedia_text"]
        elif "document" in q_data:
            text = q_data["document"]
        else:
            text = f"{q_data.get('question', '')} {q_data.get('answer', '')}"

        chunks = []
        current = ""

        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) < self.chunk_size:
                current += para + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = para + " "

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text[: self.chunk_size]]

    def _generate_augmentations(self, text: str, num: int) -> List[str]:
        """generate simple template-based augmentations."""
        augmentations = [text]
        templates = [
            text,
            f"Summary: {text}",
            f"Key information: {text}",
            f"According to the document: {text}",
            f"The text states: {text}",
            f"From the source: {text}",
        ]

        while len(augmentations) < num:
            augmentations.append(random.choice(templates))

        return augmentations[:num]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """returns batch of augmentations for one chunk."""
        chunk_data = self.chunks[idx]
        augmentations = self._generate_augmentations(
            chunk_data["chunk_text"], self.augmentations_per_chunk
        )

        encodings = self.tokenizer(
            augmentations,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
            "chunk_id": torch.tensor(idx),
            "question_id": torch.tensor(chunk_data["question_id"]),
            "question": chunk_data["question"],
            "answer": chunk_data["answer"],
        }


def create_simpleqa_dataloader(
    tokenizer: PreTrainedTokenizer,
    num_questions: int = 100,
    augmentations_per_chunk: int = 64,
    max_seq_length: int = 512,
    chunk_size: int = 500,
    seed: int = 42,
    shuffle: bool = False,
) -> DataLoader:
    """create dataloader for simpleqa document learning."""
    dataset = SimpleQADocumentDataset(
        tokenizer=tokenizer,
        num_questions=num_questions,
        augmentations_per_chunk=augmentations_per_chunk,
        max_seq_length=max_seq_length,
        chunk_size=chunk_size,
        seed=seed,
    )

    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
