"""
triviaqa fact learning dataset

loads facts from triviaqa, converts to statements, generates paraphrases

each batch contains paraphrases of a single fact
"""

from logging import getLogger
from typing import Optional, List, Dict
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = getLogger(__name__)


class TriviaQAFactDataset(Dataset):
    """
    dataset for learning triviaqa facts in sequence.
    each item is one fact paraphrased n times to fill a batch.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_facts: int = 1000,
        paraphrases_per_fact: int = 64,
        max_seq_length: int = 64,
        split: str = "test",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_facts = num_facts
        self.paraphrases_per_fact = paraphrases_per_fact
        self.max_seq_length = max_seq_length
        self.seed = seed

        logger.info(f"loading triviaqa {split}...")
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split=split)

        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), min(num_facts, len(dataset)))
        self.facts = [dataset[i] for i in indices]

        logger.info(f"loaded {len(self.facts)} facts")

    def __len__(self) -> int:
        return len(self.facts)

    def _question_to_statement(self, question: str, answer: str) -> str:
        """convert q/a pair to statement."""
        question = question.strip()
        if question.endswith("?"):
            question = question[:-1]

        question_words = ["what", "who", "where", "when", "which", "how"]
        lower_q = question.lower()
        for qw in question_words:
            if lower_q.startswith(qw + " is "):
                question = question[len(qw) + 4 :]
                return f"{question.capitalize()} is {answer}"
            elif lower_q.startswith(qw + " was "):
                question = question[len(qw) + 5 :]
                return f"{question.capitalize()} was {answer}"
            elif lower_q.startswith(qw + " "):
                question = question[len(qw) + 1 :]
                return f"{question.capitalize()} is {answer}"

        return f"{question}. {answer}"

    def _generate_paraphrases(self, statement: str, num: int) -> List[str]:
        """generate simple template-based paraphrases."""
        paraphrases = [statement]
        templates = [
            statement,
            f"It is known that {statement.lower()}",
            f"The fact is that {statement.lower()}",
            f"According to sources, {statement.lower()}",
            f"Research shows that {statement.lower()}",
            f"Evidence indicates that {statement.lower()}",
            f"Studies confirm that {statement.lower()}",
            f"Experts agree that {statement.lower()}",
        ]

        while len(paraphrases) < num:
            paraphrases.append(random.choice(templates))

        return paraphrases[:num]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """returns batch of paraphrases for one fact."""
        fact = self.facts[idx]
        question = fact["question"]
        answer = fact["answer"]["value"]

        statement = self._question_to_statement(question, answer)
        paraphrases = self._generate_paraphrases(statement, self.paraphrases_per_fact)

        encodings = self.tokenizer(
            paraphrases,
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
            "fact_id": torch.tensor(idx),
            "question": question,
            "answer": answer,
        }


def create_triviaqa_dataloader(
    tokenizer: PreTrainedTokenizer,
    num_facts: int = 1000,
    paraphrases_per_fact: int = 64,
    max_seq_length: int = 64,
    split: str = "test",
    seed: int = 42,
    shuffle: bool = False,
) -> DataLoader:
    """create dataloader for triviaqa fact learning."""
    dataset = TriviaQAFactDataset(
        tokenizer=tokenizer,
        num_facts=num_facts,
        paraphrases_per_fact=paraphrases_per_fact,
        max_seq_length=max_seq_length,
        split=split,
        seed=seed,
    )

    # batch_size=1 because each item is already a batch of paraphrases
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
