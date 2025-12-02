# Sparse Memory Finetuning

Reference implementation of "Continual Learning via Sparse Memory Finetuning" (Lin et al., 2025).

Core algorithms + configurations provided.

## Install

```bash
uv venv --python 3.12; uv venv --activate;

cd continual_learning_via_sparse_memory_finetuning; uv pip install -e .
```

## Directory structure

```
src/sparse_finetuning/
├── tfidf_ranker.py      # TF-IDF ranking + background tracking
├── gradient_mask.py     # Sparse gradient updates via hooks
└── sparse_finetuner.py  # Coordinates everything

src/data/
├── triviaqa_loader.py
├── simpleqa_loader.py
└── background_corpus_loader.py
```

## Example

```python
from src.sparse_finetuning import SparseMemoryFinetuner, SparseMemoryFinetuningArgs

args = SparseMemoryFinetuningArgs(
    enabled=True,
    top_t=500,
    use_idf=True,
)

# pass the memory VALUES parameter (not keys)
finetuner = SparseMemoryFinetuner(args, model.memory_values)

# collect background indices first (1000 batches from DCLM or similar)
finetuner.background_tracker.start_collection()
for batch in background_loader:
    indices = forward_and_get_indices(model, batch)
    finetuner.background_tracker.add_batch_indices(indices)
finetuner.save_background_indices("bg_indices.pt")

# training
finetuner.load_background_indices("bg_indices.pt")
for batch in train_loader:
    indices = forward_and_get_indices(model, batch)
    finetuner.update_trainable_indices(indices)  # selects top-t by TF-IDF

    loss.backward()  # only top-t get gradients
    optimizer.step()
```

1. **Get memory indices during forward pass**:
    - modify your memory layer to return which indices were accessed (the top-k from the key lookup)
    - return indices too

2. **Pass the values parameter**:
    - pass the `nn.Parameter` for memory values V, not keys K

example for a typical memory layer:

```python
# inside your memory layer forward:
I = topk(K @ query, k=32)  # <- capture these indices
output = softmax(K[I] @ query) @ V[I]
return output, I  # return indices too
```

- Fact learning (TriviaQA): `top_t=500`, `lr=2.0`, SGD
- Document learning (SimpleQA): `top_t=10000`
- Background: 1000 batches from DCLM
- Model: 1.3B base + 1M memory keys

## Citation

```bibtex
@article{lin2025continual,
  title={Continual Learning via Sparse Memory Finetuning},
  author={Lin, Jessy and others},
  year={2025}
}
```
