# BERT vs PubMedBERT on PubMedQA

Assignment: BERT variant comparison. I fine-tuned a general-domain BERT and a biomedical BERT on the
same task and compared them.

- Dataset: PubMedQA, `pqa_labeled` config (Hugging Face: `qiaojin/PubMedQA`, the current location of `pubmed_qa`)
- Models: `bert-base-uncased` and `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- Task: given a question and a PubMed abstract (without its conclusion), answer yes / no / maybe
- Metrics: accuracy, precision, recall, F1 (macro and weighted)

## Files

| File | What it is |
|---|---|
| `BERT_vs_PubMedBERT_PubMedQA.ipynb` | Main notebook with code, outputs, plots and analysis |
| `BERT_vs_PubMedBERT_PubMedQA.html` | The same notebook exported to HTML, for reading without Jupyter |
| `requirements.txt` | Package versions I used |
| `saved_outputs/results_per_run.csv` | Test metrics for every run (2 models x 3 seeds) |
| `saved_outputs/history_*.json` | Validation metrics per epoch for every run |
| `saved_outputs/test_probs_*.npy` | Test-set probabilities for every run (used in the error analysis) |

The fine-tuned model weights (about 440 MB each) are not included. They are saved to `outputs/best_<model>`
when the notebook runs and are used for the custom-example predictions.

## Setup

- 1000 labelled questions, stratified split 700 train / 100 validation / 200 test (same split for both models)
- Input: `[CLS] question [SEP] abstract [SEP]`, max 512 tokens, only the abstract gets truncated.
  The `long_answer` field is not used because it contains the answer.
- Same training settings for both models: lr 3e-5, effective batch size 16, 10 epochs, 10% warmup,
  class-weighted cross-entropy, best epoch picked by validation macro-F1
- Each model trained with 3 seeds (42, 7, 2024)
- Hardware: MacBook (Apple M5, 16 GB), PyTorch MPS backend, about 15 min per run

## Results (test set, mean ± std over 3 seeds)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| bert-base-uncased | 0.497 ± 0.087 | 0.411 ± 0.079 | 0.389 ± 0.065 | 0.390 ± 0.068 |
| PubMedBERT | **0.605 ± 0.096** | **0.491 ± 0.102** | **0.481 ± 0.037** | **0.461 ± 0.032** |
| Majority class (always "yes") | 0.550 | 0.183 | 0.333 | 0.237 |

Short version of the findings (details in sections 5-8 of the notebook):

- PubMedBERT is better on all four metrics, and on 2 of the 3 seeds. Most of the gain is on the "no" class.
- Neither model handles "maybe" well. PubMedBERT gets 1 of 22 test "maybe" questions right, BERT gets 5.
- PubMedBERT's tokenizer splits biomedical words much less (1.37 vs 1.53 tokens per word).
- On 6 hand-written examples PubMedBERT got 5 right and BERT got 2. Neither recognised the "maybe" case.
- Results vary a lot between seeds because the training set is small. Gu et al. (2020) see the same on PubMedQA.

## How to run

```
pip install -r requirements.txt
jupyter notebook BERT_vs_PubMedBERT_PubMedQA.ipynb
```

Runs on CUDA, Apple MPS or CPU (CPU will be very slow). Running the whole notebook trains all 6 models
from scratch, which took about 90 minutes on my laptop. The notebook writes its results to `outputs/`. If it
is re-run, runs that are already in `outputs/` are loaded instead of being trained again. `saved_outputs/`
holds a copy of the results from my run, which are the numbers shown in the notebook.
