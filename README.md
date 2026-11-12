# Named Entity Recognition: LSTM vs BiLSTM vs BiLSTM+CRF vs DistilBERT

End-to-end sequence labeling pipeline on **CoNLL-2003** that detects **PER, ORG, LOC and MISC** entities. Three architectures are built from scratch in PyTorch, a transformer is fine-tuned with HuggingFace, and the final model is deployed as a Gradio app with live, color-coded entity highlighting.

## What is inside

| Stage | Details |
|---|---|
| Data | CoNLL-2003 loaded from the HuggingFace Hub, exploration (entity counts, sentence lengths) |
| Tagging | IOB2 scheme, helpers to convert tags to spans and back (round-trip verified) |
| Embeddings | Pre-trained GloVe 100d (`glove-wiki-gigaword-100`) |
| OOV handling | Character-level CNN concatenated to the word embedding; subword tokenization for the transformer |
| Models (from scratch) | LSTM, BiLSTM, BiLSTM+CRF with a hand-written CRF layer (forward algorithm + Viterbi) |
| Transformer | `distilbert-base-cased` fine-tuned with `AutoTokenizer`, `AutoModelForTokenClassification`, `Trainer` |
| Evaluation | seqeval per-entity precision / recall / F1 (PER, ORG, LOC, MISC) |
| CRF analysis | Counts invalid IOB transitions and boundary errors per model to show where the CRF helps |
| Visualization | Color-coded span rendering in the notebook and a live Gradio app |

## Repository layout

```
.
├── NER_Sequence_Labeling.ipynb   full pipeline: data, 4 models, evaluation, comparison, Gradio
├── app.py                        standalone Gradio app (loads the saved DistilBERT model)
├── requirements.txt
└── README.md
```

## Run on Kaggle (recommended)

1. Upload `NER_Sequence_Labeling.ipynb` (File > Import Notebook).
2. Session options: **Accelerator = GPU**, **Internet = On**.
3. Run All. The fine-tuned model is saved to `/kaggle/working/ner-distilbert-final`.

Approximate training time on a Kaggle GPU: a few minutes per from-scratch model (10 epochs; BiLSTM+CRF is the slowest) and a few minutes for DistilBERT (3 epochs).

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook NER_Sequence_Labeling.ipynb
```

A GPU is strongly recommended.

## Run the Gradio app

After training, the notebook saves the model to `ner-distilbert-final/`. Then:

```bash
python app.py                      
python app.py path/to/model_dir    
```

Open the printed local URL, type any text, and entities are highlighted as you type.

## Results

Validation F1 (seqeval, entity level) of the from-scratch models from one run:

| Model | Best validation F1 |
|---|---|
| LSTM | 0.871 |
| BiLSTM | 0.930 |
| BiLSTM + CRF | see notebook |
| DistilBERT | see notebook |

Test-set precision / recall / F1 per entity type for all four models are produced by sections "Test-set evaluation" and "Compare all four architectures" in the notebook.

## Where the CRF helps

A softmax layer predicts each tag independently, so it can output inconsistent sequences such as `O I-ORG` or `B-PER I-LOC`. The CRF scores the whole tag sequence with a learned transition matrix and decodes the best valid path with Viterbi. This mainly improves the start and end boundaries of multi-token entities. The notebook quantifies it with invalid-transition and boundary-error counts.

## Notes

- CoNLL-2003 has no DATE tag, so the system covers PER, ORG, LOC and MISC. To add dates, train the same pipeline on a dataset that has that label (for example OntoNotes 5).
- Model weights are not committed to git (see `.gitignore`). Upload them to the HuggingFace Hub if you want to share them.
