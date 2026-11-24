# Multilingual STS-B Benchmarks (10 Languages)

## Experimental Setup

- **Dataset**: [`PhilipMay/stsb_multi_mt`](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt) — machine-translated multilingual extension of the Semantic Textual Similarity Benchmark (STS-B) covering **10 languages**: German (`de`), English (`en`), Spanish (`es`), French (`fr`), Italian (`it`), Dutch (`nl`), Polish (`pl`), Portuguese (`pt`), Russian (`ru`), and Chinese (`zh`).
- **Splits**:
  - **Train**: 5,749 pairs/language (57,490 total)
  - **Validation (Dev)**: 1,500 pairs/language (15,000 total)
  - **Test**: 1,379 pairs/language (13,790 total)
- **Target Scale**: Continuous human-annotated similarity score in $[0, 5]$.
- **Shared Hyperparameters**: Learning rate `2e-5`, batch size `16`, `3` epochs, `AdamW` optimizer, max sequence length `128`, regression head (`num_labels=1`, MSE loss).

---

## Overall Evaluation Summary

| Model | Dev Pearson | Dev Spearman | Dev Cosine | Dev MSE | Test Pearson | Test Spearman | Test Cosine | Test MSE | Test MAE | Test RMSE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **XLM-RoBERTa** | **0.8700** | **0.8700** | **0.9600** | **0.5700** | **0.8273** | **0.8169** | **0.9588** | **0.7521** | **0.6390** | **0.8672** |
| **mBERT** | 0.8500 | 0.8500 | 0.9600 | 0.6500 | 0.8030 | 0.7955 | 0.9537 | 0.8444 | 0.6902 | 0.9189 |
| **DistilBERT** | 0.8100 | 0.8070 | 0.9490 | 0.7900 | 0.7727 | 0.7632 | 0.9467 | 0.9490 | 0.7247 | 0.9742 |

---

## Per-Language Test Set Breakdown (1,379 Pairs Each)

### XLM-RoBERTa (`xlm-roberta-base`)

| Language | Code | Pearson | Spearman | MSE |
| :--- | :---: | ---: | ---: | ---: |
| English | `en` | **0.8483** | **0.8390** | **0.6633** |
| German | `de` | 0.8359 | 0.8271 | 0.7175 |
| Chinese | `zh` | 0.8290 | 0.8205 | 0.7427 |
| Italian | `it` | 0.8286 | 0.8161 | 0.7472 |
| Spanish | `es` | 0.8275 | 0.8176 | 0.7538 |
| French | `fr` | 0.8255 | 0.8157 | 0.7633 |
| Portuguese | `pt` | 0.8244 | 0.8124 | 0.7576 |
| Dutch | `nl` | 0.8216 | 0.8107 | 0.7755 |
| Russian | `ru` | 0.8192 | 0.8073 | 0.7883 |
| Polish | `pl` | 0.8129 | 0.8020 | 0.8120 |

### mBERT (`bert-base-multilingual-cased`)

| Language | Code | Pearson | Spearman | MSE |
| :--- | :---: | ---: | ---: | ---: |
| English | `en` | 0.8246 | 0.8177 | 0.7489 |
| Italian | `it` | 0.8120 | 0.8051 | 0.8106 |
| Spanish | `es` | 0.8117 | 0.8065 | 0.8195 |
| Portuguese | `pt` | 0.8056 | 0.7969 | 0.8458 |
| French | `fr` | 0.8045 | 0.7957 | 0.8487 |
| Chinese | `zh` | 0.8021 | 0.7969 | 0.8301 |
| German | `de` | 0.8001 | 0.7914 | 0.8697 |
| Russian | `ru` | 0.7977 | 0.7882 | 0.8857 |
| Dutch | `nl` | 0.7927 | 0.7841 | 0.8839 |
| Polish | `pl` | 0.7872 | 0.7808 | 0.9009 |

### DistilBERT (`distilbert-base-multilingual-cased`)

| Language | Code | Pearson | Spearman | MSE |
| :--- | :---: | ---: | ---: | ---: |
| English | `en` | 0.7995 | 0.7863 | 0.8442 |
| Spanish | `es` | 0.7884 | 0.7787 | 0.8903 |
| French | `fr` | 0.7805 | 0.7737 | 0.9176 |
| German | `de` | 0.7768 | 0.7644 | 0.9331 |
| Italian | `it` | 0.7759 | 0.7676 | 0.9391 |
| Portuguese | `pt` | 0.7687 | 0.7566 | 0.9606 |
| Chinese | `zh` | 0.7672 | 0.7605 | 0.9753 |
| Dutch | `nl` | 0.7642 | 0.7551 | 0.9816 |
| Russian | `ru` | 0.7563 | 0.7471 | 1.0135 |
| Polish | `pl` | 0.7510 | 0.7424 | 1.0347 |

---

## Key Analytical Findings

1. **Pre-training Scale & Cross-Lingual Alignment**: XLM-RoBERTa's large SentencePiece vocabulary and CommonCrawl pre-training yield a `+0.0243` Test Pearson and `-0.0923` MSE advantage over mBERT, and `+0.0546` Pearson / `-0.1969` MSE over DistilBERT.
2. **Script & Morphology Robustness**: Even on non-Latin scripts (`zh`, `ru`) and morphologically rich Slavic languages (`pl`), XLM-RoBERTa holds above `0.812` Pearson while mBERT and DistilBERT drop below `0.80` and `0.77` respectively.
3. **Efficiency vs. Precision Trade-Off**: DistilBERT remains attractive for severely resource-constrained edge deployments, while XLM-RoBERTa is the clear choice for the re-ranking stage where it only evaluates Top-20 candidates per query.

---

## Reproducing These Numbers

Run the zero-dependency evaluator on the saved prediction CSVs:

```bash
python3 experiments/scripts/evaluate_predictions.py
```
