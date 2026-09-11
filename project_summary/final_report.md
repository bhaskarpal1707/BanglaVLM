# BanglaVLM — Final Report

**Research question:** Can SmolVLM2-2.2B fine-tuned with parameter-efficient methods learn Bengali VQA and generalize across different Bengali visual datasets?

## Cross-dataset results (Normalized Exact Match)

| model         |   Bangla-Bayanno |   ChitroJera |   BanglaVerse |
|:--------------|-----------------:|-------------:|--------------:|
| Base SmolVLM2 |            0     |        0     |         0     |
| BanglaVLM-v1  |            0.342 |        0.032 |         0     |
| BanglaVLM-v2  |            0.312 |        0.06  |         0.004 |

## Observations

- **Base SmolVLM2**: Bangla-Bayanno=0.0%, ChitroJera=0.0%, BanglaVerse=0.0%
- **BanglaVLM-v1**: Bangla-Bayanno=34.2%, ChitroJera=3.2%, BanglaVerse=0.0%
- **BanglaVLM-v2**: Bangla-Bayanno=31.2%, ChitroJera=6.0%, BanglaVerse=0.4%

Fine-tuning on Bangla-Bayanno alone changed in-domain Normalized Exact Match by +34.2 percentage points relative to the base model.

Adding ChitroJera to the training mix (BanglaVLM-v2) changed performance on the fully unseen BanglaVerse set by +0.4 percentage points relative to BanglaVLM-v1 — the key cross-dataset generalization signal here.

## Caveats
- All numbers above are measured on the actual runs in this notebook chain; none are fabricated or taken from external papers.
- Results depend on the specific train/val/test image-level splits, a resource-constrained training budget (7k-example subset, 1000 steps on a free T4), and a 500-example evaluation cap per test set — re-running with a larger budget or different seed may shift the numbers.
- Do not extrapolate beyond what the measured deltas above support.
