# ML Challenge 2026: Business Entity Resolution — Methodology

**Team Name:** Entity Resolvers  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We built a five-stage blocking-then-classifying pipeline that anchors all matching on Source 1 (the master reference). Country-agnostic text normalisation handles English, Hindi transliterations, and French names/addresses uniformly. A union of five complementary blockers keeps blocking recall above 97%, and a LightGBM classifier trained on 40+ string, address, and multilingual embedding features makes the final match/no-match decision at a conservative F0.5-tuned threshold.

---

## 2. Methodology

### 2.1 Problem Analysis

**Key EDA findings:**
- **Name noise**: legal suffixes vary widely (Corp/Corporation, Pvt/Private, Ltd/Limited); word-order swaps are frequent; Hindi names appear both in Devanagari script and transliterated Roman; French names use SARL, SAS, etc.
- **Address noise**: abbreviations (Rd/Rd., St/Street), missing PIN/ZIP codes, landmark references ("Near SBI ATM"), reordered components (number after city in some Indian records), and non-ASCII characters.
- **Singleton prevalence**: a meaningful fraction of S1 entities have zero matches in S2/S3 — incorrectly predicting any match for these costs a full F0.5 score of 0.
- **Cardinality**: one S1 entity can match zero, one, or many S2/S3 records; S2 and S3 records are never matched against each other.
- **Open-set country**: France appears only in test; the pipeline must generalise without country-specific code.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Gradient Boosted Classifier  
**Core Innovation:** Union of five complementary blockers (postal, rare-token, TF-IDF char-n-gram, multilingual embedding ANN, and city-token) combined with a conservative F0.5-tuned LightGBM threshold and per-entity conflict resolution.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blockers Used

| Blocker | Key | Recall Contribution |
|---------|-----|---------------------|
| Country filter | Same `country` string | Reduces cross-country false positives |
| Postal code | Exact PIN/ZIP match | High precision; catches geographically co-located businesses |
| Rare-token | Share a name token with DF ≤ 1% of all records | Catches distinctive business names exactly |
| TF-IDF char-3gram | Top-30 sparse cosine neighbours per S1 entity | Robust to abbreviations and typos |
| Embedding ANN | FAISS top-25 with `paraphrase-multilingual-MiniLM-L12-v2` | Language-agnostic; handles transliterations and French |
| City token | Share last 2 significant address tokens | Catches geographically nearby businesses with similar names |

### 3.2 Design Decisions

- All blockers run **within country partitions** first (cheap and high-precision), plus a fallback TF-IDF run for unknown-country records.
- Results are **unioned** — a pair is a candidate if any single blocker fires.
- The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) supports 50+ languages and is Apache 2.0 licensed, 118M parameters (well under the 8B limit).
- Candidate recall target: ≥ 97% (measured on training set with `evaluate_blocking()`).

### 3.3 Candidate Pair Volume

| Source | Approx. candidates per S1 entity |
|--------|-----------------------------------|
| S2 | 30–80 (after union + dedup) |
| S3 | 30–80 |

---

## 4. Matching Model

### 4.1 Features Used

**Name features (21 features)**
- Jaccard similarity on word tokens and character 3/4-grams
- Levenshtein normalised distance, Jaro-Winkler similarity
- RapidFuzz partial ratio, token-sort ratio, token-set ratio
- Core-name exact match, sorted-token bag exact match
- Acronym/initialism match
- Legal suffix agreement ratio

**Address features (13 features)**
- Normalised Levenshtein edit distance on full address
- Jaccard token overlap on address token sets
- Partial ratio and token-set ratio on address strings
- Jaro-Winkler on address
- Postal code: exact match / 3-digit prefix match / both-missing / one-missing flags
- Street number exact match / both-missing flag
- Landmark token presence flags for S1 and candidate

**Semantic features (1 feature)**
- Cosine similarity of multilingual sentence embeddings (name + address concatenated)

**Context features (3 features)**
- Rank of this candidate within its S1 group (by embedding cosine)
- Score gap to the top-ranked candidate in the group
- Total number of candidates for this S1 entity

**Meta features (6 features)**
- Missing-name / missing-address indicator for S1 and candidate
- Source indicator: is S2 / is S3

### 4.2 Model Architecture

- **Model**: LightGBM binary classifier (`LGBMClassifier`)
- **Training pairs**: Ground-truth positives + up to 10× hard negatives per positive (drawn from blocking candidates that are **not** in the ground truth)
- **Validation split**: `GroupShuffleSplit(test_size=0.2)` by S1 entity — no S1 entity appears in both train and val
- **Class imbalance**: `scale_pos_weight` set to negative/positive ratio
- **Early stopping**: 50 rounds monitoring binary logloss on validation set
- **Threshold selection**: Grid search over [0.20, 0.95] in steps of 0.02, maximising **macro F0.5** on the validation group split

**Model hyperparameters:**
```
n_estimators=1000, learning_rate=0.05, num_leaves=63,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
lambda_l1=0.1, lambda_l2=0.1
```

### 4.3 Threshold Selection

The metric weights precision 2× over recall, so we tuned the decision threshold at a high value (empirically 0.65–0.80) to be conservative. Predicting singletons correctly (empty output) is treated as a first-class objective.

---

## 5. Post-Processing

### 5.1 Per-Entity Thresholding

For each S1 entity: keep candidates whose score ≥ threshold. If none pass, predict an empty list (singleton).

### 5.2 Conflict Resolution

If one S2/S3 record is claimed by multiple S1 entities with scores above threshold, it is assigned **only** to the S1 entity with the highest match score. This mirrors the real-world assumption that each noisy record belongs to a single business.

---

## 6. Results & Error Analysis

- **F0.5 Score (macro):** Measured on held-out validation group split (20% of S1 entities)
- **Common false positives (wrong merges):** Businesses in the same zip code with similar but distinct names (e.g., chain stores with the same brand name at different locations, or legal name vs. DBA trade name collisions)
- **Common false negatives (missed links):** Records with missing address fields where only name similarity signals are available; transliterated Indian names with high Levenshtein distance from the normalised form

---

## 7. Conclusion

Our pipeline combines country-agnostic normalisation (NFKD + abbreviation expansion), a union of five complementary blockers anchored on S1, 40+ pair features covering string, address, and multilingual semantic similarity, and a conservative LightGBM threshold tuned directly on the F0.5 metric. Singleton handling and conflict resolution are treated as first-class concerns. The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, Apache 2.0, 118M params) provides language-agnostic recall that generalises to unseen countries like France without any code changes.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py    – text normalisation (Stage 1)
│   ├── blocking.py     – candidate generation (Stage 2)
│   ├── features.py     – pair feature engineering (Stage 3)
│   ├── train.py        – model training & threshold tuning (Stage 4)
│   ├── predict.py      – inference & TSV output (Stage 5)
│   └── evaluate.py     – local evaluation utility
├── README.md           – end-to-end run instructions
└── requirements.txt    – pinned package versions
```

**Entry points** (run from `student_resource/` directory):
1. `python code/business_entity_resolution/src/train.py` — trains model, saves to `models/`
2. `python code/business_entity_resolution/src/predict.py` — generates `output/*.tsv`
3. `python utils/validate_submission.py --matching output/matching_results.tsv ...` — validates

### B. Licence Compliance

| Package | Licence | Params (if model) |
|---------|---------|-------------------|
| LightGBM | MIT | N/A |
| RapidFuzz | MIT | N/A |
| scikit-learn | BSD-3 | N/A |
| sentence-transformers | Apache 2.0 | N/A |
| paraphrase-multilingual-MiniLM-L12-v2 | Apache 2.0 | 118M |
| FAISS | MIT | N/A |
| PyTorch | BSD-3 | N/A |

All models are ≤ 8B parameters and use MIT or Apache 2.0 licences. No external entity-resolution APIs or internet lookups are used.
