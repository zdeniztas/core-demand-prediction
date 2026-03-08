# Core Demand Prediction Pipeline (L1 + L2 + L3)

A multi-level demand prediction pipeline that generates three submission files:

- **L1**: predicts future demand at **eclass** level
- **L2**: predicts future demand at **eclass + manufacturer** level
- **L3**: predicts future demand at **cluster** level inside each eclass using product text and feature-based grouping

This pipeline is designed for large-scale transaction data and uses chunked loading, buyer-history signals, collaborative filtering, association rules, hierarchical cold-start logic, and intra-eclass clustering.

---

## Overview

The script produces three levels of predictions:

### Level 1 — Eclass prediction
Predicts which **eclasses** a buyer is likely to need.

Core logic:
- Warm buyers: direct history + light collaborative filtering + association rules
- Cold buyers: hierarchical NACE-based profiles

### Level 2 — Eclass + Manufacturer prediction
Predicts which **eclass|manufacturer** combinations a buyer is likely to need.

Core logic:
- Start from strong L1 eclass candidates
- Assign manufacturers only when confidence is high enough
- Supports **top-2 manufacturer emission** when buyer behavior is split between brands

### Level 3 — Cluster prediction inside eclass
Predicts which **feature cluster** inside an eclass a buyer is likely to need.

Core logic:
- Cluster products within each eclass using product text
- Use regex feature extraction + TF-IDF + adaptive clustering
- Warm buyers rely on direct cluster history
- Cold buyers rely on segment-level cluster popularity

---

## Main Design Principles

- Predict **eclass first**, then refine to manufacturer or cluster
- Avoid using weather, seasonality, or SKU-level signals for L1/L2
- Use **eclass as the main abstraction** to reduce catalog noise
- Handle cold start with hierarchical industry signals:
  - `(nace_2digits, emp_bucket)`
  - `(nace_section, emp_bucket)`
  - `nace_2digits`
  - `nace_section`
  - `emp_bucket`
  - `global`
- Build L3 using **within-eclass clustering** instead of global clustering
- Keep L1 broad, L2 broader but filtered, and L3 structured and interpretable

---

## Expected Input Files

The script expects the following files:

- `plis_training_cleaned.csv`
- `customer_test_cleaned.csv`
- `les_cs.csv`

Optional:
- `nace_codes.csv`

---

## Output Files

The script writes:

- `submission_L1.csv`  
  Columns: `buyer_id,predicted_id`  
  `predicted_id = eclass`

- `submission_L2.csv`  
  Columns: `buyer_id,predicted_id`  
  `predicted_id = eclass|manufacturer`

- `submission_L3.csv`  
  Columns: `buyer_id,predicted_id`  
  `predicted_id = cluster_id`  
  Example: `19020102_C04`

- `cluster_manifest.csv`  
  Human-readable description of L3 clusters  
  Columns:
  - `cluster_id`
  - `eclass`
  - `n_skus`
  - `n_buyers`
  - `top_features`
  - `example_description`

---

## Warm vs Cold Buyer Handling

The pipeline assumes this mapping from `les_cs.csv`:

- `cs = 1` → warm-start buyer with usable history
- `cs = 0` → cold-start buyer

### Warm buyers
Warm buyers are scored using behavioral history.

For L1, signals include:
- order frequency
- month coverage
- spend
- spend share
- order share
- recency
- price proxy
- light user-user collaborative filtering
- eclass association rules

For L2:
- strong eclass candidates are selected first
- manufacturer is assigned using buyer-level, segment-level, or global manufacturer preference

For L3:
- direct buyer-cluster history is used
- cross-sell signal is added using dominant NACE cluster for eclasses already showing real demand

### Cold buyers
Cold buyers do not rely on transaction history.

For L1/L2:
- predictions come from hierarchical NACE and company-size segment profiles

For L3:
- clusters are ranked by NACE segment popularity

---

## Level-by-Level Logic

---

### Level 1 — Eclass Prediction

#### Warm-start scoring
For each buyer-eclass pair, the script computes a heuristic score using:
- spend
- frequency
- month coverage
- spend share within buyer
- order share within buyer
- price signal
- recency

Then it blends:
- direct buyer history
- user-user collaborative filtering
- association rules

This creates a broad but still behavior-driven candidate list.

#### Cold-start scoring
If the buyer has no usable history, the pipeline builds a ranked candidate list from the most specific available profile:
1. `(nace_2digits, emp_bucket)`
2. `(nace_section, emp_bucket)`
3. `nace_2digits`
4. `nace_section`
5. `emp_bucket`
6. `global`

Each profile ranks eclasses by:
- prevalence in the segment
- average value
- average number of months
- average number of orders

---

### Level 2 — Eclass + Manufacturer Prediction

L2 does **not** directly predict manufacturer from scratch.

Instead it follows:

1. generate strong eclass candidates
2. choose manufacturer with confidence gating

Manufacturer selection uses:
- buyer-level manufacturer concentration inside the eclass
- NACE 2-digit segment preference
- NACE section preference
- global preference

Manufacturer predictions are only emitted if:
- the eclass is globally brand-stable enough
- the top manufacturer has sufficient share or separation
- segment/global fallback scores are strong enough

The script can emit:
- 0 manufacturers
- 1 manufacturer
- 2 manufacturers

The top-2 case is used when the buyer clearly splits demand between two brands.

---

### Level 3 — Cluster Prediction

L3 predicts a more granular target: a **cluster inside each eclass**.

#### Step 0 — Raw data loading
A second pass over the training file loads:
- SKU-like identifier
- product text fields if available
- transaction history needed for buyer-cluster scoring

If no SKU column exists, the script falls back to a pseudo-SKU:
- `eclass|manufacturer`

#### Step 1 — Feature extraction
Product text is normalized and structured features are extracted using regex patterns.

Examples of extracted feature families:
- size
- material
- standards / norms
- pack size
- color
- condition
- numeric bins

These are combined with raw text.

#### Step 2 — Within-eclass clustering
Products are clustered **within each eclass**, not globally.

Method:
- TF-IDF vectorization on feature-enhanced text
- adaptive `k` selection using silhouette score
- clustering with:
  - `AgglomerativeClustering` for smaller sets
  - `KMeans` for larger sets

If there is not enough text, clustering falls back to manufacturer grouping.

Cluster IDs are deterministic:
- `eclass_C00`
- `eclass_C01`
- etc.

#### Step 3 — Cluster quality filtering
Clusters are kept only if they are stable.

A stable cluster should typically have:
- enough buyers
- enough months of activity
- enough internal coherence

#### Step 4 — Buyer-cluster scoring
For warm buyers, cluster scores are built from:
- number of distinct months
- average price
- recency multiplier

This creates a buyer-cluster demand signal.

#### Step 5 — L3 prediction
- Warm buyers: direct historical clusters + NACE cross-sell signal
- Cold buyers: top segment-ranked clusters for their NACE section

#### Step 6 — Cluster manifest
A readable summary is generated for all stable clusters.

This helps evaluate:
- what each cluster represents
- how many SKUs it contains
- which features define it

---

## Configuration

The script uses a large config dictionary, including settings for:

### L1 / L2
- warm thresholds
- cold top-N limits
- manufacturer confidence thresholds
- cold-start prevalence thresholds

### L3
- `warm_threshold_l3`
- `cold_top_n_l3`
- `min_cluster_buyers`
- `min_cluster_months`
- `l3_cross_sell_min_eclass_orders`

These parameters control how broad or selective each level is.

---

## Required Python Packages

Install dependencies with:

```bash
pip install pandas numpy scipy scikit-learn
