import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, normalize
import time
import re
import unicodedata
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

"""
Core Demand Prediction Pipeline — Final Submission (L1 + L2 + L3)
==================================================================

WHERE EACH LEVEL STARTS
------------------------
  Level 1 code  : function predict_level_1()        — search "def predict_level_1"
  Level 2 code  : function predict_level_2()        — search "def predict_level_2"
  Level 3 code  : line marked  >>> LEVEL 3 STARTS HERE <<<
                  - Text normalisation helpers      — search "LEVEL 3 — TEXT NORMALIZATION"
                  - Raw data loader                 — search "LEVEL 3 — STEP 0"
                  - Clustering & buyer scoring      — search "LEVEL 3 — STEPS 1-4"
                  - Prediction pipeline             — search "LEVEL 3 — STEP 5"
                  - Cluster manifest builder        — search "LEVEL 3 — STEP 6"

Level 3 outputs
---------------
  submission_L3.csv   : buyer_id,predicted_id  where predicted_id = cluster_id (e.g. 19020102_C04)
  cluster_manifest.csv: cluster_id,eclass,n_skus,n_buyers,top_features,example_description

Design principles
-----------------
1) Level 1 is predicted directly on eclass.
2) Level 2 is predicted conditionally:
      strong eclass candidate -> manufacturer assignment with confidence gating
3) No weather, no seasonal heuristics, no SKU-level signals for L1/L2.
4) Favor very high recall for L1, broader but still filtered recall for L2.
5) Use eclass as the main abstraction to avoid SKU noise.
6) Use nace hierarchy for cold start:
      (nace_2digits, emp_bucket) -> (nace_section, emp_bucket) -> nace_2digits -> nace_section -> emp_bucket -> global
7) Allow top-2 manufacturers for selected L2 eclasses when brand choice is split.
8) Level 3 clusters products within each eclass using TF-IDF + regex feature extraction
   and adaptive k clustering (silhouette-guided). Cluster IDs are deterministic.
   Cold-start uses NACE segment cluster popularity. Warm-start uses direct buy history
   plus NACE cross-sell signal for eclasses with proven demand.

Assumed cs mapping
------------------
cs = 1 -> warm-start buyer with usable history
cs = 0 -> cold-start buyer

Expected files
--------------
plis_training_cleaned.csv
customer_test_cleaned.csv
les_cs.csv
Optional:
nace_codes.csv

Outputs
-------
submission_L1.csv    : buyer_id,predicted_id where predicted_id = eclass
submission_L2.csv    : buyer_id,predicted_id where predicted_id = eclass|manufacturer
submission_L3.csv    : buyer_id,predicted_id where predicted_id = cluster_id
cluster_manifest.csv : human-readable cluster descriptions for L3 methodology evaluation
"""

# =============================================================================
# CONFIG
# =============================================================================
BASE = r"C:\Users\admin\Downloads"
TRAIN_PATH = f"{BASE}/cleaned/plis_training_cleaned.csv"
TEST_PATH = f"{BASE}/cleaned/customer_test_cleaned.csv"
LES_CS = f"{BASE}/les_cs.csv"
NACE_CODES_PATH = f"{BASE}/nace_codes.csv"
OUTPUT_L1 = f"{BASE}/submission_L1.csv"
OUTPUT_L2 = f"{BASE}/submission_L2.csv"
OUTPUT_L3 = f"{BASE}/submission_L3.csv"
OUTPUT_L3_MANIFEST = f"{BASE}/cluster_manifest.csv"

CHUNK = 1_000_000

# L3-specific columns to attempt loading (text columns are optional)
TRAIN_COLS_L3 = ["legal_entity_id", "eclass", "orderdate", "quantityvalue", "vk_per_item"]

# Product text column candidates (tried in order, loaded if present)
_L3_TEXT_COLS = [
    "product_name", "product_description", "short_description", "long_description",
] + [f"feature_{i}" for i in range(1, 11)]

TRAIN_COLS = [
    "orderdate",
    "legal_entity_id",
    "eclass",
    "manufacturer",
    "quantityvalue",
    "vk_per_item",
    "estimated_number_employees",
    "nace_code",
    "nace_section",
]

# broader settings to move closer to winner-style recall
WARM_TOP_N_L1 = 1200
COLD_TOP_N_L1 = 180

WARM_TOP_N_L2_ECLASS = 450
COLD_TOP_N_L2_ECLASS = 90

MIN_GLOBAL_BUYERS_L1 = 2
MIN_GLOBAL_BUYERS_L2 = 2
CF_K = 40  # still minor

DEFAULT_CFG = {
    # broader thresholds
    "warm_threshold_l1": 0.008,
    "warm_threshold_l2_eclass": 0.018,
    "cold_top_n_l1": COLD_TOP_N_L1,
    "cold_top_n_l2_eclass": COLD_TOP_N_L2_ECLASS,

    # L2 manufacturer confidence gating (softer)
    "min_top_mfr_order_share": 0.35,
    "min_top_mfr_month_share": 0.28,
    "min_top_mfr_gap": 0.06,
    "min_seg_mfr_score": 0.16,
    "min_global_mfr_score": 0.14,

    # brand-stable eclass gating (softer)
    "min_eclass_top_mfr_share_global": 0.16,
    "min_eclass_buyer_coverage_l2": 2,

    # top-2 manufacturer logic
    "emit_top2_when_top1_share_below": 0.55,
    "emit_top2_min_top1_share": 0.30,
    "emit_top2_min_top2_share": 0.20,

    # cold-start filtering
    "cold_min_prev_nace2": 0.006,
    "cold_min_prev_section": 0.010,
    "cold_min_prev_global": 0.010,

    # L3 cluster prediction
    "warm_threshold_l3": 0.010,
    "cold_top_n_l3": 25,
    "min_cluster_buyers": 2,
    "min_cluster_months": 2,
    "l3_cross_sell_min_eclass_orders": 3,
}


# =============================================================================
# HELPERS
# =============================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emp_bucket(x):
    if pd.isna(x) or x <= 0:
        return "unknown"
    if x < 100:
        return "small"
    if x < 500:
        return "medium"
    if x < 2000:
        return "large"
    if x < 10000:
        return "xlarge"
    return "enterprise"


def normalize_dict(d: dict) -> dict:
    if not d:
        return {}
    mx = max(d.values())
    if mx <= 0:
        return {}
    return {k: v / mx for k, v in d.items()}


def to_period_month_span(first_ym, last_ym) -> int:
    return (last_ym.year - first_ym.year) * 12 + (last_ym.month - first_ym.month) + 1


def safe_nace2(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if len(s) >= 2:
        return s[:2]
    return None


# =============================================================================
# NACE ENRICHMENT
# =============================================================================
def load_nace_lookup():
    """
    Optional support for nace_codes.csv.
    Returns dicts keyed by nace_code.
    """
    try:
        nace = pd.read_csv(NACE_CODES_PATH, dtype=str)
        nace.columns = [c.strip() for c in nace.columns]
        if "nace_code" not in nace.columns:
            log("nace_codes.csv found but nace_code column missing; skipping lookup")
            return {}, {}

        nace["nace_code"] = nace["nace_code"].astype(str).str.strip()
        nace["nace_2digits"] = nace["nace_2digits"].astype(str).str.strip() if "nace_2digits" in nace.columns else nace["nace_code"].str[:2]
        code_to_nace2 = dict(zip(nace["nace_code"], nace["nace_2digits"]))
        code_to_desc = {}
        if "n_nace_description" in nace.columns:
            code_to_desc = dict(zip(nace["nace_code"], nace["n_nace_description"]))
        log(f"Loaded nace lookup with {len(code_to_nace2):,} codes")
        return code_to_nace2, code_to_desc
    except Exception as e:
        log(f"Could not load nace_codes.csv, using raw nace codes only ({e})")
        return {}, {}


# =============================================================================
# DATA LOADING / AGGREGATION
# =============================================================================
def load_and_aggregate():
    """
    One-pass chunked loader for:
    - L1 buyer-eclass aggregates
    - L2 buyer-(eclass|manufacturer) aggregates
    - manufacturer preference maps
    - cold-start segment profiles
    """
    log("Loading training data...")

    code_to_nace2, _ = load_nace_lookup()

    # buyer profile
    bp_demo = {}

    # ----------------------
    # L1: buyer x eclass
    # ----------------------
    l1_n = defaultdict(int)
    l1_val = defaultdict(float)
    l1_psum = defaultdict(float)
    l1_months = defaultdict(set)
    l1_first = {}
    l1_last = {}

    g1_buyers = defaultdict(set)
    g1_n = defaultdict(int)
    g1_psum = defaultdict(float)

    # ----------------------
    # L2: buyer x eclass|manufacturer
    # ----------------------
    l2_n = defaultdict(int)
    l2_val = defaultdict(float)
    l2_psum = defaultdict(float)
    l2_months = defaultdict(set)
    l2_first = {}
    l2_last = {}

    g2_buyers = defaultdict(set)
    g2_n = defaultdict(int)
    g2_psum = defaultdict(float)
    g2_eclass = {}
    g2_mfr = {}

    # buyer-eclass-manufacturer preference
    bem_n = defaultdict(int)
    bem_val = defaultdict(float)
    bem_months = defaultdict(set)
    bem_last = {}

    # segment-level manufacturer signal
    # by nace_section
    sem_buyers = defaultdict(set)
    sem_orders = defaultdict(int)
    sem_value = defaultdict(float)

    # by nace_2digits
    n2m_buyers = defaultdict(set)
    n2m_orders = defaultdict(int)
    n2m_value = defaultdict(float)

    # global eclass -> manufacturer stats
    gem_buyers = defaultdict(set)
    gem_orders = defaultdict(int)
    gem_value = defaultdict(float)

    rows_loaded = 0

    dtype_map = {
        "legal_entity_id": str,
        "eclass": str,
        "manufacturer": str,
        "nace_code": str,
        "nace_section": str,
    }

    for chunk in pd.read_csv(TRAIN_PATH, usecols=TRAIN_COLS, chunksize=CHUNK, dtype=dtype_map):
        chunk["eclass"] = chunk["eclass"].fillna("").astype(str).str.strip()
        chunk["manufacturer"] = chunk["manufacturer"].fillna("").astype(str).str.strip()
        chunk["legal_entity_id"] = chunk["legal_entity_id"].astype(str)
        chunk["nace_code"] = chunk["nace_code"].fillna("").astype(str).str.strip()
        chunk["nace_section"] = chunk["nace_section"].fillna("").astype(str).str.strip()

        chunk = chunk[chunk["eclass"] != ""].copy()
        chunk["eclass_mfr"] = chunk["eclass"] + "|" + chunk["manufacturer"]
        chunk["dt"] = pd.to_datetime(chunk["orderdate"], errors="coerce")
        chunk = chunk.dropna(subset=["dt"])
        chunk["ym"] = chunk["dt"].dt.to_period("M")
        chunk["quantityvalue"] = pd.to_numeric(chunk["quantityvalue"], errors="coerce").fillna(1)
        chunk["vk_per_item"] = pd.to_numeric(chunk["vk_per_item"], errors="coerce").fillna(0)
        chunk["line_val"] = (chunk["quantityvalue"] * chunk["vk_per_item"]).clip(lower=0)

        # derive nace_2digits
        if code_to_nace2:
            chunk["nace_2digits"] = chunk["nace_code"].map(code_to_nace2)
            chunk["nace_2digits"] = chunk["nace_2digits"].fillna(chunk["nace_code"].map(safe_nace2))
        else:
            chunk["nace_2digits"] = chunk["nace_code"].map(safe_nace2)

        for row in chunk.itertuples(index=False):
            buyer = row.legal_entity_id
            eclass = row.eclass
            manufacturer = row.manufacturer
            emfr = row.eclass_mfr
            ym = row.ym
            lv = row.line_val
            price = row.vk_per_item
            nace = row.nace_section
            nace2 = row.nace_2digits
            emp = row.estimated_number_employees

            if buyer not in bp_demo:
                bp_demo[buyer] = {
                    "nace_section": nace,
                    "nace_code": row.nace_code,
                    "nace_2digits": nace2,
                    "emp": emp,
                }

            # ---------------- L1 ----------------
            k1 = (buyer, eclass)
            l1_n[k1] += 1
            l1_val[k1] += lv
            l1_psum[k1] += price
            l1_months[k1].add(ym)

            if k1 not in l1_first or ym < l1_first[k1]:
                l1_first[k1] = ym
            if k1 not in l1_last or ym > l1_last[k1]:
                l1_last[k1] = ym

            g1_buyers[eclass].add(buyer)
            g1_n[eclass] += 1
            g1_psum[eclass] += price

            # ---------------- L2 ----------------
            if manufacturer != "":
                k2 = (buyer, emfr)
                l2_n[k2] += 1
                l2_val[k2] += lv
                l2_psum[k2] += price
                l2_months[k2].add(ym)

                if k2 not in l2_first or ym < l2_first[k2]:
                    l2_first[k2] = ym
                if k2 not in l2_last or ym > l2_last[k2]:
                    l2_last[k2] = ym

                g2_buyers[emfr].add(buyer)
                g2_n[emfr] += 1
                g2_psum[emfr] += price
                g2_eclass[emfr] = eclass
                g2_mfr[emfr] = manufacturer

                km = (buyer, eclass, manufacturer)
                bem_n[km] += 1
                bem_val[km] += lv
                bem_months[km].add(ym)
                if km not in bem_last or ym > bem_last[km]:
                    bem_last[km] = ym

                ks = (nace, eclass, manufacturer)
                sem_buyers[ks].add(buyer)
                sem_orders[ks] += 1
                sem_value[ks] += lv

                if nace2:
                    k2d = (nace2, eclass, manufacturer)
                    n2m_buyers[k2d].add(buyer)
                    n2m_orders[k2d] += 1
                    n2m_value[k2d] += lv

                kg = (eclass, manufacturer)
                gem_buyers[kg].add(buyer)
                gem_orders[kg] += 1
                gem_value[kg] += lv

        rows_loaded += len(chunk)
        log(f"processed {rows_loaded:,} training rows")

    def make_bi(n_map, v_map, p_map, months_map, first_map, last_map, item_to_eclass=None):
        keys = []
        for k in n_map.keys():
            span = to_period_month_span(first_map[k], last_map[k])
            freq = n_map[k] / max(span, 1)
            avg_price = p_map[k] / max(n_map[k], 1)
            econ = np.sqrt(max(avg_price, 0.01)) * freq

            if n_map[k] >= 2 or len(months_map[k]) >= 2 or econ > 0.2:
                keys.append(k)

        bi = pd.DataFrame({
            "buyer": [k[0] for k in keys],
            "item": [k[1] for k in keys],
            "n_orders": [n_map[k] for k in keys],
            "total_value": [v_map[k] for k in keys],
            "avg_price": [p_map[k] / max(n_map[k], 1) for k in keys],
            "n_months": [len(months_map[k]) for k in keys],
            "first_ym": [first_map[k] for k in keys],
            "last_ym": [last_map[k] for k in keys],
        })

        if bi.empty:
            return bi

        max_ym = bi["last_ym"].max()
        bi["recency"] = bi["last_ym"].apply(
            lambda x: (max_ym.year - x.year) * 12 + (max_ym.month - x.month)
        )
        bi["span_months"] = bi.apply(
            lambda r: to_period_month_span(r["first_ym"], r["last_ym"]), axis=1
        )

        if item_to_eclass is not None:
            bi["eclass"] = bi["item"].map(item_to_eclass)
        else:
            bi["eclass"] = bi["item"]

        return bi

    def make_ig(buyers_map, n_map, p_map, item_to_eclass=None, item_to_mfr=None):
        items = list(n_map.keys())
        ig = pd.DataFrame({
            "item": items,
            "n_buyers": [len(buyers_map[i]) for i in items],
            "n_orders": [n_map[i] for i in items],
            "avg_price": [p_map[i] / max(n_map[i], 1) for i in items],
        })
        if item_to_eclass is not None:
            ig["eclass"] = ig["item"].map(item_to_eclass)
        else:
            ig["eclass"] = ig["item"]
        if item_to_mfr is not None:
            ig["manufacturer"] = ig["item"].map(item_to_mfr)
        return ig

    bi_l1 = make_bi(l1_n, l1_val, l1_psum, l1_months, l1_first, l1_last)
    ig_l1 = make_ig(g1_buyers, g1_n, g1_psum)

    bi_l2 = make_bi(l2_n, l2_val, l2_psum, l2_months, l2_first, l2_last, item_to_eclass=g2_eclass)
    ig_l2 = make_ig(g2_buyers, g2_n, g2_psum, item_to_eclass=g2_eclass, item_to_mfr=g2_mfr)

    bp = pd.DataFrame([
        {
            "buyer": b,
            "nace_section": d["nace_section"],
            "nace_code": d["nace_code"],
            "nace_2digits": d["nace_2digits"],
            "emp": d["emp"],
            "emp_bucket": emp_bucket(d["emp"]),
        }
        for b, d in bp_demo.items()
    ])

    mfr_pref = pd.DataFrame([
        {
            "buyer": b,
            "eclass": e,
            "manufacturer": m,
            "n_orders": bem_n[(b, e, m)],
            "total_value": bem_val[(b, e, m)],
            "n_months": len(bem_months[(b, e, m)]),
            "last_ym": bem_last[(b, e, m)],
        }
        for (b, e, m) in bem_n.keys()
    ])
    if not mfr_pref.empty:
        max_ym2 = mfr_pref["last_ym"].max()
        mfr_pref["recency"] = mfr_pref["last_ym"].apply(
            lambda x: (max_ym2.year - x.year) * 12 + (max_ym2.month - x.month)
        )

    seg_mfr = pd.DataFrame([
        {
            "nace_section": n,
            "eclass": e,
            "manufacturer": m,
            "n_buyers": len(sem_buyers[(n, e, m)]),
            "n_orders": sem_orders[(n, e, m)],
            "total_value": sem_value[(n, e, m)],
        }
        for (n, e, m) in sem_orders.keys()
    ])

    nace2_mfr = pd.DataFrame([
        {
            "nace_2digits": n2,
            "eclass": e,
            "manufacturer": m,
            "n_buyers": len(n2m_buyers[(n2, e, m)]),
            "n_orders": n2m_orders[(n2, e, m)],
            "total_value": n2m_value[(n2, e, m)],
        }
        for (n2, e, m) in n2m_orders.keys()
    ])

    global_mfr = pd.DataFrame([
        {
            "eclass": e,
            "manufacturer": m,
            "n_buyers": len(gem_buyers[(e, m)]),
            "n_orders": gem_orders[(e, m)],
            "total_value": gem_value[(e, m)],
        }
        for (e, m) in gem_orders.keys()
    ])

    log(f"L1 aggregates: {len(bi_l1):,} buyer-eclass pairs, {len(ig_l1):,} eclasses")
    log(f"L2 aggregates: {len(bi_l2):,} buyer-eclass|mfr pairs, {len(ig_l2):,} eclass|mfr combos")

    # ---- L3: second-pass product text load + cluster building ---------------
    sku_text_l3, sku_stats_l3, buyer_sku_data_l3, sku_mfr_l3 = _load_l3_raw_data()
    l3_data = build_l3_clusters(sku_text_l3, sku_stats_l3, buyer_sku_data_l3, sku_mfr_l3, bp)

    return {
        "bi_l1":     bi_l1,
        "ig_l1":     ig_l1,
        "bi_l2":     bi_l2,
        "ig_l2":     ig_l2,
        "bp":        bp,
        "mfr_pref":  mfr_pref,
        "seg_mfr":   seg_mfr,
        "nace2_mfr": nace2_mfr,
        "global_mfr": global_mfr,
        # L3 cluster data
        "l3_clusters":      l3_data["cluster_map"],
        "l3_buyer_clusters": l3_data["buyer_clusters"],
        "l3_cluster_stats": l3_data["cluster_stats"],
        "l3_seg_clusters":  l3_data["seg_clusters"],
    }


# =============================================================================
# SPARSE MATRIX / INDEXES
# =============================================================================
def build_sparse(bi: pd.DataFrame, ig: pd.DataFrame, min_buyers: int):
    valid_items = set(ig[ig["n_buyers"] >= min_buyers]["item"])
    bi_f = bi[bi["item"].isin(valid_items)].copy()

    buyer_enc = LabelEncoder().fit(bi_f["buyer"])
    item_enc = LabelEncoder().fit(bi_f["item"])

    rows = buyer_enc.transform(bi_f["buyer"])
    cols = item_enc.transform(bi_f["item"])
    vals = np.log1p(bi_f["total_value"].values).astype(np.float32)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(buyer_enc.classes_), len(item_enc.classes_)))
    mat_norm = normalize(mat, norm="l2", axis=1)

    log(f"sparse matrix: {mat.shape[0]:,} buyers x {mat.shape[1]:,} items, {mat.nnz:,} nnz")
    return mat, mat_norm, buyer_enc, item_enc


def build_buyer_index(bi: pd.DataFrame):
    return {buyer: grp for buyer, grp in bi.groupby("buyer")}


# =============================================================================
# ASSOCIATION RULES
# =============================================================================
def build_association_rules_from_eclass_history(bi_l1: pd.DataFrame, min_support_frac=0.005, min_confidence=0.08):
    buyer_items = bi_l1.groupby("buyer")["item"].apply(set).to_dict()
    n_buyers = len(buyer_items)
    min_count = max(int(n_buyers * min_support_frac), 5)

    freq = Counter()
    for items in buyer_items.values():
        for i in items:
            freq[i] += 1

    frequent = {i for i, c in freq.items() if c >= min_count}
    pair_counts = Counter()

    for items in buyer_items.values():
        arr = sorted(i for i in items if i in frequent)
        for a in range(len(arr)):
            for b in range(a + 1, len(arr)):
                pair_counts[(arr[a], arr[b])] += 1

    rules = defaultdict(list)
    for (a, b), cnt in pair_counts.items():
        lift = cnt * n_buyers / max(freq[a] * freq[b], 1)
        if lift <= 1.0:
            continue

        conf_ab = cnt / max(freq[a], 1)
        conf_ba = cnt / max(freq[b], 1)

        if conf_ab >= min_confidence:
            rules[a].append((b, conf_ab, lift))
        if conf_ba >= min_confidence:
            rules[b].append((a, conf_ba, lift))

    return rules


# =============================================================================
# LEVEL 1 SCORING
# =============================================================================
def direct_score_l1(buyer_id: str, bi_idx: dict) -> dict:
    bdata = bi_idx.get(buyer_id)
    if bdata is None or bdata.empty:
        return {}

    total_orders = max(bdata["n_orders"].sum(), 1)
    total_spend = max(bdata["total_value"].sum(), 1e-9)

    out = {}
    for _, r in bdata.iterrows():
        freq = r["n_orders"] / max(r["span_months"], 1)
        month_cov = r["n_months"] / max(r["span_months"], 1)
        spend = np.log1p(r["total_value"])
        spend_share = r["total_value"] / total_spend
        order_share = r["n_orders"] / total_orders
        rec = np.exp(-0.08 * r["recency"])
        econ = np.sqrt(max(r["avg_price"], 0.01))

        score = (
            0.24 * spend
            + 0.22 * freq
            + 0.18 * month_cov
            + 0.14 * spend_share
            + 0.08 * order_share
            + 0.08 * econ
            + 0.06 * rec
        )
        out[r["item"]] = score

    return out


def user_cf_score(buyer_id, mat_norm, buyer_enc, item_enc, K=CF_K):
    try:
        bidx = np.searchsorted(buyer_enc.classes_, buyer_id)
        if bidx >= len(buyer_enc.classes_) or buyer_enc.classes_[bidx] != buyer_id:
            return {}
    except Exception:
        return {}

    bvec = mat_norm[bidx]
    sims = mat_norm.dot(bvec.T).toarray().ravel()
    sims[bidx] = -1
    top_k = np.argpartition(sims, -K)[-K:]
    own_items = set(bvec.nonzero()[1])

    scores = defaultdict(float)
    for nidx in top_k:
        sim = sims[nidx]
        if sim <= 0:
            continue

        nitems = mat_norm[nidx].nonzero()[1]
        nvals = mat_norm[nidx, nitems].toarray().ravel()

        if len(nitems) > 40:
            top = np.argpartition(nvals, -40)[-40:]
            nitems = nitems[top]
            nvals = nvals[top]

        for iidx, val in zip(nitems, nvals):
            if iidx not in own_items:
                scores[iidx] += sim * val

    return {item_enc.inverse_transform([i])[0]: s for i, s in scores.items()}


def assoc_rule_score_l1(buyer_id: str, bi_idx: dict, rules_dict: dict) -> dict:
    bdata = bi_idx.get(buyer_id)
    if bdata is None or bdata.empty:
        return {}

    own = set(bdata["item"])
    scores = defaultdict(float)

    for eclass in own:
        for con, conf, lift in rules_dict.get(eclass, []):
            if con not in own:
                scores[con] += conf * lift

    return dict(scores)


def score_warm_buyer_l1(buyer_id, bi_idx, mat, mat_norm, buyer_enc, item_enc, rules_dict):
    d = normalize_dict(direct_score_l1(buyer_id, bi_idx))
    u = normalize_dict(user_cf_score(buyer_id, mat_norm, buyer_enc, item_enc, K=CF_K))
    a = normalize_dict(assoc_rule_score_l1(buyer_id, bi_idx, rules_dict))

    all_items = set(d) | set(u) | set(a)
    out = {}

    for item in all_items:
        out[item] = (
            0.82 * d.get(item, 0.0)
            + 0.04 * u.get(item, 0.0)
            + 0.14 * a.get(item, 0.0)
        )

    return out


# =============================================================================
# COLD START - LEVEL 1
# =============================================================================
def build_industry_profiles_l1(bi_l1: pd.DataFrame, bp: pd.DataFrame, min_seg_buyers=8):
    """
    Build hierarchical profiles:
    - (nace_2digits, emp_bucket)
    - (nace_section, emp_bucket)
    - nace_2digits
    - nace_section
    - emp_bucket
    - global
    """
    merged = bi_l1.merge(bp, left_on="buyer", right_on="buyer", how="left")

    p_n2_emp = {}
    p_sec_emp = {}
    p_n2 = {}
    p_sec = {}
    p_emp = {}

    def make_profile(grp):
        n_b = grp["buyer"].nunique()
        stats = grp.groupby("item").agg(
            n_buyers=("buyer", "nunique"),
            avg_value=("total_value", "mean"),
            avg_months=("n_months", "mean"),
            avg_orders=("n_orders", "mean"),
        ).reset_index()
        stats["prevalence"] = stats["n_buyers"] / max(n_b, 1)
        stats["score"] = (
            0.44 * stats["prevalence"]
            + 0.24 * np.log1p(stats["avg_value"])
            + 0.17 * np.log1p(stats["avg_months"])
            + 0.15 * np.log1p(stats["avg_orders"])
        )
        return stats.sort_values("score", ascending=False), n_b

    for (n2, empb), grp in merged.groupby(["nace_2digits", "emp_bucket"]):
        if pd.isna(n2):
            continue
        prof, n_b = make_profile(grp)
        if n_b >= min_seg_buyers:
            p_n2_emp[(n2, empb)] = prof

    for (sec, empb), grp in merged.groupby(["nace_section", "emp_bucket"]):
        prof, n_b = make_profile(grp)
        if n_b >= min_seg_buyers:
            p_sec_emp[(sec, empb)] = prof

    for n2, grp in merged.groupby("nace_2digits"):
        if pd.isna(n2):
            continue
        prof, n_b = make_profile(grp)
        if n_b >= min_seg_buyers:
            p_n2[n2] = prof

    for sec, grp in merged.groupby("nace_section"):
        prof, n_b = make_profile(grp)
        if n_b >= min_seg_buyers:
            p_sec[sec] = prof

    for empb, grp in merged.groupby("emp_bucket"):
        prof, n_b = make_profile(grp)
        if n_b >= min_seg_buyers:
            p_emp[empb] = prof

    global_profile, _ = make_profile(merged)

    return p_n2_emp, p_sec_emp, p_n2, p_sec, p_emp, global_profile


def predict_cold_l1(
    buyer_row,
    p_n2_emp,
    p_sec_emp,
    p_n2,
    p_sec,
    p_emp,
    global_profile,
    valid_items,
    top_n=180,
    cfg=None,
):
    nace2 = buyer_row.get("nace_2digits", None)
    nace_sec = buyer_row.get("nace_section", None)
    empb = emp_bucket(buyer_row.get("estimated_number_employees", None))

    candidates = None
    min_prev = 0.01

    if nace2 and (nace2, empb) in p_n2_emp:
        candidates = p_n2_emp[(nace2, empb)]
        min_prev = cfg["cold_min_prev_nace2"]
    elif (nace_sec, empb) in p_sec_emp:
        candidates = p_sec_emp[(nace_sec, empb)]
        min_prev = cfg["cold_min_prev_section"]
    elif nace2 and nace2 in p_n2:
        candidates = p_n2[nace2]
        min_prev = cfg["cold_min_prev_nace2"]
    elif nace_sec in p_sec:
        candidates = p_sec[nace_sec]
        min_prev = cfg["cold_min_prev_section"]
    elif empb in p_emp:
        candidates = p_emp[empb]
        min_prev = cfg["cold_min_prev_section"]
    else:
        candidates = global_profile
        min_prev = cfg["cold_min_prev_global"]

    candidates = candidates[candidates["prevalence"] >= min_prev]
    candidates = candidates[candidates["item"].isin(valid_items)]
    return candidates.head(top_n)["item"].tolist()


# =============================================================================
# MANUFACTURER MAPS / CONFIDENCE
# =============================================================================
def build_manufacturer_maps(mfr_pref, seg_mfr, nace2_mfr, global_mfr):
    buyer_eclass_pref = {}
    buyer_eclass_meta = {}

    if not mfr_pref.empty:
        grp = mfr_pref.groupby(["buyer", "eclass"])
        for key, sub in grp:
            total_orders = max(sub["n_orders"].sum(), 1)
            total_months = max(sub["n_months"].sum(), 1)
            total_value = max(sub["total_value"].sum(), 1e-9)

            scored = []
            for _, r in sub.iterrows():
                share_orders = r["n_orders"] / total_orders
                share_months = r["n_months"] / total_months
                share_value = r["total_value"] / total_value
                rec = np.exp(-0.08 * r["recency"])

                score = (
                    0.45 * share_orders
                    + 0.25 * share_months
                    + 0.20 * share_value
                    + 0.10 * rec
                )

                scored.append({
                    "manufacturer": r["manufacturer"],
                    "score": score,
                    "share_orders": share_orders,
                    "share_months": share_months,
                    "share_value": share_value,
                })

            scored = sorted(scored, key=lambda x: -x["score"])
            buyer_eclass_pref[key] = scored

            if scored:
                top1 = scored[0]
                top2 = scored[1] if len(scored) > 1 else {"score": 0.0, "share_orders": 0.0}
                buyer_eclass_meta[key] = {
                    "top_score": top1["score"],
                    "top_share_orders": top1["share_orders"],
                    "top_share_months": top1["share_months"],
                    "top_share_value": top1["share_value"],
                    "gap": top1["score"] - top2["score"],
                    "top2_share_orders": top2["share_orders"],
                    "n_manufacturers": len(scored),
                }

    def build_pref(df, keys):
        pref = {}
        if df.empty:
            return pref
        grp = df.groupby(keys)
        for key, sub in grp:
            total_buyers = max(sub["n_buyers"].sum(), 1)
            total_orders = max(sub["n_orders"].sum(), 1)
            total_value = max(sub["total_value"].sum(), 1e-9)
            scored = []
            for _, r in sub.iterrows():
                score = (
                    0.45 * (r["n_buyers"] / total_buyers)
                    + 0.30 * (r["n_orders"] / total_orders)
                    + 0.25 * (r["total_value"] / total_value)
                )
                scored.append((r["manufacturer"], score))
            scored.sort(key=lambda x: -x[1])
            pref[key] = scored
        return pref

    seg_pref = build_pref(seg_mfr, ["nace_section", "eclass"])
    nace2_pref = build_pref(nace2_mfr, ["nace_2digits", "eclass"])

    global_pref = {}
    eclass_stability = {}

    if not global_mfr.empty:
        grp = global_mfr.groupby("eclass")
        for eclass, sub in grp:
            total_buyers = max(sub["n_buyers"].sum(), 1)
            total_orders = max(sub["n_orders"].sum(), 1)
            total_value = max(sub["total_value"].sum(), 1e-9)

            scored = []
            for _, r in sub.iterrows():
                score = (
                    0.45 * (r["n_buyers"] / total_buyers)
                    + 0.30 * (r["n_orders"] / total_orders)
                    + 0.25 * (r["total_value"] / total_value)
                )
                scored.append({
                    "manufacturer": r["manufacturer"],
                    "score": score,
                    "buyer_share": r["n_buyers"] / total_buyers,
                    "order_share": r["n_orders"] / total_orders,
                    "value_share": r["total_value"] / total_value,
                })

            scored = sorted(scored, key=lambda x: -x["score"])
            global_pref[eclass] = [(x["manufacturer"], x["score"]) for x in scored]

            if scored:
                top1 = scored[0]
                top2_score = scored[1]["score"] if len(scored) > 1 else 0.0
                eclass_stability[eclass] = {
                    "top_score": top1["score"],
                    "top_buyer_share": top1["buyer_share"],
                    "gap": top1["score"] - top2_score,
                    "n_manufacturers": len(scored),
                }

    return buyer_eclass_pref, buyer_eclass_meta, seg_pref, nace2_pref, global_pref, eclass_stability


def choose_manufacturers_with_confidence(
    buyer_id,
    eclass,
    nace_section,
    nace_2digits,
    buyer_eclass_pref,
    buyer_eclass_meta,
    seg_pref,
    nace2_pref,
    global_pref,
    eclass_stability,
    cfg
):
    """
    Returns a list of 0, 1, or 2 manufacturers.
    """
    stab = eclass_stability.get(eclass)
    if stab is None:
        return []

    if stab["top_buyer_share"] < cfg["min_eclass_top_mfr_share_global"]:
        return []

    bkey = (buyer_id, eclass)
    if bkey in buyer_eclass_pref and bkey in buyer_eclass_meta:
        scored = buyer_eclass_pref[bkey]
        meta = buyer_eclass_meta[bkey]
        top1 = scored[0]["manufacturer"]

        # strong one-brand case
        if (
            meta["top_share_orders"] >= cfg["min_top_mfr_order_share"]
            and meta["top_share_months"] >= cfg["min_top_mfr_month_share"]
            and meta["gap"] >= cfg["min_top_mfr_gap"]
        ):
            return [top1]

        # split-brand case -> emit top2 if both are meaningful
        if len(scored) > 1:
            top2 = scored[1]["manufacturer"]
            if (
                meta["top_share_orders"] < cfg["emit_top2_when_top1_share_below"]
                and meta["top_share_orders"] >= cfg["emit_top2_min_top1_share"]
                and meta["top2_share_orders"] >= cfg["emit_top2_min_top2_share"]
            ):
                return [top1, top2]

    # nace_2digits fallback
    n2key = (nace_2digits, eclass)
    if nace_2digits and n2key in nace2_pref and len(nace2_pref[n2key]) > 0:
        top_mfr, top_score = nace2_pref[n2key][0]
        if top_score >= cfg["min_seg_mfr_score"]:
            return [top_mfr]

    # nace_section fallback
    skey = (nace_section, eclass)
    if skey in seg_pref and len(seg_pref[skey]) > 0:
        top_mfr, top_score = seg_pref[skey][0]
        if top_score >= cfg["min_seg_mfr_score"]:
            return [top_mfr]

    # global fallback
    if eclass in global_pref and len(global_pref[eclass]) > 0:
        top_mfr, top_score = global_pref[eclass][0]
        if top_score >= cfg["min_global_mfr_score"]:
            return [top_mfr]

    return []


# =============================================================================
# PREDICTION PIPELINES
# =============================================================================
def predict_level_1(data, cfg):
    log("Running Level 1 pipeline...")

    bi = data["bi_l1"]
    ig = data["ig_l1"]
    bp = data["bp"]

    mat, mat_norm, buyer_enc, item_enc = build_sparse(bi, ig, MIN_GLOBAL_BUYERS_L1)
    bi_idx = build_buyer_index(bi)
    rules_dict = build_association_rules_from_eclass_history(bi)
    p_n2_emp, p_sec_emp, p_n2, p_sec, p_emp, global_profile = build_industry_profiles_l1(bi, bp)

    valid_items = set(ig[ig["n_buyers"] >= MIN_GLOBAL_BUYERS_L1]["item"])

    test = pd.read_csv(TEST_PATH, dtype={"legal_entity_id": str, "nace_code": str, "nace_section": str})
    test["nace_code"] = test["nace_code"].fillna("").astype(str).str.strip()
    test["nace_2digits"] = test["nace_code"].map(safe_nace2)

    les = pd.read_csv(LES_CS, dtype={"legal_entity_id": str})
    test = test.merge(les, on="legal_entity_id", how="left")

    preds = []

    for idx, row in enumerate(test.itertuples(index=False), start=1):
        bid = row.legal_entity_id
        is_warm = (row.cs == 1) and (bid in bi_idx)

        if is_warm:
            scores = score_warm_buyer_l1(
                bid, bi_idx, mat, mat_norm, buyer_enc, item_enc, rules_dict
            )

            ordered = [
                item for item, s in sorted(scores.items(), key=lambda x: -x[1])
                if s >= cfg["warm_threshold_l1"] and item in valid_items
            ]

            n_hist = bi_idx[bid]["item"].nunique()
            dynamic_cap = min(WARM_TOP_N_L1, max(350, int(0.55 * n_hist)))
            buyer_preds = ordered[:dynamic_cap]
        else:
            buyer_preds = predict_cold_l1(
                row._asdict(),
                p_n2_emp,
                p_sec_emp,
                p_n2,
                p_sec,
                p_emp,
                global_profile,
                valid_items,
                top_n=cfg["cold_top_n_l1"],
                cfg=cfg,
            )

        for item in buyer_preds:
            preds.append({"buyer_id": bid, "predicted_id": item})

        if idx % 10 == 0:
            log(f"L1 buyers processed: {idx}/{len(test)}")

    sub = pd.DataFrame(preds).drop_duplicates(["buyer_id", "predicted_id"])
    return sub


def predict_level_2(data, cfg, sub_l1: pd.DataFrame):
    log("Running Level 2 pipeline...")

    bi_l1 = data["bi_l1"]
    ig_l1 = data["ig_l1"]
    bp = data["bp"]
    mfr_pref = data["mfr_pref"]
    seg_mfr = data["seg_mfr"]
    nace2_mfr = data["nace2_mfr"]
    global_mfr = data["global_mfr"]
    ig_l2 = data["ig_l2"]

    mat_l1, mat_norm_l1, buyer_enc_l1, item_enc_l1 = build_sparse(bi_l1, ig_l1, MIN_GLOBAL_BUYERS_L1)
    bi_idx_l1 = build_buyer_index(bi_l1)
    rules_dict = build_association_rules_from_eclass_history(bi_l1)
    p_n2_emp, p_sec_emp, p_n2, p_sec, p_emp, global_profile = build_industry_profiles_l1(bi_l1, bp)

    (
        buyer_eclass_pref,
        buyer_eclass_meta,
        seg_pref,
        nace2_pref,
        global_pref,
        eclass_stability,
    ) = build_manufacturer_maps(mfr_pref, seg_mfr, nace2_mfr, global_mfr)

    valid_eclasses = set(ig_l1[ig_l1["n_buyers"] >= MIN_GLOBAL_BUYERS_L1]["item"])
    valid_emfr = set(ig_l2[ig_l2["n_buyers"] >= MIN_GLOBAL_BUYERS_L2]["item"])

    test = pd.read_csv(TEST_PATH, dtype={"legal_entity_id": str, "nace_code": str, "nace_section": str})
    test["nace_code"] = test["nace_code"].fillna("").astype(str).str.strip()
    test["nace_2digits"] = test["nace_code"].map(safe_nace2)

    les = pd.read_csv(LES_CS, dtype={"legal_entity_id": str})
    test = test.merge(les, on="legal_entity_id", how="left")

    l1_by_buyer = sub_l1.groupby("buyer_id")["predicted_id"].apply(list).to_dict()
    preds = []

    for idx, row in enumerate(test.itertuples(index=False), start=1):
        bid = row.legal_entity_id
        nace = row.nace_section
        nace2 = row.nace_2digits
        is_warm = (row.cs == 1) and (bid in bi_idx_l1)

        if is_warm:
            scores = score_warm_buyer_l1(
                bid, bi_idx_l1, mat_l1, mat_norm_l1, buyer_enc_l1, item_enc_l1, rules_dict
            )

            ordered_eclass = [
                item for item, s in sorted(scores.items(), key=lambda x: -x[1])
                if s >= cfg["warm_threshold_l2_eclass"] and item in valid_eclasses
            ]

            hist_cap = min(WARM_TOP_N_L2_ECLASS, max(140, int(0.32 * bi_idx_l1[bid]["item"].nunique())))
            eclass_candidates = ordered_eclass[:hist_cap]
        else:
            eclass_candidates = l1_by_buyer.get(bid, [])[:cfg["cold_top_n_l2_eclass"]]
            if not eclass_candidates:
                eclass_candidates = predict_cold_l1(
                    row._asdict(),
                    p_n2_emp,
                    p_sec_emp,
                    p_n2,
                    p_sec,
                    p_emp,
                    global_profile,
                    valid_eclasses,
                    top_n=cfg["cold_top_n_l2_eclass"],
                    cfg=cfg,
                )

        buyer_preds = []
        seen_emfr = set()

        for eclass in eclass_candidates:
            manufacturers = choose_manufacturers_with_confidence(
                buyer_id=bid,
                eclass=eclass,
                nace_section=nace,
                nace_2digits=nace2,
                buyer_eclass_pref=buyer_eclass_pref,
                buyer_eclass_meta=buyer_eclass_meta,
                seg_pref=seg_pref,
                nace2_pref=nace2_pref,
                global_pref=global_pref,
                eclass_stability=eclass_stability,
                cfg=cfg,
            )

            for mfr in manufacturers:
                emfr = f"{eclass}|{mfr}"
                if emfr in valid_emfr and emfr not in seen_emfr:
                    seen_emfr.add(emfr)
                    buyer_preds.append(emfr)

        for item in buyer_preds:
            preds.append({"buyer_id": bid, "predicted_id": item})

        if idx % 10 == 0:
            log(f"L2 buyers processed: {idx}/{len(test)}")

    sub = pd.DataFrame(preds).drop_duplicates(["buyer_id", "predicted_id"])
    return sub


# #############################################################################
# >>>>>>>>>>>>>>>>>>>>>>>  LEVEL 3 STARTS HERE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# #############################################################################
# =============================================================================
# LEVEL 3 — TEXT NORMALIZATION & FEATURE EXTRACTION
# =============================================================================

# Layer A: structured regex patterns for feature extraction
_RE_SIZE  = re.compile(r'\b(xs|xxl|xl|x[sl]|\d+\s*mm|\d+\s*cm|\d+\s*ml|\d+\s*l\b|\d+\s*kg|[sml]\b)\b', re.I)
_RE_MAT   = re.compile(r'\b(nitrile|latex|vinyl|neoprene|polyester|cotton|stahl|steel|alumin\w+|pp\b|pe\b|pvc)\b', re.I)
_RE_NORM  = re.compile(r'\b(EN\s*\d+|ISO\s*\d+|DIN\s*\d+|CE\s*\d*|FDA)\b', re.I)
_RE_PACK  = re.compile(r'\b(\d+\s*st[üu]ck|\d+\s*pcs|\d+\s*pack|\d+er\s*pack|\d+\s*rollen?)\b', re.I)
_RE_COLOR = re.compile(r'\b(wei[ßs]|schwarz|blau|rot|gelb|gr[üu]n|transparent|white|black|blue|red|yellow|green)\b', re.I)
_RE_COND  = re.compile(r'\b(steril|unsteril|puderfrei|powder.?free|latexfrei|latex.?free|ungepudert)\b', re.I)
_RE_NUM   = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(mm|cm|ml|kg|bar|mpa|kpa|m\b|g\b|w\b|v\b|a\b)?', re.I)


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower().strip()


def _bin_numeric(val: float, unit: str) -> str:
    unit = (unit or "").lower()
    if unit == "mm":
        if val < 10:   return "num_0_10mm"
        if val < 50:   return "num_10_50mm"
        if val < 200:  return "num_50_200mm"
        return "num_200plusmm"
    if unit == "cm":
        if val < 5:    return "num_0_5cm"
        if val < 25:   return "num_5_25cm"
        if val < 100:  return "num_25_100cm"
        return "num_100pluscm"
    if unit == "kg":
        if val < 1:    return "num_0_1kg"
        if val < 10:   return "num_1_10kg"
        if val < 50:   return "num_10_50kg"
        return "num_50pluskg"
    if val < 10:   return "num_small"
    if val < 100:  return "num_medium"
    if val < 1000: return "num_large"
    return "num_xlarge"


def _extract_feature_tokens(text: str) -> list:
    """Layer A: regex-based structured feature extraction."""
    tokens = []
    for m in _RE_SIZE.finditer(text):
        tokens.append("sz_" + m.group(1).lower().replace(" ", ""))
    for m in _RE_MAT.finditer(text):
        tokens.append("mat_" + m.group(1).lower())
    for m in _RE_NORM.finditer(text):
        tokens.append("nrm_" + m.group(1).lower().replace(" ", ""))
    for m in _RE_PACK.finditer(text):
        tokens.append("pkg_" + m.group(1).lower().replace(" ", ""))
    for m in _RE_COLOR.finditer(text):
        tokens.append("col_" + m.group(1).lower())
    for m in _RE_COND.finditer(text):
        tokens.append("cnd_" + m.group(1).lower().replace(" ", "").replace("-", "").replace(".", ""))
    # Layer C: numeric binning
    for m in _RE_NUM.finditer(text):
        try:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2) or ""
            tokens.append(_bin_numeric(val, unit))
        except Exception:
            pass
    return tokens


# =============================================================================
# LEVEL 3 — STEP 0: RAW DATA LOADING
# =============================================================================

def _load_l3_raw_data():
    """
    Second-pass chunked reader for L3 product text and buyer-SKU transactions.

    Returns
    -------
    sku_text       : {sku_id: (eclass, raw_text)}
    sku_stats      : {sku_id: {"buyers": set, "orders": int, "total_value": float}}
    buyer_sku_data : {(buyer, sku_id): {"months": set, "order_dates": set,
                                        "total_value": float, "prices": list,
                                        "last_ym": Period, "eclass": str}}
    sku_manufacturer : {sku_id: str}  — most common manufacturer per SKU
    """
    log("[L3] Step 0 — Loading L3 product text and transaction data...")

    # Inspect CSV header to discover available columns
    try:
        header_df = pd.read_csv(TRAIN_PATH, nrows=0)
        all_cols = set(header_df.columns.str.strip())
    except Exception as e:
        log(f"[L3] Cannot read CSV header ({e}); skipping L3 data load")
        return {}, {}, {}, {}

    required = {"legal_entity_id", "eclass", "orderdate", "quantityvalue", "vk_per_item"}
    if not required.issubset(all_cols):
        log("[L3] Required columns missing; skipping L3 data load")
        return {}, {}, {}, {}

    load_cols = list(required)

    # Discover SKU column (try several candidate names)
    sku_col = None
    for candidate in ("sku", "product_id", "article_id", "item_id", "ean", "article_number"):
        if candidate in all_cols:
            sku_col = candidate
            load_cols.append(sku_col)
            log(f"[L3] Using '{sku_col}' as SKU column")
            break
    if sku_col is None:
        log("[L3] No SKU column found; using eclass|manufacturer as pseudo-SKU")

    # Optional manufacturer column
    mfr_col = "manufacturer" if "manufacturer" in all_cols else None
    if mfr_col:
        load_cols.append(mfr_col)

    # Optional product text columns
    text_cols_avail = [c for c in _L3_TEXT_COLS if c in all_cols]
    load_cols += text_cols_avail
    if text_cols_avail:
        log(f"[L3] Text columns found: {text_cols_avail}")
    else:
        log("[L3] No product text columns found; will use manufacturer-based clustering fallback")

    sku_text = {}
    sku_stats = {}
    buyer_sku_data = {}
    sku_mfr_counter = defaultdict(Counter)

    rows_loaded = 0
    dtype_map = {"legal_entity_id": str, "eclass": str}
    if mfr_col:
        dtype_map[mfr_col] = str

    for chunk in pd.read_csv(TRAIN_PATH, usecols=load_cols, chunksize=CHUNK, dtype=dtype_map):
        chunk["eclass"] = chunk["eclass"].fillna("").astype(str).str.strip()
        chunk["legal_entity_id"] = chunk["legal_entity_id"].astype(str)
        chunk = chunk[chunk["eclass"] != ""].copy()

        chunk["dt"] = pd.to_datetime(chunk["orderdate"], errors="coerce")
        chunk = chunk.dropna(subset=["dt"])
        chunk["ym"] = chunk["dt"].dt.to_period("M")
        chunk["quantityvalue"] = pd.to_numeric(chunk["quantityvalue"], errors="coerce").fillna(1)
        chunk["vk_per_item"] = pd.to_numeric(chunk["vk_per_item"], errors="coerce").fillna(0)
        chunk["line_val"] = (chunk["quantityvalue"] * chunk["vk_per_item"]).clip(lower=0)

        # Build raw_text by concatenating available text fields
        if text_cols_avail:
            chunk["raw_text"] = (
                chunk[text_cols_avail]
                .fillna("")
                .apply(lambda r: " ".join(str(v) for v in r if str(v).strip()), axis=1)
                .apply(_normalize_text)
            )
        else:
            chunk["raw_text"] = ""

        # Build standardized sku_id column
        if sku_col and sku_col in chunk.columns:
            chunk["sku_id"] = chunk[sku_col].fillna("").astype(str).str.strip()
        else:
            mfr_vals = (
                chunk[mfr_col].fillna("").astype(str).str.strip()
                if mfr_col
                else pd.Series("", index=chunk.index)
            )
            chunk["sku_id"] = chunk["eclass"] + "|" + mfr_vals

        for row in chunk.itertuples(index=False):
            buyer   = row.legal_entity_id
            eclass  = row.eclass
            sku     = row.sku_id
            ym      = row.ym
            lv      = row.line_val
            price   = row.vk_per_item
            raw_txt = row.raw_text
            order_date = str(row.dt.date())

            if sku not in sku_text:
                sku_text[sku] = (eclass, raw_txt)

            if sku not in sku_stats:
                sku_stats[sku] = {"buyers": set(), "orders": 0, "total_value": 0.0}
            sku_stats[sku]["buyers"].add(buyer)
            sku_stats[sku]["orders"] += 1
            sku_stats[sku]["total_value"] += lv

            bsk = (buyer, sku)
            if bsk not in buyer_sku_data:
                buyer_sku_data[bsk] = {
                    "months": set(), "order_dates": set(),
                    "total_value": 0.0, "prices": [],
                    "last_ym": ym, "eclass": eclass,
                }
            bsd = buyer_sku_data[bsk]
            bsd["months"].add(ym)
            bsd["order_dates"].add(order_date)
            bsd["total_value"] += lv
            bsd["prices"].append(price)
            if ym > bsd["last_ym"]:
                bsd["last_ym"] = ym

            if mfr_col:
                mfr_val = getattr(row, mfr_col, "")
                if mfr_val and str(mfr_val).strip():
                    sku_mfr_counter[sku][str(mfr_val).strip()] += 1

        rows_loaded += len(chunk)
        log(f"[L3] Processed {rows_loaded:,} rows")

    # Derive most common manufacturer per SKU
    sku_manufacturer = {
        sku: ctr.most_common(1)[0][0]
        for sku, ctr in sku_mfr_counter.items()
        if ctr
    }

    log(f"[L3] Loaded {len(sku_text):,} unique SKUs, {len(buyer_sku_data):,} buyer-SKU pairs")
    return sku_text, sku_stats, buyer_sku_data, sku_manufacturer


# =============================================================================
# LEVEL 3 — STEPS 1-4: CLUSTERING & BUYER SCORING
# =============================================================================

def build_l3_clusters(sku_text, sku_stats, buyer_sku_data, sku_manufacturer, bp):
    """
    Steps 1-4 of the L3 pipeline.

    Step 1: Extract feature fingerprints per SKU (Layer A regex + within-eclass TF-IDF).
    Step 2: Within-eclass clustering with adaptive k selection.
    Step 3: Cluster quality filter (stable vs noise).
    Step 4: Buyer-cluster scoring using the same economic proxy as L1.

    Returns dict with keys:
        cluster_map     : {sku_id → cluster_id}
        buyer_clusters  : {(buyer, cluster_id) → stats_dict}
        cluster_stats   : {cluster_id → stats_dict}
        seg_clusters    : {(nace_section, cluster_id) → segment_score}
    """
    if not sku_text:
        log("[L3] No SKU data; L3 will produce empty output")
        return {"cluster_map": {}, "buyer_clusters": {}, "cluster_stats": {}, "seg_clusters": {}}

    # --- Step 1: Group SKUs by eclass (memory-safe: process one eclass at a time) ---
    log("[L3] Step 1 — Grouping SKUs by eclass...")
    eclass_skus = defaultdict(list)
    for sku, (eclass, _) in sku_text.items():
        eclass_skus[eclass].append(sku)

    sku_to_cluster = {}          # sku_id → cluster_id string
    cluster_sku_lists = defaultdict(list)  # cluster_id → [sku_id, ...]

    # --- Step 2: Within-eclass clustering ---
    log(f"[L3] Step 2 — Clustering {len(eclass_skus):,} eclasses (adaptive k)...")

    for eclass, skus in eclass_skus.items():
        n_skus = len(skus)
        n_buyers_ec = len({b for s in skus for b in sku_stats.get(s, {}).get("buyers", set())})

        # Eclasses too small to split: assign monolithic cluster C00
        if n_skus < 5 or n_buyers_ec < 2:
            cid = f"{eclass}_C00"
            for s in skus:
                sku_to_cluster[s] = cid
                cluster_sku_lists[cid].append(s)
            continue

        # Build feature strings: Layer A tokens + raw text (within-eclass IDF via TfidfVectorizer)
        feature_strings = []
        for s in skus:
            _, raw_text = sku_text[s]
            layer_a = _extract_feature_tokens(raw_text)
            combined = (" ".join(layer_a) + " " + raw_text).strip()
            feature_strings.append(combined if combined else "unknown")

        # If virtually no text content, use manufacturer-based clustering as fallback
        non_empty = sum(1 for fs in feature_strings if fs not in ("", "unknown"))
        if non_empty < 3:
            mfr_groups = defaultdict(list)
            for s in skus:
                mfr_groups[sku_manufacturer.get(s, "unknown")].append(s)
            # Assign labels by group size descending (deterministic)
            for label, (mfr, grp) in enumerate(
                sorted(mfr_groups.items(), key=lambda x: -len(x[1]))
            ):
                cid = f"{eclass}_C{label:02d}"
                for s in grp:
                    sku_to_cluster[s] = cid
                    cluster_sku_lists[cid].append(s)
            continue

        # TF-IDF vectorization (within-eclass IDF, not global)
        try:
            min_df = max(1, int(n_skus * 0.02))
            vec = TfidfVectorizer(
                max_features=200, min_df=min_df, max_df=0.95,
                sublinear_tf=True, token_pattern=r"[a-z0-9_]+",
            )
            X = vec.fit_transform(feature_strings)
        except Exception:
            cid = f"{eclass}_C00"
            for s in skus:
                sku_to_cluster[s] = cid
                cluster_sku_lists[cid].append(s)
            continue

        # Adaptive k: try k=2..8, pick by silhouette score; skip if too few SKUs
        if n_skus < 10:
            best_k = 2
        else:
            best_k, best_sil = 2, -1.0
            k_max = min(9, n_skus // 2 + 1)
            for k in range(2, k_max):
                try:
                    if n_skus < 200:
                        labels_k = AgglomerativeClustering(
                            n_clusters=k, linkage="ward"
                        ).fit_predict(X.toarray())
                    else:
                        labels_k = KMeans(
                            n_clusters=k, random_state=42, n_init=10
                        ).fit_predict(X)
                    if len(set(labels_k)) < 2:
                        continue
                    sil = silhouette_score(
                        X, labels_k,
                        sample_size=min(500, n_skus),
                        random_state=42,
                    )
                    if sil > best_sil:
                        best_sil = sil
                        best_k = k
                except Exception:
                    break

        # Final clustering with best_k
        try:
            if n_skus < 200:
                labels = AgglomerativeClustering(
                    n_clusters=best_k, linkage="ward"
                ).fit_predict(X.toarray())
            else:
                labels = KMeans(
                    n_clusters=best_k, random_state=42, n_init=10
                ).fit_predict(X)
        except Exception:
            labels = [0] * n_skus

        # Remap cluster labels by size descending → deterministic C00, C01, ...
        label_counts = Counter(labels)
        sorted_labels = [lbl for lbl, _ in sorted(label_counts.items(), key=lambda x: -x[1])]
        label_remap = {old: new for new, old in enumerate(sorted_labels)}

        for i, s in enumerate(skus):
            remapped = label_remap[labels[i]]
            cid = f"{eclass}_C{remapped:02d}"
            sku_to_cluster[s] = cid
            cluster_sku_lists[cid].append(s)

    # --- Step 3: Cluster quality filter ---
    log(f"[L3] Step 3 — Quality filtering {len(cluster_sku_lists):,} clusters...")

    # Efficiently aggregate buyer/month/value stats per cluster in one pass
    cluster_buyers_set  = defaultdict(set)
    cluster_months_set  = defaultdict(set)
    cluster_orders_cnt  = defaultdict(int)
    cluster_value_total = defaultdict(float)

    for (buyer, sku), bsd in buyer_sku_data.items():
        cid = sku_to_cluster.get(sku)
        if cid is None:
            continue
        cluster_buyers_set[cid].add(buyer)
        cluster_months_set[cid].update(bsd["months"])
        cluster_orders_cnt[cid] += len(bsd["order_dates"])
        cluster_value_total[cid] += bsd["total_value"]

    cluster_stats = {}

    for cid, skus in cluster_sku_lists.items():
        eclass = cid.rsplit("_C", 1)[0]
        n_skus_in   = len(skus)
        n_buyers    = len(cluster_buyers_set[cid])
        n_months    = len(cluster_months_set[cid])
        orders_tot  = cluster_orders_cnt[cid]
        value_tot   = cluster_value_total[cid]

        # Per-SKU feature token sets for coherence check
        token_counter = Counter()
        sku_token_sets = []
        for s in skus:
            _, raw_text = sku_text.get(s, (eclass, ""))
            tok_set = set(_extract_feature_tokens(raw_text))
            sku_token_sets.append(tok_set)
            token_counter.update(tok_set)

        # Stability criteria: ≥2 buyers AND ≥2 distinct months
        is_stable = n_buyers >= 2 and n_months >= 2

        # Coherence sub-check (relaxed for small clusters):
        # at least 30% of SKUs share the top feature token, unless cluster has many buyers
        if is_stable and token_counter and n_skus_in >= 3:
            top_tok = token_counter.most_common(1)[0][0]
            skus_with_top = sum(1 for ts in sku_token_sets if top_tok in ts)
            coherence = skus_with_top / n_skus_in
            if coherence < 0.30 and n_buyers < 5:
                is_stable = False

        top_features = [tok for tok, _ in token_counter.most_common(5)]

        cluster_stats[cid] = {
            "n_skus":      n_skus_in,
            "n_buyers":    n_buyers,
            "n_orders":    orders_tot,
            "n_months":    n_months,
            "total_value": value_tot,
            "avg_value":   value_tot / max(orders_tot, 1),
            "top_features": top_features,
            "eclass":      eclass,
            "is_stable":   is_stable,
        }

    stable_clusters = {cid for cid, st in cluster_stats.items() if st["is_stable"]}
    log(f"[L3] {len(stable_clusters):,} stable clusters out of {len(cluster_stats):,} total")

    # --- Step 4: Buyer-cluster scoring ---
    log("[L3] Step 4 — Scoring buyer-cluster pairs...")

    # Find global max_ym for recency computation
    all_last_yms = [bsd["last_ym"] for bsd in buyer_sku_data.values()]
    global_max_ym = max(all_last_yms) if all_last_yms else None

    buyer_cluster_agg = defaultdict(lambda: {
        "n_order_dates": 0, "distinct_months": set(),
        "total_value": 0.0, "prices": [], "last_ym": None,
    })

    for (buyer, sku), bsd in buyer_sku_data.items():
        cid = sku_to_cluster.get(sku)
        if cid is None or cid not in stable_clusters:
            continue
        key = (buyer, cid)
        agg = buyer_cluster_agg[key]
        agg["n_order_dates"] += len(bsd["order_dates"])
        agg["distinct_months"].update(bsd["months"])
        agg["total_value"] += bsd["total_value"]
        agg["prices"].extend(bsd["prices"])
        if agg["last_ym"] is None or bsd["last_ym"] > agg["last_ym"]:
            agg["last_ym"] = bsd["last_ym"]

    buyer_clusters = {}
    for (buyer, cid), agg in buyer_cluster_agg.items():
        distinct_months = len(agg["distinct_months"])
        avg_price = float(np.mean(agg["prices"])) if agg["prices"] else 0.0
        last_ym   = agg["last_ym"]

        recency_months = 999
        if last_ym is not None and global_max_ym is not None:
            recency_months = (
                (global_max_ym.year  - last_ym.year)  * 12 +
                (global_max_ym.month - last_ym.month)
            )

        if   recency_months <= 3:  rec_mult = 2.0
        elif recency_months <= 6:  rec_mult = 1.5
        elif recency_months <= 12: rec_mult = 1.0
        elif recency_months <= 18: rec_mult = 0.7
        else:                      rec_mult = 0.4

        score = np.sqrt(max(avg_price, 0.01)) * distinct_months * rec_mult

        buyer_clusters[(buyer, cid)] = {
            "n_order_dates":  agg["n_order_dates"],
            "distinct_months": distinct_months,
            "total_value":    agg["total_value"],
            "avg_price":      avg_price,
            "last_ym":        last_ym,
            "recency_months": recency_months,
            "score":          score,
        }

    log(f"[L3] {len(buyer_clusters):,} buyer-cluster pairs scored")

    # Build segment-level cluster signals for cold-start
    # seg_clusters[(nace_section, cluster_id)] = n_buyers * sqrt(avg_value)
    bp_nace = {}
    if bp is not None and not bp.empty:
        bp_nace = bp.set_index("buyer")["nace_section"].to_dict()

    seg_buyers_set  = defaultdict(set)
    seg_value_total = defaultdict(float)

    for (buyer, cid), bcs in buyer_clusters.items():
        nace = bp_nace.get(buyer, "")
        if nace:
            seg_buyers_set[(nace, cid)].add(buyer)
            seg_value_total[(nace, cid)] += bcs["total_value"]

    seg_clusters = {
        (nace, cid): len(buyers) * np.sqrt(max(seg_value_total[(nace, cid)] / max(len(buyers), 1), 0.01))
        for (nace, cid), buyers in seg_buyers_set.items()
    }

    log(f"[L3] {len(seg_clusters):,} segment-cluster signals built")

    return {
        "cluster_map":    sku_to_cluster,
        "buyer_clusters": buyer_clusters,
        "cluster_stats":  cluster_stats,
        "seg_clusters":   seg_clusters,
    }


# =============================================================================
# LEVEL 3 — STEP 5: PREDICTION PIPELINE
# =============================================================================

def predict_level_3(data, cfg, sub_l1):
    """
    Level 3 prediction: eclass + feature cluster (cluster_id like '19020102_C04').

    Warm buyers  → direct buyer-cluster scores + NACE cross-sell signal
    Cold buyers  → hierarchical NACE segment cluster ranking
    """
    log("[L3] Running Level 3 pipeline...")

    buyer_clusters = data.get("l3_buyer_clusters", {})
    cluster_stats  = data.get("l3_cluster_stats",  {})
    seg_clusters   = data.get("l3_seg_clusters",   {})
    bp     = data["bp"]
    bi_l1  = data["bi_l1"]

    if not cluster_stats:
        log("[L3] No cluster data available; returning empty L3 submission")
        return pd.DataFrame(columns=["buyer_id", "predicted_id"])

    threshold    = cfg["warm_threshold_l3"]
    cold_top_n   = cfg["cold_top_n_l3"]
    cross_min    = cfg["l3_cross_sell_min_eclass_orders"]

    # ---- Pre-build lookup structures ----------------------------------------

    # Buyers that appear in cluster history
    warm_buyers_l3 = {b for (b, _) in buyer_clusters}

    # buyer → NACE section
    buyer_nace = bp.set_index("buyer")["nace_section"].to_dict() if not bp.empty else {}

    # buyer x eclass → n_orders (for cross-sell threshold)
    buyer_eclass_n = (
        bi_l1.set_index(["buyer", "item"])["n_orders"].to_dict()
        if not bi_l1.empty else {}
    )

    # buyer → {cluster_id: score} for clusters above warm threshold
    buyer_warm_clusters = defaultdict(dict)
    for (buyer, cid), bcs in buyer_clusters.items():
        if cluster_stats.get(cid, {}).get("is_stable", False) and bcs["score"] >= threshold:
            buyer_warm_clusters[buyer][cid] = bcs["score"]

    # (nace_section, eclass) → top-1 stable cluster (cross-sell anchor)
    nace_eclass_cluster_scores = defaultdict(list)
    for (nace, cid), score in seg_clusters.items():
        cs = cluster_stats.get(cid, {})
        if cs.get("is_stable", False):
            eclass = cs.get("eclass", "")
            nace_eclass_cluster_scores[(nace, eclass)].append((cid, score))
    nace_eclass_top_cluster = {
        key: sorted(pairs, key=lambda x: -x[1])[0][0]
        for key, pairs in nace_eclass_cluster_scores.items()
    }

    # nace_section → stable clusters ranked by segment score (cold-start)
    nace_top_clusters = defaultdict(list)
    for (nace, cid), score in seg_clusters.items():
        if cluster_stats.get(cid, {}).get("is_stable", False):
            nace_top_clusters[nace].append((cid, score))
    for nace in nace_top_clusters:
        nace_top_clusters[nace].sort(key=lambda x: -x[1])

    # L1 predictions per buyer (as sets, for cross-sell eclass iteration)
    l1_by_buyer = sub_l1.groupby("buyer_id")["predicted_id"].apply(set).to_dict()

    # ---- Load test set (same pattern as L1/L2) ------------------------------
    test = pd.read_csv(
        TEST_PATH,
        dtype={"legal_entity_id": str, "nace_code": str, "nace_section": str},
    )
    test["nace_code"] = test["nace_code"].fillna("").astype(str).str.strip()
    les = pd.read_csv(LES_CS, dtype={"legal_entity_id": str})
    test = test.merge(les, on="legal_entity_id", how="left")

    preds = []

    for idx, row in enumerate(test.itertuples(index=False), start=1):
        bid  = row.legal_entity_id
        nace = row.nace_section
        is_warm = (row.cs == 1) and (bid in warm_buyers_l3)

        buyer_preds = set()

        if is_warm:
            # 1. Clusters the buyer has directly ordered from (above threshold)
            for cid in buyer_warm_clusters.get(bid, {}):
                buyer_preds.add(cid)

            # 2. Cross-sell: dominant cluster in NACE for eclasses with proven demand
            for eclass in l1_by_buyer.get(bid, set()):
                if buyer_eclass_n.get((bid, eclass), 0) >= cross_min:
                    top_cid = nace_eclass_top_cluster.get((nace, eclass))
                    if top_cid and top_cid not in buyer_preds:
                        if cluster_stats.get(top_cid, {}).get("is_stable", False):
                            buyer_preds.add(top_cid)

        else:
            # Cold start: top-N clusters by NACE segment score
            for cid, _ in nace_top_clusters.get(nace, [])[:cold_top_n]:
                buyer_preds.add(cid)

        for cid in buyer_preds:
            preds.append({"buyer_id": bid, "predicted_id": cid})

        if idx % 10 == 0:
            log(f"[L3] Buyers processed: {idx}/{len(test)}")

    sub = pd.DataFrame(preds).drop_duplicates(["buyer_id", "predicted_id"])
    return sub


# =============================================================================
# LEVEL 3 — STEP 6: CLUSTER MANIFEST
# =============================================================================

def build_cluster_manifest(cluster_stats: dict) -> pd.DataFrame:
    """
    Build a human-readable manifest of all stable clusters for methodology evaluation.
    Output columns: cluster_id, eclass, n_skus, n_buyers, top_features, example_description
    """
    rows = []
    for cid, st in cluster_stats.items():
        if not st.get("is_stable", False):
            continue
        feats = st.get("top_features", [])
        top_feats_str = "|".join(feats[:5])
        # Human-readable description: strip prefix codes from feature tokens
        readable = []
        for f in feats[:3]:
            for prefix in ("mat_", "sz_", "cnd_", "col_", "nrm_", "pkg_", "num_"):
                if f.startswith(prefix):
                    f = f[len(prefix):]
                    break
            readable.append(f.replace("_", " "))
        example = " ".join(readable).strip() or cid
        rows.append({
            "cluster_id":          cid,
            "eclass":              st.get("eclass", ""),
            "n_skus":              st.get("n_skus", 0),
            "n_buyers":            st.get("n_buyers", 0),
            "top_features":        top_feats_str,
            "example_description": example,
        })
    return pd.DataFrame(
        rows,
        columns=["cluster_id", "eclass", "n_skus", "n_buyers", "top_features", "example_description"],
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()

    print("=" * 90)
    print("CORE DEMAND PREDICTION PIPELINE")
    print("L1 very broad eclass recall | L2 broader manufacturer-confidence gating")
    print("Hierarchical cold start with nace_2digits")
    print("No weather | No seasonal heuristics | No SKU signals")
    print("=" * 90)

    data = load_and_aggregate()
    cfg = DEFAULT_CFG.copy()
    log(f"Using config: {cfg}")

    sub_l1 = predict_level_1(data, cfg)
    sub_l1.to_csv(OUTPUT_L1, index=False)
    log(f"Level 1 saved to {OUTPUT_L1} | rows={len(sub_l1):,} | buyers={sub_l1['buyer_id'].nunique()}")

    sub_l2 = predict_level_2(data, cfg, sub_l1)
    sub_l2.to_csv(OUTPUT_L2, index=False)
    log(f"Level 2 saved to {OUTPUT_L2} | rows={len(sub_l2):,} | buyers={sub_l2['buyer_id'].nunique()}")

    sub_l3 = predict_level_3(data, cfg, sub_l1)
    sub_l3.to_csv(OUTPUT_L3, index=False)
    log(f"Level 3 saved to {OUTPUT_L3} | rows={len(sub_l3):,} | clusters={sub_l3['predicted_id'].nunique()}")

    manifest = build_cluster_manifest(data["l3_cluster_stats"])
    manifest.to_csv(OUTPUT_L3_MANIFEST, index=False)
    log(f"Cluster manifest saved to {OUTPUT_L3_MANIFEST} | clusters={len(manifest)}")

    elapsed = time.time() - t0
    print("=" * 90)
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"L1: {OUTPUT_L1}")
    print(f"L2: {OUTPUT_L2}")
    print(f"L3: {OUTPUT_L3}")
    print(f"L3 manifest: {OUTPUT_L3_MANIFEST}")
    print("=" * 90)


if __name__ == "__main__":
    main()