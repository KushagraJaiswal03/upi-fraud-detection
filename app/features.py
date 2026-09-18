"""
Feature engineering for the UPI Fraud Detection project.

Turns the raw Kaggle CSV into a clean, model-ready dataframe.
This exact function is used both at training time and inside the
Streamlit app at prediction time, so behavior always stays in sync.
"""

import ast
import numpy as np
import pandas as pd

# Columns that are IDs, free text, or constant across the whole dataset —
# they carry no predictive signal, so we drop them.
DROP_COLS = [
    "transaction_id", "user_id", "merchant_id", "device_id", "ip_address",
    "description", "timestamp", "request_description",
    "relationship_to_requester", "social_media_presence",
    "upi_handle_age", "handle_contains_official_terms",
]

# Columns stored as stringified Python lists, e.g. "[]" or "['camera']"
LIST_COLS = [
    "recent_app_installs", "permissions_granted",
    "recognized_screen_sharing_apps", "request_description_keywords",
]

CATEGORICAL_COLS = [
    "merchant_category_code", "session_source", "pin_entry_method",
    "authorization_method", "transaction_type", "handle_typo_analysis",
    "handle_registration_pattern", "handle_verification_status",
]


def _parse_location(loc):
    """'(lat, lon)' string -> (lat, lon) floats. Bad/missing -> (0.0, 0.0)."""
    try:
        lat, lon = ast.literal_eval(loc)
        return float(lat), float(lon)
    except Exception:
        return 0.0, 0.0


def _list_len(val):
    """Stringified list -> item count. '[]' -> 0, \"['a','b']\" -> 2."""
    try:
        parsed = ast.literal_eval(val)
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe (with or without is_fraud) -> cleaned feature dataframe."""
    df = df.copy()

    # --- location -> latitude / longitude ---
    if "location" in df.columns:
        lat_lon = df["location"].apply(_parse_location)
        df["latitude"] = lat_lon.apply(lambda t: t[0])
        df["longitude"] = lat_lon.apply(lambda t: t[1])
        df = df.drop(columns=["location"])

    # --- list-string columns -> item counts ---
    for col in LIST_COLS:
        if col in df.columns:
            df[col + "_count"] = df[col].apply(_list_len)
            df = df.drop(columns=[col])

    # --- high-cardinality text -> presence flags ---
    if "url_referrer" in df.columns:
        df["has_referrer"] = df["url_referrer"].notna().astype(int)
        df = df.drop(columns=["url_referrer"])

    if "business_name_match" in df.columns:
        df["has_business_name_match"] = (
            df["business_name_match"].fillna("none").str.lower().ne("none").astype(int)
        )
        df = df.drop(columns=["business_name_match"])

    # --- drop pure ID / constant / free-text columns ---
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- fill any remaining numeric NaNs (e.g. geographic_disparity) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    return df


NOISE_COLS = [
    "unusual_device_flag", "unusual_ip_flag", "unusual_location_flag",
    "handle_verification_status", "handle_typo_analysis", "session_source",
    "pin_entry_method", "authorization_method", "handle_registration_pattern",
]


def inject_realistic_noise(df: pd.DataFrame, noise_rate: float = 0.08, seed: int = 42) -> pd.DataFrame:
    """
    This synthetic dataset makes several risk flags perfectly deterministic
    (flag=1 implies fraud with 100% certainty). Real fraud-detection signals
    are never that clean — device/IP/handle checks have false positives and
    false negatives. We randomly flip a small percentage of these flags,
    independent of the true label, so the model has to learn from imperfect,
    realistic signals rather than memorize a lookup table.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    for col in NOISE_COLS:
        if col not in df.columns:
            continue
        values = df[col].unique()
        if len(values) != 2:
            continue
        a, b = values
        flip_mask = rng.rand(len(df)) < noise_rate
        df.loc[flip_mask, col] = df.loc[flip_mask, col].apply(lambda v: b if v == a else a)
    return df


def get_feature_lists(df: pd.DataFrame):
    """Split a built-feature dataframe into (numeric_cols, categorical_cols),
    excluding the label if present."""
    cols = [c for c in df.columns if c != "is_fraud"]
    categorical = [c for c in CATEGORICAL_COLS if c in cols]
    numeric = [c for c in cols if c not in categorical]
    return numeric, categorical
