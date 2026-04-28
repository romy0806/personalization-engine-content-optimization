"""
Data processing utilities for the personalization engine.
"""

import pandas as pd


def load_news(path: str) -> pd.DataFrame:
    """Load news/article metadata."""
    return pd.read_csv(path, sep="\t", header=None)


def load_behaviors(path: str) -> pd.DataFrame:
    """Load user behavior or impression logs."""
    return pd.read_csv(path, sep="\t", header=None)


def clean_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Basic text cleanup for title, abstract, or content fields."""
    df = df.copy()
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df
