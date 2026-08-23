from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def mind_news() -> pd.DataFrame:
    rows = []
    for index in range(1, 13):
        rows.append(
            {
                "news_id": f"N{index}",
                "category": ["sports", "finance", "health"][index % 3],
                "subcategory": f"sub{index % 5}",
                "title": f"Title {index}",
                "abstract": "Abstract",
                "url": "https://example.com",
                "title_entities": "[]",
                "abstract_entities": "[]",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def mind_behaviors() -> pd.DataFrame:
    rows = []
    for index in range(90):
        group = index % 3
        if group == 0:
            history = "N1 N4 N7 N10"
            impressions = "N1-1 N2-0 N3-0 N4-1"
            session_count = 3
        elif group == 1:
            history = "N2 N3"
            impressions = "N5-1 N6-0 N8-0 N9-0 N10-0"
            session_count = 2
        else:
            history = ""
            impressions = "N8-0 N9-0 N11-0 N12-0"
            session_count = 1
        for session in range(session_count):
            rows.append(
                {
                    "impression_id": f"I{index}-{session}",
                    "user_id": f"U{index:03d}",
                    "timestamp": pd.Timestamp("2019-11-01")
                    + pd.Timedelta(days=group * 8 + session),
                    "history": history,
                    "impressions": impressions,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def mind_validation_behaviors(mind_behaviors: pd.DataFrame) -> pd.DataFrame:
    validation = mind_behaviors.copy()
    validation["user_id"] = "V" + validation["user_id"].str[1:]
    validation["impression_id"] = "V" + validation["impression_id"]
    validation["timestamp"] = validation["timestamp"] + pd.Timedelta(days=30)
    return validation
