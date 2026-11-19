
import pandas as pd


def calculate_analytics(history_df):

    if history_df.empty:
        return {
            "total": 0,
            "healthy": 0,
            "diseased": 0,
            "average_confidence": 0,
            "most_detected": "N/A"
        }

    total = len(history_df)

    healthy = history_df["Disease"].str.contains(
        "healthy",
        case=False,
        na=False
    ).sum()

    diseased = total - healthy

    average_confidence = history_df["Confidence"].mean()

    most_detected = history_df["Disease"].value_counts().idxmax()

    return {
        "total": total,
        "healthy": int(healthy),
        "diseased": int(diseased),
        "average_confidence": round(average_confidence, 2),
        "most_detected": most_detected
    }
