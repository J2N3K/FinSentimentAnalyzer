#!/usr/bin/env python
"""
Wizualizacja bieżących nastrojów rynkowych:
1) linia sentymentu w czasie,
2) liczebność nagłówków w ostatniej godzinie,
3) (opcja) heat-mapa natężenia artykułów.
"""

# ---------- KONFIG ----------
MONGO_URI  = "mongodb://root:admin@localhost:27017/?authSource=admin"
DB_NAME    = "financial_sentiment_db"
COLL_NAME  = "aggregated_sentiment"

TIME_WINDOW_H = 6          # ile godzin wstecz pobierać do linii trendu
BAR_WINDOW_MIN = 60        # przedział do wykresu słupkowego
# ----------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone

def load_data():
    """Pobiera dane z Mongo i zwraca DataFrame’a."""
    client = MongoClient(MONGO_URI)
    coll   = client[DB_NAME][COLL_NAME]

    # zakres czasowy
    time_from = datetime.now(timezone.utc) - timedelta(hours=TIME_WINDOW_H)
    cursor = coll.find({"window_end": {"$gte": time_from}})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise SystemExit("Brak danych w zadanym zakresie czasu.")
    # casty
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"]   = pd.to_datetime(df["window_end"])
    client.close()
    return df

def lineplot(df):
    """Średni sentyment w czasie dla każdej etykiety."""
    pivot = (
        df.pivot_table(index="window_end",
                       columns="sentiment_label",
                       values="average_score",
                       aggfunc="mean")
          .sort_index()
          .interpolate("time")
    )

    ax = pivot.plot(figsize=(11, 4), title="Średni sentyment ({} h)".format(TIME_WINDOW_H))
    ax.set_xlabel("Czas (UTC)")
    ax.set_ylabel("avg(score)")
    ax.legend(title="Etykieta")
    plt.tight_layout()
    plt.show()

def barplot(df):
    """Liczba nagłówków w ostatnim BAR_WINDOW_MIN minutach."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=BAR_WINDOW_MIN)
    subset = df[df["window_end"] >= cutoff]

    bar = (subset.groupby("sentiment_label")["article_count"]
                  .sum()
                  .reindex(["positive", "neutral", "negative"])  # uporządkuj
                  .fillna(0))

    ax = bar.plot(kind="bar", fontsize=12, rot=0,
                  title=f"Nagłówki w ostatnich {BAR_WINDOW_MIN} min",
                  figsize=(6, 4))
    ax.set_ylabel("Liczba artykułów")
    plt.tight_layout()
    plt.show()

def heatmap(df):
    """(Opcjonalnie) heat-mapa liczebności artykułów w pięciominutowych oknach."""
    df["ts"] = df["window_end"].dt.floor("5min")
    mat = (df.pivot_table(index="ts",
                          columns="sentiment_label",
                          values="article_count",
                          aggfunc="sum")
             .fillna(0))

    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mat.T, aspect="auto", origin="lower")

    ax.set_xticks(range(len(mat.index)))
    ax.set_xticklabels(mat.index.strftime("%H:%M"), rotation=90)
    ax.set_yticks(range(len(mat.columns)))
    ax.set_yticklabels(mat.columns)

    fig.colorbar(im, ax=ax, label="Liczba artykułów")
    ax.set_title("Heat-mapa artykułów (5-min okna)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_data()
    lineplot(data)
    barplot(data)
    # heatmap(data)       # odkomentuj, gdy chcesz trzeci wykres
