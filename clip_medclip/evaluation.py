import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def xray_probe(train_x, train_y, test_x, test_y, seed):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
    }


def summary_table(df, metric, models, methods):
    st = df[df["metric"] == metric].groupby(["method", "model"])["value"].agg(["mean", "std"])
    st["итог"] = st.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return st["итог"].unstack("model")[models].reindex(methods)


def add_metric_rows(rows, model, method, seed, metrics):
    for metric, value in metrics.items():
        rows.append({
            "model": model, "method": method, "seed": seed,
            "metric": metric, "value": value,
        })


def metric_summary(df, model, method, metric="balanced_accuracy"):
    values = df[
        (df["model"] == model) &
        (df["method"] == method) &
        (df["metric"] == metric)
    ]["value"]
    return f"{values.mean():.3f} ± {values.std():.3f}"
