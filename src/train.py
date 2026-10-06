"""Step 2: Train and evaluate models with a strict time-based setup.

  TRAIN       sets released 1995-2009 -> growth from release to Dec 2015
  VALIDATION  sets released 2010-2012 -> growth from release to Dec 2015 (used to pick the model)
  FORWARD     sets released 2013-2014 -> growth from release to Apr 2019
  TEST        These sets were still on store shelves in 2015 and had no resale history. The
              model never sees them or their prices, so this is a genuine "predict the future"
              test, like using the model on release day and checking back four years later.

Compares: global median baseline, theme-average baseline, Ridge regression, gradient boosting.
Outputs: models/model.joblib, models/metadata.json, reports/metrics.json, reports/*.png
"""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA = Path("data/processed/lego_model_data.csv")
MODELS, REPORTS = Path("models"), Path("reports")

NUMERIC = ["log_pieces", "log_rrp", "price_per_piece", "minifigs", "minifigs_per_100_pieces",
           "is_licensed", "size_group", "years_held"]
CATEGORICAL = ["theme_grouped"]
FEATURES = NUMERIC + CATEGORICAL


def make_gbm():
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
    ])
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=7, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=[False] * len(NUMERIC) + [True], random_state=0)
    return Pipeline([("pre", pre), ("model", model)])


def make_ridge():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-2, 3, 20)))])


def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    k = max(1, len(y) // 10)
    top = np.argsort(pred)[-k:]  # the 10% of sets the model likes best
    return {
        "MAE_pct_points": round(100 * mean_absolute_error(y, pred), 2),
        "R2": round(r2_score(y, pred), 3),
        "rank_correlation": round(pd.Series(y).corr(pd.Series(pred), method="spearman"), 3) if np.ptp(pred) > 0 else None,
        "top_10pct_picks_actual_growth_pct": round(100 * y[top].mean(), 2),
        "all_sets_actual_growth_pct": round(100 * y.mean(), 2),
    }


def trim(d, col):
    """Drop the most extreme 1% of growth rates on each side (mostly data errors)."""
    lo, hi = d[col].quantile([0.01, 0.99])
    return d[d[col].between(lo, hi)]


def main():
    df = pd.read_csv(DATA)
    # years_held = how long after release the value was measured. It's an input the user chooses
    # ("what will this be worth 5 years from now?"), so it's not leakage. It matters because sets
    # usually sell BELOW retail while still in stores and only climb after they retire.
    train = trim(df[(df["release_year"] <= 2009) & df["cagr_2015"].notna()], "cagr_2015")
    val = trim(df[df["release_year"].between(2010, 2012) & df["cagr_2015"].notna()], "cagr_2015")
    fwd = trim(df[df["release_year"].between(2013, 2014) & df["cagr_2019"].notna()], "cagr_2019")
    train, val = (d.assign(years_held=d["years_to_2015"]) for d in (train, val))
    fwd = fwd.assign(years_held=fwd["years_to_2019"])
    print(f"Train {len(train):,} | Validation {len(val):,} | Forward test {len(fwd):,} sets")

    results = {"n_train": len(train), "n_validation": len(val), "n_forward_test": len(fwd), "validation": {}}
    ytr, yval = train["cagr_2015"], val["cagr_2015"]
    theme_avg = train.groupby("theme_grouped")["cagr_2015"].mean()
    results["validation"]["baseline_global_median"] = scores(yval, np.full(len(val), ytr.median()))
    results["validation"]["baseline_theme_average"] = scores(yval, val["theme_grouped"].map(theme_avg).fillna(ytr.median()))
    candidates = {"ridge": make_ridge, "gradient_boosting": make_gbm}
    for name, make in candidates.items():
        results["validation"][name] = scores(yval, make().fit(train[FEATURES], ytr).predict(val[FEATURES]))

    best = min(candidates, key=lambda n: results["validation"][n]["MAE_pct_points"])
    results["chosen_model"] = best
    print(f"Best on validation: {best}")

    # Refit the chosen model on everything up to 2012, then run the one-time forward test.
    known = pd.concat([train, val])
    model = candidates[best]().fit(known[FEATURES], known["cagr_2015"])
    pred_fwd = model.predict(fwd[FEATURES])
    fwd_theme = fwd["theme_grouped"].map(known.groupby("theme_grouped")["cagr_2015"].mean()).fillna(known["cagr_2015"].median())
    results["forward_test_2013_2014_sets"] = {
        "model": scores(fwd["cagr_2019"], pred_fwd),
        "baseline_global_median": scores(fwd["cagr_2019"], np.full(len(fwd), known["cagr_2015"].median())),
        "baseline_theme_average": scores(fwd["cagr_2019"], fwd_theme),
    }

    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # ---- plots ----
    imp = permutation_importance(model, fwd[FEATURES], fwd["cagr_2019"], n_repeats=20,
                                 random_state=0, scoring="neg_mean_absolute_error")
    order = np.argsort(imp.importances_mean)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(FEATURES)[order], 100 * imp.importances_mean[order], color="#d01012")
    plt.xlabel("Increase in error when the feature is shuffled (pct. points)")
    plt.title("What drives LEGO set appreciation?")
    plt.tight_layout(); plt.savefig(REPORTS / "feature_importance.png", dpi=150); plt.close()

    q = pd.qcut(pred_fwd, 5, labels=["Lowest", "2", "3", "4", "Highest"])
    by_q = fwd.assign(q=q).groupby("q", observed=True)["cagr_2019"].mean() * 100
    plt.figure(figsize=(7, 4.5))
    plt.bar(by_q.index.astype(str), by_q.values, color="#ffcf00", edgecolor="black")
    plt.xlabel("Model's predicted growth (quintile)"); plt.ylabel("Actual annual growth by 2019 (%)")
    plt.title("Forward test: sets released 2013-2014")
    plt.tight_layout(); plt.savefig(REPORTS / "forward_test_quintiles.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(100 * fwd["cagr_2019"], 100 * pred_fwd, s=8, alpha=0.5)
    lims = [min(100 * fwd["cagr_2019"].min(), 100 * pred_fwd.min()), max(100 * fwd["cagr_2019"].max(), 100 * pred_fwd.max())]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Actual annual growth (%)"); plt.ylabel("Predicted annual growth (%)")
    plt.title("Forward test: predicted vs. actual")
    plt.tight_layout(); plt.savefig(REPORTS / "predicted_vs_actual.png", dpi=150); plt.close()

    themes = known.groupby("theme_grouped")["cagr_2015"].agg(["median", "count"]).query("count >= 15").sort_values("median")
    plt.figure(figsize=(8, 7))
    plt.barh(themes.index, 100 * themes["median"], color="#006cb7")
    plt.xlabel("Median annual growth since release (%)"); plt.title("Appreciation by theme (sets released 1995-2012)")
    plt.tight_layout(); plt.savefig(REPORTS / "growth_by_theme.png", dpi=150); plt.close()

    # The app uses exactly the model that passed the forward test.
    joblib.dump(model, MODELS / "model.joblib")
    meta = {"features": FEATURES, "numeric": NUMERIC, "model": best, "years_held_range": [float(known["years_held"].min()), float(known["years_held"].max())],
            "themes": sorted(known["theme_grouped"].unique()),
            "error_pct_points": results["forward_test_2013_2014_sets"]["model"]["MAE_pct_points"]}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved models/model.joblib and plots in reports/")


if __name__ == "__main__":
    main()
