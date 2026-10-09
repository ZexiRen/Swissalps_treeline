import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


# =============================================================
# Settings
# =============================================================

file = r"xx"

USE_GROUP_CV = False
GROUP_COL = "region_id"
N_SPLITS = 5

target = "delta_elevation_m"

features = [
    "delta_Tmin",
    "delta_GST",
    "delta_Prcp",
    "clay_current_0cm",
    "aspect_current_nasadem30m_deg",
    "ghm_current",
    "soil_n_current_0cm",
    "slope_current_nasadem30m_deg",
    "available_p_current_0_20cm",
    "soil_oc_current_0cm",
]

labels = {
    "delta_Tmin": "ΔTmin",
    "delta_GST": "ΔGST",
    "delta_Prcp": "ΔPrcp",
    "clay_current_0cm": "Clay",
    "aspect_current_nasadem30m_deg": "Aspect",
    "ghm_current": "Human modification",
    "soil_n_current_0cm": "Soil nitrogen",
    "slope_current_nasadem30m_deg": "Slope",
    "available_p_current_0_20cm": "Available phosphorus",
    "soil_oc_current_0cm": "Soil organic carbon",
}

groups = {
    "delta_Tmin": "Climate",
    "delta_GST": "Climate",
    "delta_Prcp": "Climate",
    "clay_current_0cm": "Soil",
    "aspect_current_nasadem30m_deg": "Topography",
    "ghm_current": "Anthropogenic",
    "soil_n_current_0cm": "Soil",
    "slope_current_nasadem30m_deg": "Topography",
    "available_p_current_0_20cm": "Soil",
    "soil_oc_current_0cm": "Soil",
}

colors = {
    "Anthropogenic": "#C7D3C0",
    "Climate": "#86B3E8",
    "Soil": "#C9B8A6",
    "Topography": "#A8A39B",
}

rf_params = dict(
    n_estimators=500,
    max_features=4,
    min_samples_leaf=5,
    oob_score=True,
    random_state=123,
    n_jobs=-1,
)


# =============================================================
# Read CSV
# =============================================================

raw = pd.read_csv(file)

keep_cols = [target] + features

if USE_GROUP_CV:
    if GROUP_COL not in raw.columns:
        raise ValueError(
            f"USE_GROUP_CV=True, but the CSV is missing '{GROUP_COL}'"
        )
    keep_cols.append(GROUP_COL)

data = raw[keep_cols].dropna().reset_index(drop=True)

print("Full sample:", len(data))

X = data[features]
y = data[target]


# =============================================================
# 5-fold cross-validation
# =============================================================

if USE_GROUP_CV:
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = cv.split(X, y, groups=data[GROUP_COL])
    cv_name = f"Spatial {N_SPLITS}-fold (by {GROUP_COL})"
else:
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=123
    )
    splits = cv.split(X)
    cv_name = f"Random {N_SPLITS}-fold"

print(f"CV scheme: {cv_name}\n")

importance_values = []
oob_r2_list = []
test_r2_list = []

for fold, (train_idx, test_idx) in enumerate(splits, start=1):

    model = RandomForestRegressor(**rf_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    # Out-of-bag R²
    oob_r2 = model.oob_score_

    # Test R²
    test_r2 = r2_score(
        y.iloc[test_idx],
        model.predict(X.iloc[test_idx])
    )

    oob_r2_list.append(oob_r2)
    test_r2_list.append(test_r2)

    print(
        f"Fold {fold}: "
        f"n_train = {len(train_idx):,} | "
        f"n_test = {len(test_idx):,} | "
        f"OOB R² = {oob_r2:.3f} | "
        f"Test R² = {test_r2:.3f}"
    )

    result = permutation_importance(
        model,
        X.iloc[test_idx],
        y.iloc[test_idx],
        scoring="r2",
        n_repeats=20,
        random_state=123 + fold,
        n_jobs=-1,
    )

    importance_values.append(result.importances_mean)


cv_oob_mean = np.mean(oob_r2_list)
cv_oob_sd = np.std(oob_r2_list, ddof=1)

cv_test_mean = np.mean(test_r2_list)
cv_test_sd = np.std(test_r2_list, ddof=1)

print(
    f"\n{N_SPLITS}-fold OOB R² = "
    f"{cv_oob_mean:.3f} ± {cv_oob_sd:.3f}"
)

print(
    f"{N_SPLITS}-fold Test R² = "
    f"{cv_test_mean:.3f} ± {cv_test_sd:.3f}"
)


# =============================================================
# Full-sample random forest
# =============================================================

model_full = RandomForestRegressor(**rf_params)
model_full.fit(X, y)

oob_r2_full = model_full.oob_score_

print(f"Full-sample OOB R² = {oob_r2_full:.3f}\n")


# =============================================================
# Variable importance summary
# =============================================================

importance_folds = pd.DataFrame(
    importance_values,
    columns=features
)

importance = pd.DataFrame({
    "feature": features,
    "importance": importance_folds.mean(axis=0).values,
    "sd": importance_folds.std(axis=0, ddof=1).values,
}).sort_values("importance", ascending=True)

print(
    importance.sort_values("importance", ascending=False)
    .assign(label=lambda d: d["feature"].map(labels))
    [["label", "importance", "sd"]]
    .to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}"
    )
)


# =============================================================
# Visualization
# =============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(
    importance["feature"].map(labels),
    importance["importance"],
    xerr=importance["sd"],
    error_kw=dict(
        ecolor="#444444",
        elinewidth=0.8,
        capsize=2.5
    ),
    color=importance["feature"].map(groups).map(colors),
    height=0.68,
)

ax.set_xlabel("Permutation importance")
ax.set_ylabel("")
ax.set_title(
    "Variable importance (full sample)",
    fontweight="bold"
)

ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")


# ---- R² annotation (optional) ----
# ax.text(
#     0.98, 0.12,
#     f"OOB $R^2$ = {oob_r2_full:.2f}\n"
#     f"CV $R^2$ = {cv_test_mean:.2f} ± {cv_test_sd:.2f}",
#     transform=ax.transAxes,
#     ha="right",
#     va="bottom",
#     fontsize=9,
#     fontweight="bold",
# )


# =============================================================
# Legend
# =============================================================

category_names = [
    "Anthropogenic",
    "Climate",
    "Soil",
    "Topography"
]

handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=colors[name]
    )
    for name in category_names
]

ax.legend(
    handles,
    category_names,
    loc="lower right",
    frameon=False,
    fontsize=8,
    ncol=4,
    columnspacing=0.8,
    handlelength=1.2,
)

plt.tight_layout()
plt.show()