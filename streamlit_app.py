import streamlit as st
import pandas as pd, numpy as np, re
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor

# =========================
# Parameters
# =========================
PATH  = "data/QSPR_data_app.xlsx"
EXPECTED_POLYMERS = ["PE","PP","PS"]
RANDOM_STATE = 42
Q2_TEST_METHOD = "F1"         # "F1", "F2", "F3"

# --- Feature selection config ---
FEATURE_MODE = "manual"        # "combined" albo "manual"
TOP_K = 1                      # używane tylko w trybie "combined"

# RĘCZNY wybór: globalnie
MANUAL_FEATURES = ["logD", "M","π"]   # <- wpisz swoje kolumny

# RĘCZNY wybór: per-polimer (opcjonalnie nadpisuje globalne)
MANUAL_FEATURES_PER_POLYMER = {
    # "PE": ["logD", "π"],
    # "PP": ["logD", "M", "q−"],
    # "PS": ["logD", "εβ"],
}

# Czy dopuszczamy fallback do combined, gdy ręczna lista jest pusta/nieprawidłowa?
FALLBACK_TO_COMBINED_IF_INVALID = False   # ustaw True, jeśli chcesz miękki fallback

# Pliki pomocnicze
SPLIT_FILE = "data/train_test_compounds.xlsx"
BEST_PARAMS_FILE = "data/model_REPORT_GB_by_polymer_logD+1.xlsx"

# Gradient Boosting + Grid
GB_ESTIMATOR = GradientBoostingRegressor(random_state=RANDOM_STATE)
GB_PARAM_GRID = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.6, 0.8, 1.0],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4],
}

# =========================
# Load & clean
# =========================
df = pd.read_excel(PATH)
compound_col = "Organic compound" if "Organic compound" in df.columns else "Organic compounds"

print(df.head())
COLMAP = {"q-":"q−","q–":"q−","q—":"q−", "V’":"V'", "V´":"V'", "V`":"V'", "Vʼ":"V'"}
df = df.rename(columns={c: COLMAP.get(c, c) for c in df.columns})

def norm_polymer(s):
    if pd.isna(s): return s
    s = str(s).strip().upper()
    replacements = {
        "POLYETHYLENE":"PE", "POLYPROPYLENE":"PP", "POLYSTYRENE":"PS",
        "PE":"PE", "PP":"PP", "PS":"PS"
    }
    return replacements.get(s, s)

if "Polymer" not in df.columns:
    raise ValueError("Column 'Polymer' not found.")
df["Polymer"] = df["Polymer"].apply(norm_polymer)

num_cols = ["LogKd","logD","εα","εβ","π","M","q−","V'"]
num_cols = [c for c in num_cols if c in df.columns]

def clean_number(x):
    if pd.isna(x): return np.nan
    s = str(x).replace("\u00A0","").replace("−","-").replace(",",".").strip()
    if s in {"", "-"}: return np.nan
    try: return float(s)
    except: return np.nan

for c in num_cols:
    df[c] = df[c].apply(clean_number)

# =========================
# Build X & Y — per POLYMER
# =========================
feat_cols = [c for c in ["logD","εα","εβ","π","M","q−","V'"] if c in df.columns]

Y_wide = df.pivot_table(index=compound_col, columns="Polymer", values="LogKd", aggfunc="first")
Y_wide = Y_wide.reindex(columns=[p for p in EXPECTED_POLYMERS if p in Y_wide.columns])

X_by_comp = (df[[compound_col]+feat_cols]
             .drop_duplicates(subset=[compound_col])
             .set_index(compound_col)
             .apply(pd.to_numeric, errors="coerce"))
X_by_comp = X_by_comp[~X_by_comp.isna().any(axis=1)]

common_idx = X_by_comp.index.intersection(Y_wide.index)
X_by_comp = X_by_comp.loc[common_idx]
Y_wide    = Y_wide.loc[common_idx]

print(f"After cleaning: {len(X_by_comp)} compounds; features: {list(X_by_comp.columns)}; polymer targets: {list(Y_wide.columns)}")

# =========================
# Grouped train/test split — użyj zapisanych w Excelu, jeśli są
# =========================
try:
    x_train_prev = pd.read_excel(SPLIT_FILE, sheet_name="X_train", index_col=0)
    x_test_prev  = pd.read_excel(SPLIT_FILE, sheet_name="X_test",  index_col=0)

    train_compounds = x_train_prev.index.astype(str)
    test_compounds  = x_test_prev.index.astype(str)

    train_compounds = [c for c in train_compounds if c in X_by_comp.index]
    test_compounds  = [c for c in test_compounds if c in X_by_comp.index]

    if len(train_compounds) == 0 or len(test_compounds) == 0:
        raise ValueError("Brak wspólnych związków między SPLIT_FILE a aktualnym X_by_comp.")

    print(f"[SPLIT] Używam podziału z pliku '{SPLIT_FILE}': "
          f"{len(train_compounds)} train, {len(test_compounds)} test związków.")

except Exception as e:
    print(f"[SPLIT] Nie udało się wczytać podziału z '{SPLIT_FILE}' ({e}). "
          f"Tworzę nowy GroupShuffleSplit.")
    groups_all = np.array(X_by_comp.index)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_by_comp, Y_wide, groups=groups_all))
    train_compounds = X_by_comp.index[train_idx]
    test_compounds  = X_by_comp.index[test_idx]

X_train_all = X_by_comp.loc[train_compounds].copy()
X_test_all  = X_by_comp.loc[test_compounds].copy()
Y_train_all = Y_wide.loc[train_compounds].copy()
Y_test_all  = Y_wide.loc[test_compounds].copy()

print(f"Train compounds: {len(train_compounds)} | Test compounds: {len(test_compounds)}")

# Hold predictions for plots
pred_train = {}
pred_test  = {}

# =========================
# (OPTIONAL) Feature ranking (combined) — TYLKO gdy FEATURE_MODE != "manual"
# =========================
score_df = pd.DataFrame()
selected_combined = []

if FEATURE_MODE.lower() != "manual":
    def safe_abs_corr(a, b, method="pearson"):
        try:
            s = pd.Series(a).astype(float); t = pd.Series(b).astype(float)
            r = s.corr(t, method=method)
            return float(abs(r)) if np.isfinite(r) else np.nan
        except Exception:
            return np.nan

    pearson_scores, spearman_scores, mi_scores = {}, {}, {}
    for col in X_train_all.columns:
        p_vals, s_vals, mi_vals = [], [], []
        for pol in Y_train_all.columns:
            y = Y_train_all[pol]; mask = y.notna()
            if mask.sum() >= 3:
                xv = X_train_all.loc[mask, col].values
                yv = y.loc[mask].values
                if np.nanstd(xv) == 0 or np.nanstd(yv) == 0: continue
                p_vals.append(safe_abs_corr(xv, yv, "pearson"))
                s_vals.append(safe_abs_corr(xv, yv, "spearman"))
                try:
                    mi = mutual_info_regression(xv.reshape(-1,1), yv, random_state=RANDOM_STATE)
                    mi_vals.append(float(mi[0]))
                except Exception:
                    pass
        pearson_scores[col]  = float(np.nanmean(p_vals)) if p_vals else np.nan
        spearman_scores[col] = float(np.nanmean(s_vals)) if s_vals else np.nan
        mi_scores[col]       = float(np.nanmean(mi_vals)) if mi_vals else np.nan

    score_df = pd.DataFrame({
        "pearson_abs": pd.Series(pearson_scores),
        "spearman_abs": pd.Series(spearman_scores),
        "mi": pd.Series(mi_scores),
    }).fillna(0.0)

    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean())/sd if sd and np.isfinite(sd) and sd>0 else pd.Series(0.0, index=s.index)

    score_df["combined"] = z(score_df["pearson_abs"]) + z(score_df["spearman_abs"]) + z(score_df["mi"])
    score_df = score_df.sort_values("combined", ascending=False)
    ranked_features = list(score_df.index) if np.any(score_df["combined"].values) \
                      else list(X_train_all.var().sort_values(ascending=False).index)

    def select_combined_features(ranked, X, top_k):
        selected = []
        for f in ranked:
            if len(selected) >= top_k: break
            if not selected:
                selected.append(f); continue
            mx = X[[f] + selected].corr().abs()[f].drop(f).max()
            if not np.isfinite(mx) or mx < 0.95:
                selected.append(f)
        if len(selected) < top_k:
            for f in ranked:
                if f not in selected:
                    selected.append(f)
                    if len(selected) == top_k: break
        return selected

    selected_combined = select_combined_features(ranked_features, X_train_all, TOP_K)

    print("\n=== PRE-FS feature ranking (top 10, combined) ===")
    print(score_df.head(10))
    print(f"\nSelected descriptors by combined (TOP_K={TOP_K}): {selected_combined}")
else:
    print("\n[INFO] FEATURE_MODE='manual' — pomijam całkowicie automatyczny ranking/selektor.")

# =========================
# Helpers
# =========================
def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def bias(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))

def mpe(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    mask = y != 0
    if mask.sum() == 0: return np.nan
    return float(100 * np.mean((np.asarray(y_pred, dtype=float)[mask] - y[mask]) / y[mask]))

def mne(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    mask = np.abs(y) != 0
    if mask.sum() == 0: return np.nan
    return float(100 * np.mean(np.abs(np.asarray(y_pred, dtype=float)[mask] - y[mask]) / np.abs(y[mask])))

def external_q2(y_true, y_pred, y_train_mean, method="F1"):
    if method == "F1":
        num = np.sum((y_pred - y_true)**2); den = np.sum((y_true - y_train_mean)**2)
    elif method == "F2":
        num = np.sum((y_pred - y_true)**2); den = np.sum((y_true - np.mean(y_true))**2)
    elif method == "F3":
        num = np.sum((y_pred - y_true)**2); den = np.sum((y_pred - y_train_mean)**2)
    else:
        raise ValueError("Q2_TEST_METHOD must be 'F1', 'F2', or 'F3'")
    return 1 - num/den if den > 0 else np.nan

def tune_with_groups(estimator, param_grid, X_tr, y_tr, groups, scoring="neg_root_mean_squared_error"):
    if isinstance(y_tr, pd.DataFrame):
        if y_tr.shape[1] == 1: y_tr = y_tr.iloc[:, 0]
        else: raise ValueError("y_tr must be 1D.")
    n_splits = min(5, max(2, len(np.unique(groups))))
    cv = GroupKFold(n_splits=n_splits)
    gs = GridSearchCV(
        estimator=estimator, param_grid=param_grid,
        scoring=scoring, cv=cv, n_jobs=-1, verbose=0, refit=True, error_score="raise"
    )
    gs.fit(X_tr, y_tr, groups=groups)
    print(f"[GB] Best params: {gs.best_params_} | best CV score ({scoring}): {gs.best_score_:.5f}")
    return gs.best_estimator_, gs.best_params_

# -------------------------
# Manual feature utility
# -------------------------
def sanitize_manual_features(candidates, available_cols):
    if not candidates: return [], []
    cand = [c for c in map(str, candidates)]
    ok = [c for c in cand if c in available_cols]
    missing = [c for c in cand if c not in available_cols]
    return ok, missing

def get_selected_features_for_polymer(pol, X_train_cols):
    if FEATURE_MODE.lower() == "manual":
        cand = MANUAL_FEATURES_PER_POLYMER.get(pol, MANUAL_FEATURES)
        ok, missing = sanitize_manual_features(cand, X_train_cols)
        if missing:
            print(f"[WARN] [{pol}] Missing manual features (ignored): {missing}")
        if len(ok) == 0:
            if FALLBACK_TO_COMBINED_IF_INVALID and selected_combined:
                print(f"[WARN] [{pol}] Manual list empty/invalid — fallback to 'combined': {selected_combined}")
                return selected_combined
            else:
                raise ValueError(f"[{pol}] Manual feature list empty/invalid and fallback disabled.")
        return ok
    # combined mode
    return selected_combined

# =========================
# Wczytywanie zapisanych hiperparametrów GB z Excela (per polimer)
# =========================
best_params_external = {}

try:
    bp_df = pd.read_excel(BEST_PARAMS_FILE, sheet_name="Best params per polymer")

    for _, row in bp_df.iterrows():
        pol_name = str(row["Polymer"])
        params = {}

        if "n_estimators" in row:
            params["n_estimators"] = int(row["n_estimators"])
        if "learning_rate" in row:
            params["learning_rate"] = float(row["learning_rate"])
        if "max_depth" in row:
            params["max_depth"] = int(row["max_depth"])
        if "subsample" in row:
            params["subsample"] = float(row["subsample"])
        if "max_features" in row:
            mf = row["max_features"]
            params["max_features"] = None if pd.isna(mf) else mf
        if "min_samples_split" in row:
            params["min_samples_split"] = int(row["min_samples_split"])
        if "min_samples_leaf" in row:
            params["min_samples_leaf"] = int(row["min_samples_leaf"])

        best_params_external[pol_name] = params

    print(f"[HP] Wczytano zapisane hiperparametry dla polimerów z '{BEST_PARAMS_FILE}': "
          f"{list(best_params_external.keys())}")

except FileNotFoundError:
    print(f"[HP] Plik '{BEST_PARAMS_FILE}' nie istnieje — będę stroić GB GridSearchem.")
except Exception as e:
    print(f"[HP] Problem z wczytaniem hiperparametrów z '{BEST_PARAMS_FILE}' ({e}) — "
          f"będę stroić GB GridSearchem.")

# =========================
# Train/tune/evaluate per POLYMER (ONLY Gradient Boosting)
# =========================
stats_rows = []
global_stats = []
pred_test_tables = []
best_params_per_polymer = {}
gb_importances_per_polymer = {}
pred_tables = []

print(f"Feature mode: {FEATURE_MODE} "
      f"{'(per-polymer overrides active)' if (FEATURE_MODE=='manual' and MANUAL_FEATURES_PER_POLYMER) else ''}")

for pol in Y_wide.columns:
    y_tr_full = Y_train_all[pol]; tr_mask = y_tr_full.notna()
    n_tr = int(tr_mask.sum())
    if n_tr < 4:
        print(f"\n[{pol}] Not enough training samples (n={n_tr}) — skipping.")
        continue

    selected_features = get_selected_features_for_polymer(pol, X_train_all.columns)

    X_tr_all = X_train_all.loc[tr_mask, selected_features]
    y_tr     = y_tr_full.loc[tr_mask]

    y_te_full = Y_test_all[pol]; te_mask = y_te_full.notna()
    X_te_all = X_test_all.loc[te_mask, selected_features]
    y_te     = y_te_full.loc[te_mask]
    n_te = int(te_mask.sum())

    groups_train = np.array(X_tr_all.index)

    print(f"\n=== [{pol}] GradientBoosting | features: {selected_features} ===")

    # Jeśli mamy zapisane hiperparametry dla tego polimeru — użyj ich
    if pol in best_params_external:
        params = best_params_external[pol]
        est = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        est.fit(X_tr_all, y_tr)
        best_params = params
        print(f"[GB] Używam zapisanych hiperparametrów dla {pol}: {params}")
    else:
        # fallback: klasyczny GridSearchCV
        est, best_params = tune_with_groups(
            GB_ESTIMATOR, GB_PARAM_GRID, X_tr_all, y_tr, groups_train
        )

    best_params_per_polymer[pol] = best_params

    # OOF Q² (GroupKFold)
    n_splits = min(5, max(2, len(np.unique(groups_train))))
    cv = GroupKFold(n_splits=n_splits)
    oof = np.full(shape=len(X_tr_all), fill_value=np.nan, dtype=float)
    for tr_idx, va_idx in cv.split(X_tr_all, y_tr, groups_train):
        est_cv = clone(est)
        est_cv.fit(X_tr_all.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof[va_idx] = est_cv.predict(X_tr_all.iloc[va_idx])

    sse = float(np.sum((oof - y_tr.values)**2))
    sst = float(np.sum((y_tr.values - y_tr.values.mean())**2))
    Q2_train = 1 - sse/sst if sst > 0 else np.nan

    # Train metrics
    y_tr_pred = est.predict(X_tr_all)
    stats_rows.append({
        "Polymer": pol, "Model": "GradientBoosting", "Set": "Training set (CV)",
        "n": n_tr,
        "R2": r2_score(y_tr, y_tr_pred),
        "R2_OOF": r2_score(y_tr, oof),
        "Q2": Q2_train,
        "RMSE": rmse(y_tr, y_tr_pred),
        "MAE": mean_absolute_error(y_tr, y_tr_pred),
        "BIAS": bias(y_tr, y_tr_pred),
        "MPE": mpe(y_tr, y_tr_pred),
        "MNE": mne(y_tr, y_tr_pred),
    })

    pred_train[pol] = pd.DataFrame({
        f"{pol}_true": y_tr,
        f"{pol}_pred": y_tr_pred
    }, index=X_tr_all.index)

    # Test metrics
    if n_te >= 1:
        y_te_pred = est.predict(X_te_all)
        q2_ext = external_q2(y_true=y_te.values,
                             y_pred=y_te_pred,
                             y_train_mean=float(y_tr.values.mean()),
                             method=Q2_TEST_METHOD)
        stats_rows.append({
            "Polymer": pol, "Model": "GradientBoosting", "Set": "Test set",
            "n": n_te,
            "R2": r2_score(y_te, y_te_pred),
            "R2_OOF": np.nan,
            "Q2": q2_ext,
            "RMSE": rmse(y_te, y_te_pred),
            "MAE": mean_absolute_error(y_te, y_te_pred),
            "BIAS": bias(y_te, y_te_pred),
            "MPE": mpe(y_te, y_te_pred),
            "MNE": mne(y_te, y_te_pred),
        })

        pred_test[pol] = pd.DataFrame({
            f"{pol}_true": y_te,
            f"{pol}_pred": y_te_pred
        }, index=X_te_all.index)

        pred_df = pd.DataFrame({
            f"{pol}_true": y_te,
            f"{pol}_pred_GB": y_te_pred
        }, index=X_te_all.index)
        pred_test_tables.append(pred_df)

        pred_tables.append(pd.DataFrame({
            f"{pol}_true": y_te,
            f"{pol}_pred": y_te_pred
        }, index=X_te_all.index))

    # Globalny model per polimer — importances
    y_full = Y_wide[pol].dropna()
    if len(y_full) >= 4:
        X_full = X_by_comp.loc[y_full.index, selected_features]
        est_global = clone(est)
        est_global.fit(X_full, y_full)
        y_full_pred = est_global.predict(X_full)
        sse_g = np.sum((y_full_pred - y_full.values) ** 2)
        sst_g = np.sum((y_full.values - np.mean(y_full.values)) ** 2)
        Q2_model = 1 - sse_g / sst_g if sst_g > 0 else np.nan
        global_stats.append({
            "Polymer": pol, "Model": "GradientBoosting", "Set": "Model (all data)",
            "n": len(y_full),
            "R2": r2_score(y_full, y_full_pred),
            "R2_OOF": np.nan,
            "Q2": Q2_model,
            "RMSE": rmse(y_full, y_full_pred),
            "MAE": mean_absolute_error(y_full, y_full_pred),
            "BIAS": bias(y_full, y_full_pred),
            "MPE": mpe(y_full, y_full_pred),
            "MNE": mne(y_full, y_full_pred),
        })
        if hasattr(est_global, "feature_importances_"):
            gb_importances_per_polymer[pol] = pd.Series(
                est_global.feature_importances_, index=selected_features
            ).sort_values(ascending=False)

# =========================
# OUTPUT: tabelki główne
# =========================
stats_df = pd.DataFrame(stats_rows + global_stats)

if not stats_df.empty:
    for c in ["R2","R2_OOF","Q2","RMSE","MAE","BIAS","MPE","MNE"]:
        if c in stats_df.columns:
            stats_df[c] = stats_df[c].astype(float).round(6)

    set_order = ["Model (all data)", "Training set (CV)", "Test set"]
    stats_df["Set"] = pd.Categorical(stats_df["Set"], categories=set_order, ordered=True)
    stats_df = stats_df.sort_values(["Polymer","Model","Set"]).reset_index(drop=True)

    print("\n=== MODEL (ONLY GradientBoosting) — by POLYMER ===")
    print(stats_df)

    stats_df.to_excel("benchmark_GB_by_polymer.xlsx", index=False)
    print("Saved: benchmark_GB_by_polymer.xlsx")
else:
    print("No results (not enough data after filtering).")

# =========================
# ONE-LINE AVERAGES (macro across polymers)
# =========================
AVG_SETS = ["Training set (CV)", "Test set"]
METRICS  = ["R2","R2_OOF","Q2","RMSE","MAE","BIAS","MPE","MNE"]

def add_macro_avg_row(stats_df, set_name, model_name="GradientBoosting"):
    df_set = stats_df[(stats_df["Set"] == set_name) & (stats_df["Polymer"].isin(Y_wide.columns))]
    if df_set.empty:
        return stats_df
    macro = df_set[METRICS].astype(float).mean(numeric_only=True)

    row = {"Polymer": "ALL",
           "Model": model_name,
           "Set": f"{set_name} (avg)"}
    for m in METRICS:
        v = float(macro.get(m, np.nan)) if m in macro else np.nan
        row[m] = np.round(v, 3) if np.isfinite(v) else np.nan
    row["n"] = int(df_set["n"].sum()) if "n" in df_set.columns else np.nan

    return pd.concat([stats_df, pd.DataFrame([row])], ignore_index=True)

for s in AVG_SETS:
    stats_df = add_macro_avg_row(stats_df, s)

set_order = ["Model (all data)", "Training set (CV)", "Training set (avg)", "Test set", "Test set (avg)"]
stats_df["Set"] = pd.Categorical(stats_df["Set"], categories=set_order, ordered=True)
stats_df = stats_df.sort_values(["Polymer","Model","Set"]).reset_index(drop=True)

# =========================
# SUMMARY & REPORT
# =========================
def summarize_set(df_set):
    if df_set.empty:
        return {}
    metrics = ["R2","R2_OOF","Q2","RMSE","MAE","BIAS","MPE","MNE"]
    out = {}
    macro = df_set[metrics].mean(numeric_only=True)
    for m in metrics:
        out[f"macro_{m}"] = float(macro.get(m, np.nan))
    n = df_set["n"].astype(float).values
    w = n / n.sum() if n.sum() > 0 else np.ones_like(n)/len(n)
    def wavg(vals):
        vals = np.asarray(vals, dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() == 0: return np.nan
        return float(np.sum(vals[mask] * w[mask]) / np.sum(w[mask]))
    for m in metrics:
        out[f"micro_{m}"] = wavg(df_set[m].values)
    out["total_n"] = int(df_set["n"].sum())
    out["outputs"] = int(df_set["Polymer"].nunique())
    out["models"]  = int(df_set["Model"].nunique())
    return out

summary_rows = []
if not stats_df.empty:
    for set_name in ["Training set (CV)", "Test set"]:
        df_set = stats_df[stats_df["Set"] == set_name]
        if not df_set.empty:
            s = summarize_set(df_set)
            s["Set"] = set_name
            summary_rows.append(s)
summary_df = pd.DataFrame(summary_rows)

# Best params per polymer
rows = []
for pol, params in best_params_per_polymer.items():
    rows.append({"Polymer": pol, **params})
best_params_df = pd.DataFrame(rows) if rows else pd.DataFrame()

# Raport o wyborze cech
manual_global_df = pd.DataFrame({"rank": range(1, len(MANUAL_FEATURES)+1), "feature": MANUAL_FEATURES}) if MANUAL_FEATURES else pd.DataFrame()
manual_per_polymer_df = None
if MANUAL_FEATURES_PER_POLYMER:
    rows = []
    for pol, feats in MANUAL_FEATURES_PER_POLYMER.items():
        for r, f in enumerate(feats, 1):
            rows.append({"Polymer": pol, "rank": r, "feature": f})
    manual_per_polymer_df = pd.DataFrame(rows)

prefs_df = score_df.reset_index().rename(columns={"index":"feature"}) if not score_df.empty else pd.DataFrame()
selected_combined_df = pd.DataFrame({"rank": range(1, len(selected_combined)+1), "feature": selected_combined}) if selected_combined else pd.DataFrame()

# GB importances per polymer
gb_importances_table = None
if gb_importances_per_polymer:
    frames = []
    for pol, s in gb_importances_per_polymer.items():
        frames.append(pd.DataFrame({"Polymer": pol, "feature": s.index, "importance": s.values}))
    gb_importances_table = pd.concat(frames, axis=0)

# Predykcje testowe
predictions_test_df = None
if pred_test_tables:
    predictions_test_df = pd.concat(pred_test_tables, axis=1)


# =====================================================================
#                       STREAMLIT UI (NA WIERZCHU MODELU)
# =====================================================================

# Konfiguracja strony
st.set_page_config(
    page_title="LogKd prediction for microplastics",
    layout="wide"
)

st.title("LogKd prediction for microplastics")
st.write(
    """
This app wraps the **existing GradientBoosting model** (per polymer, by descriptors)
without changing its logic or training procedure.

You can:
- predict **LogKd** for a single compound (manual input),
- or upload a **file with descriptors** and get batch predictions for selected polymers.
"""
)

# Polimery dostępne faktycznie w danych
AVAILABLE_POLYMERS = [p for p in EXPECTED_POLYMERS if p in Y_wide.columns]
if not AVAILABLE_POLYMERS:
    st.error("No polymers found in Y_wide. Check that the model script loaded correctly.")
    st.stop()

# Cechy wejściowe (dokładnie te z X_by_comp z Twojego skryptu)
FEATURE_COLUMNS = list(X_by_comp.columns)


@st.cache_resource
def get_trained_estimator_for_polymer(polymer: str):
    """
    Trenuje finalny model GradientBoosting dla danego polimeru
    na CAŁYM zbiorze (tak jak w sekcji 'Globalny model per polimer').

    Wykorzystuje:
    - get_selected_features_for_polymer (ta sama logika wyboru cech),
    - best_params_per_polymer (hiperparametry z Twojego skryptu),
    - RANDOM_STATE ten sam co w oryginalnym kodzie.
    """
    if polymer not in Y_wide.columns:
        raise ValueError(f"Polymer {polymer} not found in Y_wide.")

    selected_features = get_selected_features_for_polymer(polymer, X_by_comp.columns)

    y_full = Y_wide[polymer].dropna()
    if len(y_full) < 4:
        raise ValueError(f"Not enough data to train model for polymer {polymer} (n={len(y_full)}).")

    X_full = X_by_comp.loc[y_full.index, selected_features]

    params = best_params_per_polymer.get(polymer, {})
    est = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    est.fit(X_full, y_full)

    return est, selected_features


def predict_single_compound(descriptor_dict: dict, polymers: list[str], compound_name: str = "compound") -> pd.DataFrame:
    X_new = pd.DataFrame([descriptor_dict], index=[compound_name])
    rows = []

    for pol in polymers:
        try:
            est, selected_feats = get_trained_estimator_for_polymer(pol)
            X_pol = X_new[selected_feats]
            y_pred = est.predict(X_pol)[0]
            rows.append({
                "Compound": compound_name,
                "Polymer": pol,
                "LogKd_pred": y_pred
            })
        except Exception as e:
            rows.append({
                "Compound": compound_name,
                "Polymer": pol,
                "LogKd_pred": np.nan,
                "Error": str(e)
            })

    return pd.DataFrame(rows)


def predict_batch(df_input: pd.DataFrame, polymers: list[str]) -> pd.DataFrame:
    df = df_input.copy()

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    for pol in polymers:
        est, selected_feats = get_trained_estimator_for_polymer(pol)
        X_pol = df[selected_feats]
        preds = est.predict(X_pol)
        df[f"LogKd_{pol}"] = preds

    return df


# Sidebar – wybór polimerów
st.sidebar.header("Model settings")

selected_polymers = st.sidebar.multiselect(
    "Select polymers for prediction",
    options=AVAILABLE_POLYMERS,
    default=AVAILABLE_POLYMERS
)

if not selected_polymers:
    st.warning("Select at least one polymer to run predictions.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write("Descriptors used by the model:")
for c in FEATURE_COLUMNS:
    st.sidebar.write(f"- {c}")

# Zakładki: pojedyncza predykcja vs batch
tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ---------------- Single prediction ----------------
with tab_single:
    st.subheader("Single compound prediction")

    compound_name = st.text_input("Compound ID / name", value="your_compound")

    st.write("Provide descriptor values. These must match the scale used in the training data.")

    descriptor_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            descriptor_values[feat] = st.number_input(
                label=f"{feat}",
                value=0.0,
                format="%.6f",
                key=f"single_{feat}"
            )

    if st.button("Predict LogKd for this compound"):
        with st.spinner("Running predictions..."):
            df_res = predict_single_compound(descriptor_values, selected_polymers, compound_name)
        st.write("Predictions:")
        st.dataframe(df_res, use_container_width=True)

# ---------------- Batch prediction ----------------
with tab_batch:
    st.subheader("Batch prediction from file")

    st.write(
        """
Upload a **CSV or Excel** file that contains at least the following columns  
(these are the descriptors used in the original model):
"""
    )
    st.code(", ".join(FEATURE_COLUMNS), language="text")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df_input.head(), use_container_width=True)

        missing_cols = [c for c in FEATURE_COLUMNS if c not in df_input.columns]
        if missing_cols:
            st.error(f"Missing required feature columns: {missing_cols}")
        else:
            if st.button("Run batch prediction"):
                with st.spinner("Running batch predictions..."):
                    try:
                        df_pred = predict_batch(df_input, selected_polymers)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                    else:
                        st.write("Predictions (first rows):")
                        st.dataframe(df_pred.head(), use_container_width=True)

                        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download full predictions as CSV",
                            data=csv_bytes,
                            file_name="logKd_predictions.csv",
                            mime="text/csv"
                        )

st.markdown("---")
st.caption(
    "This is a Streamlit wrapper around the original GradientBoosting-by-polymer model. "
    "The training code and evaluation logic above are kept intact."
)
