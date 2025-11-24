import io

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# =========================
# Global parameters (wspólne)
# =========================

EXPECTED_POLYMERS = ["PE", "PP", "PS"]
RANDOM_STATE = 42
Q2_TEST_METHOD = "F1"  # "F1", "F2", "F3"

# --- Feature selection config (wspólne dla obu modeli) ---
FEATURE_MODE = "manual"  # "combined" albo "manual"
TOP_K = 1

MANUAL_FEATURES = ["logD", "M", "π"]

MANUAL_FEATURES_PER_POLYMER = {
    # "PE": ["logD", "π"],
    # "PP": ["logD", "M", "q−"],
    # "PS": ["logD", "εβ"],
}

FALLBACK_TO_COMBINED_IF_INVALID = False

GB_ESTIMATOR = GradientBoostingRegressor(random_state=RANDOM_STATE)


MODEL_CONFIGS = {
    "Gaussian descriptors": {
        # Dane z deskryptorami z Gaussiana
        "data_path": "data/QSPR_data_app.xlsx",
        "sheet_name": None,
        "split_file": "data/train_test_compounds.xlsx",
        "report_file": "data/model_REPORT_GB_by_polymer_logD+1.xlsx",
    },
    "RDKit descriptors": {
        "data_path": "data/QSPR_data_app_rdkit.xlsx",
        "sheet_name": None,
        "split_file": "data/train_test_compounds_rdkit.xlsx",
        "report_file": "data/model_REPORT_GB_by_polymer_rdkit.xlsx",
    },
}

# Helpers

def clean_number(x):
    if pd.isna(x):
        return np.nan
    s = (
        str(x)
        .replace("\u00A0", "")
        .replace("−", "-")
        .replace(",", ".")
        .strip()
    )
    if s in {"", "-"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def norm_polymer(s):
    if pd.isna(s):
        return s
    s = str(s).strip().upper()
    replacements = {
        "POLYETHYLENE": "PE",
        "POLYPROPYLENE": "PP",
        "POLYSTYRENE": "PS",
        "PE": "PE",
        "PP": "PP",
        "PS": "PS",
    }
    return replacements.get(s, s)


def rmse(y_true, y_pred):
    return float(
        np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred)))
    )


def bias(y_true, y_pred):
    return float(
        np.mean(
            np.asarray(y_pred, float) - np.asarray(y_true, float)
        )
    )


def mpe(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    mask = y != 0
    if mask.sum() == 0:
        return np.nan
    return float(
        100
        * np.mean(
            (np.asarray(y_pred, dtype=float)[mask] - y[mask]) / y[mask]
        )
    )


def mne(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    mask = np.abs(y) != 0
    if mask.sum() == 0:
        return np.nan
    return float(
        100
        * np.mean(
            np.abs(np.asarray(y_pred, dtype=float)[mask] - y[mask])
            / np.abs(y[mask])
        )
    )


def external_q2(y_true, y_pred, y_train_mean, method="F1"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if method == "F1":
        num = np.sum((y_pred - y_true) ** 2)
        den = np.sum((y_true - y_train_mean) ** 2)
    elif method == "F2":
        num = np.sum((y_pred - y_true) ** 2)
        den = np.sum((y_true - np.mean(y_true)) ** 2)
    elif method == "F3":
        num = np.sum((y_pred - y_true) ** 2)
        den = np.sum((y_pred - y_train_mean) ** 2)
    else:
        raise ValueError("Q2_TEST_METHOD must be 'F1', 'F2', or 'F3'")
    return 1 - num / den if den > 0 else np.nan


def sanitize_manual_features(candidates, available_cols):
    if not candidates:
        return [], []
    cand = [c for c in map(str, candidates)]
    ok = [c for c in cand if c in available_cols]
    missing = [c for c in cand if c not in available_cols]
    return ok, missing


def select_features_for_polymer(pol, X_train_cols, selected_combined):
    """Zwraca listę cech dla danego polimeru, zgodnie z FEATURE_MODE."""
    if FEATURE_MODE.lower() == "manual":
        cand = MANUAL_FEATURES_PER_POLYMER.get(pol, MANUAL_FEATURES)
        ok, missing = sanitize_manual_features(cand, X_train_cols)
        if missing:
            print(f"[WARN] [{pol}] Missing manual features (ignored): {missing}")
        if len(ok) == 0:
            if FALLBACK_TO_COMBINED_IF_INVALID and selected_combined:
                print(
                    f"[WARN] [{pol}] Manual list empty/invalid — fallback to 'combined': {selected_combined}"
                )
                return selected_combined
            else:
                raise ValueError(
                    f"[{pol}] Manual feature list empty/invalid and fallback disabled."
                )
        return ok
    else:
        # combined mode – wybrane cechy globalne
        return selected_combined


def compute_combined_ranking(X_train_all, Y_train_all, top_k=1):
    """Zwraca (score_df, selected_combined) – jak w Twoich skryptach."""
    score_df = pd.DataFrame()

    def safe_abs_corr(a, b, method="pearson"):
        try:
            s = pd.Series(a).astype(float)
            t = pd.Series(b).astype(float)
            r = s.corr(t, method=method)
            return float(abs(r)) if np.isfinite(r) else np.nan
        except Exception:
            return np.nan

    pearson_scores, spearman_scores, mi_scores = {}, {}, {}
    for col in X_train_all.columns:
        p_vals, s_vals, mi_vals = [], [], []
        for pol in Y_train_all.columns:
            y = Y_train_all[pol]
            mask = y.notna()
            if mask.sum() >= 3:
                xv = X_train_all.loc[mask, col].values
                yv = y.loc[mask].values
                if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
                    continue
                p_vals.append(safe_abs_corr(xv, yv, "pearson"))
                s_vals.append(safe_abs_corr(xv, yv, "spearman"))
                try:
                    mi = mutual_info_regression(
                        xv.reshape(-1, 1), yv, random_state=RANDOM_STATE
                    )
                    mi_vals.append(float(mi[0]))
                except Exception:
                    pass
        pearson_scores[col] = float(np.nanmean(p_vals)) if p_vals else np.nan
        spearman_scores[col] = float(np.nanmean(s_vals)) if s_vals else np.nan
        mi_scores[col] = float(np.nanmean(mi_vals)) if mi_vals else np.nan

    score_df = pd.DataFrame(
        {
            "pearson_abs": pd.Series(pearson_scores),
            "spearman_abs": pd.Series(spearman_scores),
            "mi": pd.Series(mi_scores),
        }
    ).fillna(0.0)

    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else pd.Series(
            0.0, index=s.index
        )

    score_df["combined"] = (
        z(score_df["pearson_abs"])
        + z(score_df["spearman_abs"])
        + z(score_df["mi"])
    )
    score_df = score_df.sort_values("combined", ascending=False)
    ranked_features = (
        list(score_df.index)
        if np.any(score_df["combined"].values)
        else list(X_train_all.var().sort_values(ascending=False).index)
    )

    def select_combined_features(ranked, X, top_k):
        selected = []
        for f in ranked:
            if len(selected) >= top_k:
                break
            if not selected:
                selected.append(f)
                continue
            mx = X[[f] + selected].corr().abs()[f].drop(f).max()
            if not np.isfinite(mx) or mx < 0.95:
                selected.append(f)
        if len(selected) < top_k:
            for f in ranked:
                if f not in selected:
                    selected.append(f)
                    if len(selected) == top_k:
                        break
        return selected

    selected_combined = select_combined_features(ranked_features, X_train_all, top_k)
    return score_df, selected_combined


# Data structure for models

class ModelState:
    def __init__(self, name, config, X_by_comp, Y_wide,
                 X_train_all, X_test_all, Y_train_all, Y_test_all,
                 feature_columns, best_params_per_polymer, selected_combined):
        self.name = name
        self.config = config
        self.X_by_comp = X_by_comp
        self.Y_wide = Y_wide
        self.X_train_all = X_train_all
        self.X_test_all = X_test_all
        self.Y_train_all = Y_train_all
        self.Y_test_all = Y_test_all
        self.feature_columns = feature_columns
        self.best_params_per_polymer = best_params_per_polymer
        self.selected_combined = selected_combined


# Pipeline loading

@st.cache_resource
def load_model_state(model_name: str) -> ModelState:
    cfg = MODEL_CONFIGS[model_name]

    data_path = cfg["data_path"]
    sheet_name = cfg.get("sheet_name", None)
    split_file = cfg.get("split_file", None)
    report_file = cfg.get("report_file", None)

    # Load the data
    if sheet_name:
        df = pd.read_excel(data_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(data_path)

    compound_col = "Organic compound" if "Organic compound" in df.columns else "Organic compounds"

    COLMAP = {"q-": "q−", "q–": "q−", "q—": "q−", "V’": "V'", "V´": "V'", "V`": "V'", "Vʼ": "V'"}
    df = df.rename(columns={c: COLMAP.get(c, c) for c in df.columns})

    if "Polymer" not in df.columns:
        raise ValueError("Column 'Polymer' not found in dataset.")
    df["Polymer"] = df["Polymer"].apply(norm_polymer)

    num_cols = ["LogKd", "logD", "εα", "εβ", "π", "M", "q−", "V'"]
    num_cols = [c for c in num_cols if c in df.columns]
    for c in num_cols:
        df[c] = df[c].apply(clean_number)

    feat_cols = [c for c in ["logD", "εα", "εβ", "π", "M", "q−", "V'"] if c in df.columns]

    Y_wide = df.pivot_table(
        index=compound_col, columns="Polymer", values="LogKd", aggfunc="first"
    )
    Y_wide = Y_wide.reindex(columns=[p for p in EXPECTED_POLYMERS if p in Y_wide.columns])

    X_by_comp = (
        df[[compound_col] + feat_cols]
        .drop_duplicates(subset=[compound_col])
        .set_index(compound_col)
        .apply(pd.to_numeric, errors="coerce")
    )
    X_by_comp = X_by_comp[~X_by_comp.isna().any(axis=1)]

    common_idx = X_by_comp.index.intersection(Y_wide.index)
    X_by_comp = X_by_comp.loc[common_idx]
    Y_wide = Y_wide.loc[common_idx]

    print(
        f"[{model_name}] After cleaning: {len(X_by_comp)} compounds; "
        f"features: {list(X_by_comp.columns)}; polymer targets: {list(Y_wide.columns)}"
    )

    # Train/test split
    if split_file:
        try:
            x_train_prev = pd.read_excel(split_file, sheet_name="X_train", index_col=0)
            x_test_prev = pd.read_excel(split_file, sheet_name="X_test", index_col=0)

            train_compounds = x_train_prev.index.astype(str)
            test_compounds = x_test_prev.index.astype(str)

            train_compounds = [c for c in train_compounds if c in X_by_comp.index]
            test_compounds = [c for c in test_compounds if c in X_by_comp.index]

            if len(train_compounds) == 0 or len(test_compounds) == 0:
                raise ValueError("Brak wspólnych związków między SPLIT_FILE a aktualnym X_by_comp.")

            print(
                f"[{model_name}][SPLIT] Używam podziału z pliku '{split_file}': "
                f"{len(train_compounds)} train, {len(test_compounds)} test związków."
            )

        except Exception as e:
            print(
                f"[{model_name}][SPLIT] Nie udało się wczytać podziału z '{split_file}' ({e}). "
                f"Tworzę nowy GroupShuffleSplit."
            )
            groups_all = np.array(X_by_comp.index)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=RANDOM_STATE)
            train_idx, test_idx = next(gss.split(X_by_comp, Y_wide, groups=groups_all))
            train_compounds = X_by_comp.index[train_idx]
            test_compounds = X_by_comp.index[test_idx]
    else:
        print(f"[{model_name}][SPLIT] Brak split_file w config – tworzę GroupShuffleSplit.")
        groups_all = np.array(X_by_comp.index)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X_by_comp, Y_wide, groups=groups_all))
        train_compounds = X_by_comp.index[train_idx]
        test_compounds = X_by_comp.index[test_idx]

    X_train_all = X_by_comp.loc[train_compounds].copy()
    X_test_all = X_by_comp.loc[test_compounds].copy()
    Y_train_all = Y_wide.loc[train_compounds].copy()
    Y_test_all = Y_wide.loc[test_compounds].copy()

    print(f"[{model_name}] Train compounds: {len(train_compounds)} | Test compounds: {len(test_compounds)}")

    feature_columns = list(X_by_comp.columns)

    # Feature ranking
    selected_combined = []
    if FEATURE_MODE.lower() != "manual":
        score_df, selected_combined = compute_combined_ranking(X_train_all, Y_train_all, TOP_K)
        print(f"[{model_name}] Selected descriptors by combined (TOP_K={TOP_K}): {selected_combined}")
    else:
        print(f"[{model_name}] FEATURE_MODE='manual' — pomijam automatyczny ranking.")

    # Hyperparameters
    best_params_per_polymer = {}
    if report_file:
        try:
            bp_df = pd.read_excel(report_file, sheet_name="Best params per polymer")
            for _, row in bp_df.iterrows():
                pol_name = str(row["Polymer"])
                params = {}

                if "n_estimators" in row and pd.notna(row["n_estimators"]):
                    params["n_estimators"] = int(row["n_estimators"])
                if "learning_rate" in row and pd.notna(row["learning_rate"]):
                    params["learning_rate"] = float(row["learning_rate"])
                if "max_depth" in row and pd.notna(row["max_depth"]):
                    params["max_depth"] = int(row["max_depth"])
                if "subsample" in row and pd.notna(row["subsample"]):
                    params["subsample"] = float(row["subsample"])
                if "max_features" in row:
                    mf = row["max_features"]
                    params["max_features"] = None if pd.isna(mf) else mf
                if "min_samples_split" in row and pd.notna(row["min_samples_split"]):
                    params["min_samples_split"] = int(row["min_samples_split"])
                if "min_samples_leaf" in row and pd.notna(row["min_samples_leaf"]):
                    params["min_samples_leaf"] = int(row["min_samples_leaf"])

                best_params_per_polymer[pol_name] = params

            print(
                f"[{model_name}][HP] Wczytano zapisane hiperparametry dla polimerów z '{report_file}': "
                f"{list(best_params_per_polymer.keys())}"
            )
        except FileNotFoundError:
            print(f"[{model_name}][HP] Plik '{report_file}' nie istnieje — użyję domyślnych parametrów GB.")
        except Exception as e:
            print(
                f"[{model_name}][HP] Problem z wczytaniem hiperparametrów z '{report_file}' ({e}) — "
                f"użyję domyślnych parametrów GB."
            )
    else:
        print(f"[{model_name}][HP] Brak report_file w config — użyję domyślnych parametrów GB.")

    state = ModelState(
        name=model_name,
        config=cfg,
        X_by_comp=X_by_comp,
        Y_wide=Y_wide,
        X_train_all=X_train_all,
        X_test_all=X_test_all,
        Y_train_all=Y_train_all,
        Y_test_all=Y_test_all,
        feature_columns=feature_columns,
        best_params_per_polymer=best_params_per_polymer,
        selected_combined=selected_combined,
    )
    return state



# Functions depend on the condition (polymer)

def get_estimator_for_polymer(state: ModelState, polymer: str):
    """Tworzy i trenuje estymator na pełnym zbiorze (Y_wide notna), z zapisanymi hiperparametrami jeśli są."""
    if polymer not in state.Y_wide.columns:
        raise ValueError(f"Polymer {polymer} not found in Y_wide for model '{state.name}'.")

    # wybór cech
    selected_features = select_features_for_polymer(
        polymer,
        state.X_by_comp.columns,
        state.selected_combined,
    )

    y_full = state.Y_wide[polymer].dropna()
    if len(y_full) < 4:
        raise ValueError(
            f"Not enough data to train model for polymer {polymer} (n={len(y_full)})."
        )

    X_full = state.X_by_comp.loc[y_full.index, selected_features]

    params = state.best_params_per_polymer.get(polymer, {})
    est = GradientBoostingRegressor(
        random_state=RANDOM_STATE, **params
    )
    est.fit(X_full, y_full)

    return est, selected_features, X_full, y_full


def get_ad_basis_for_polymer(state: ModelState, polymer: str):
    """Zwraca (est, selected_feats, XtX_inv, h_crit, X_full) dla danego modelu i polimeru."""
    est, selected_feats, X_full, y_full = get_estimator_for_polymer(state, polymer)

    X_design = np.column_stack([np.ones(len(X_full)), X_full.values])
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)

    n = X_full.shape[0]
    p = X_full.shape[1]
    h_crit = 3.0 * (p + 1) / n

    return est, selected_feats, XtX_inv, float(h_crit), X_full


def compute_insubria_for_polymer(state: ModelState, polymer: str):
    est, selected_feats, XtX_inv, h_crit, X_full = get_ad_basis_for_polymer(
        state, polymer
    )

    y_pred = est.predict(X_full)

    X_design = np.column_stack([np.ones(len(X_full)), X_full.values])
    AX = X_design @ XtX_inv
    h = np.sum(AX * X_design, axis=1)

    df_ins = pd.DataFrame(
        {
            "Compound": X_full.index.astype(str),
            "LogKd_pred": y_pred,
            "Leverage_h": h,
        }
    ).set_index("Compound")

    df_ins["AD_flag"] = np.where(
        df_ins["Leverage_h"] <= h_crit, "Inside AD", "Outside AD"
    )

    return df_ins, float(h_crit)


def predict_single_compound(
    state: ModelState, descriptor_dict: dict, polymers: list[str], compound_name: str = "compound"
) -> pd.DataFrame:
    X_new = pd.DataFrame([descriptor_dict], index=[compound_name])
    rows = []

    for pol in polymers:
        try:
            est, selected_feats, XtX_inv, h_crit, X_full = get_ad_basis_for_polymer(
                state, pol
            )

            X_pol = X_new[selected_feats]
            y_pred = est.predict(X_pol)[0]

            x_vals = X_pol.iloc[0].values.astype(float)
            x_design = np.concatenate(([1.0], x_vals))
            x_vec = x_design.reshape(1, -1)
            h_new = float((x_vec @ XtX_inv @ x_vec.T)[0, 0])

            ad_flag = "Inside AD" if h_new <= h_crit else "Outside AD"

            rows.append(
                {
                    "Compound": compound_name,
                    "Polymer": pol,
                    "LogKd_pred": y_pred,
                    "Leverage_h": h_new,
                    "h_crit": h_crit,
                    "AD_flag": ad_flag,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "Compound": compound_name,
                    "Polymer": pol,
                    "LogKd_pred": np.nan,
                    "Leverage_h": np.nan,
                    "h_crit": np.nan,
                    "AD_flag": f"Error: {e}",
                }
            )

    return pd.DataFrame(rows)


def predict_batch(state: ModelState, df_input: pd.DataFrame, polymers: list[str]) -> pd.DataFrame:
    """
    Dla każdej próbki i każdego polimeru dodaje:
    - LogKd_{POL}
    - Leverage_h_{POL}
    - AD_flag_{POL}
    """
    df = df_input.copy()

    FEATURE_COLUMNS = list(state.feature_columns)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    for pol in polymers:
        est, selected_feats, XtX_inv, h_crit, X_full = get_ad_basis_for_polymer(
            state, pol
        )

        X_pol = df[selected_feats]
        preds = est.predict(X_pol)
        df[f"LogKd_{pol}"] = preds

        X_design_new = np.column_stack([np.ones(len(X_pol)), X_pol.values])
        AX_new = X_design_new @ XtX_inv
        h_new = np.sum(AX_new * X_design_new, axis=1)

        df[f"Leverage_h_{pol}"] = h_new
        df[f"AD_flag_{pol}"] = np.where(
            h_new <= h_crit, "Inside AD", "Outside AD"
        )

    return df


# STREAMLIT UI

st.set_page_config(
    page_title="LogKd prediction for microplastics – two models",
    layout="wide",
)

st.title("LogKd prediction for microplastics")

st.write(
    """
This app allows you to use **two alternative GradientBoosting models**:

- **Gaussian descriptors** – model built with descriptors (e.g. π) computed via *Gaussian*,
- **RDKit descriptors** – model built with descriptors computed via *RDKit*.

Both models share the same interface:
- **Single prediction** (with applicability domain),
- **Batch prediction** (with AD),
- **Insubria plot** for the training domain.

Use the selector in the sidebar to choose which model you want to apply.
"""
)

# --- wybór modelu ---
model_name = st.sidebar.selectbox(
    "Choose model / descriptor set",
    list(MODEL_CONFIGS.keys()),
    index=0,
)

# --- załaduj odpowiedni ModelState ---
with st.spinner(f"Loading model data for: {model_name}"):
    state = load_model_state(model_name)

AVAILABLE_POLYMERS = [p for p in EXPECTED_POLYMERS if p in state.Y_wide.columns]
if not AVAILABLE_POLYMERS:
    st.error(f"No polymers found in Y_wide for model '{model_name}'. Check that the data loaded correctly.")
    st.stop()

FEATURE_COLUMNS = list(state.feature_columns)

st.sidebar.markdown(f"**Active model:** `{model_name}`")
st.sidebar.markdown("---")
st.sidebar.header("Model settings")

selected_polymers = st.sidebar.multiselect(
    "Select polymers for prediction",
    options=AVAILABLE_POLYMERS,
    default=AVAILABLE_POLYMERS,
)

if not selected_polymers:
    st.warning("Select at least one polymer to run predictions.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write("Descriptors used by the *selected* model:")
for c in FEATURE_COLUMNS:
    st.sidebar.write(f"- {c}")

tab_single, tab_batch, tab_insubria = st.tabs(
    ["Single prediction", "Batch prediction", "Insubria plot (training data)"]
)

# Single prediction
with tab_single:
    st.subheader(f"Single compound prediction – {model_name}")

    compound_name = st.text_input("Compound ID / name", value="your_compound")

    st.write(
        "Provide descriptor values. These must match the scale used in the training data "
        f"for **{model_name}**."
    )

    descriptor_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            descriptor_values[feat] = st.number_input(
                label=f"{feat}",
                value=0.0,
                format="%.6f",
                key=f"{model_name}_single_{feat}",
            )

    if st.button("Predict LogKd for this compound", key=f"{model_name}_btn_single"):
        with st.spinner("Running predictions..."):
            df_res = predict_single_compound(
                state, descriptor_values, selected_polymers, compound_name
            )

        st.write("Predictions with applicability domain info:")
        st.dataframe(df_res, use_container_width=True)

        st.markdown("### Insubria plots for this compound")

        for pol in selected_polymers:
            st.markdown(f"**Polymer: {pol}**")

            row_pol = df_res[df_res["Polymer"] == pol]
            if row_pol.empty:
                st.warning(f"No prediction for polymer {pol}.")
                continue

            row_pol = row_pol.iloc[0]
            h_new = row_pol["Leverage_h"]
            y_new = row_pol["LogKd_pred"]

            if pd.isna(h_new) or pd.isna(y_new):
                st.warning(
                    f"Cannot build Insubria plot for {pol} (no valid AD data for this prediction)."
                )
                continue

            try:
                df_ins_train, h_crit = compute_insubria_for_polymer(state, pol)
            except Exception as e:
                st.error(f"Cannot build Insubria plot for {pol}: {e}")
                continue

            fig, ax = plt.subplots(figsize=(6, 4))

            df_in = df_ins_train[df_ins_train["AD_flag"] == "Inside AD"]
            df_out = df_ins_train[df_ins_train["AD_flag"] == "Outside AD"]

            if not df_in.empty:
                ax.scatter(
                    df_in["Leverage_h"],
                    df_in["LogKd_pred"],
                    label="Training – inside AD",
                    alpha=0.7,
                )
            if not df_out.empty:
                ax.scatter(
                    df_out["Leverage_h"],
                    df_out["LogKd_pred"],
                    label="Training – outside AD",
                    marker="s",
                    alpha=0.8,
                )

            ax.scatter(
                h_new,
                y_new,
                marker="*",
                s=140,
                linewidths=1.2,
                label=f"New compound: {compound_name}",
            )

            ax.axvline(
                h_crit,
                linestyle="--",
                linewidth=1.5,
                label=f"h* = {h_crit:.3f}",
            )

            y_min_tr = float(df_ins_train["LogKd_pred"].min())
            y_max_tr = float(df_ins_train["LogKd_pred"].max())
            x_min, x_max = ax.get_xlim()
            ax.hlines(
                [y_min_tr, y_max_tr],
                x_min,
                x_max,
                linewidth=1.0,
            )
            ax.set_xlim(x_min, x_max)

            ax.set_xlabel("Leverage (h)")
            ax.set_ylabel("Predicted LogKd")
            ax.set_title(f"Insubria plot – {pol} – {model_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label=f"Download Insubria plot for {pol} as PNG",
                data=buf,
                file_name=f"insubria_single_{compound_name}_{pol}_{model_name}.png",
                mime="image/png",
                key=f"dl_single_{pol}_{model_name}",
            )

# Batch prediction
with tab_batch:
    st.subheader(f"Batch prediction from file – {model_name}")

    st.write(
        """
Upload a **CSV or Excel** file that contains at least the following columns  
(these are the descriptors used in the selected model):
"""
    )
    st.code(", ".join(FEATURE_COLUMNS), language="text")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx"], key=f"{model_name}_upload"
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
            if st.button("Run batch prediction", key=f"{model_name}_btn_batch"):
                with st.spinner("Running batch predictions..."):
                    try:
                        df_pred = predict_batch(state, df_input, selected_polymers)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                    else:
                        st.write(
                            "Predictions with applicability domain info (first rows):"
                        )
                        st.dataframe(
                            df_pred.head(), use_container_width=True
                        )

                        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download full predictions as CSV",
                            data=csv_bytes,
                            file_name=f"logKd_predictions_with_AD_{model_name}.csv",
                            mime="text/csv",
                        )

                        st.markdown("### Insubria plots for this batch (all selected polymers)")

                        for pol_batch in selected_polymers:
                            st.markdown(f"**Polymer: {pol_batch}**")

                            col_h = f"Leverage_h_{pol_batch}"
                            col_y = f"LogKd_{pol_batch}"
                            if col_h not in df_pred.columns or col_y not in df_pred.columns:
                                st.warning(f"No prediction columns for polymer {pol_batch} in batch output.")
                                continue

                            try:
                                df_ins_train, h_crit = compute_insubria_for_polymer(state, pol_batch)
                            except Exception as e:
                                st.error(f"Cannot build Insubria plot for {pol_batch}: {e}")
                                continue

                            h_new = df_pred[col_h].values
                            y_new = df_pred[col_y].values

                            fig_b, ax_b = plt.subplots(figsize=(6, 4))

                            df_in = df_ins_train[df_ins_train["AD_flag"] == "Inside AD"]
                            df_out = df_ins_train[df_ins_train["AD_flag"] == "Outside AD"]

                            if not df_in.empty:
                                ax_b.scatter(
                                    df_in["Leverage_h"],
                                    df_in["LogKd_pred"],
                                    label="Training – inside AD",
                                    alpha=0.7,
                                )
                            if not df_out.empty:
                                ax_b.scatter(
                                    df_out["Leverage_h"],
                                    df_out["LogKd_pred"],
                                    label="Training – outside AD",
                                    marker="s",
                                    alpha=0.8,
                                )

                            ax_b.scatter(
                                h_new,
                                y_new,
                                marker="^",
                                s=60,
                                label="Batch samples",
                                alpha=0.9,
                            )

                            ax_b.axvline(
                                h_crit,
                                linestyle="--",
                                linewidth=1.5,
                                label=f"h* = {h_crit:.3f}",
                            )

                            y_min_tr = float(df_ins_train["LogKd_pred"].min())
                            y_max_tr = float(df_ins_train["LogKd_pred"].max())
                            x_min, x_max = ax_b.get_xlim()
                            ax_b.hlines(
                                [y_min_tr, y_max_tr],
                                x_min,
                                x_max,
                                linewidth=1.0,
                            )
                            ax_b.set_xlim(x_min, x_max)

                            ax_b.set_xlabel("Leverage (h)")
                            ax_b.set_ylabel("Predicted LogKd")
                            ax_b.set_title(
                                f"Insubria plot – batch vs training ({pol_batch}) – {model_name}"
                            )
                            ax_b.legend()
                            ax_b.grid(True, alpha=0.3)

                            st.pyplot(fig_b)

                            buf_b = io.BytesIO()
                            fig_b.savefig(
                                buf_b,
                                format="png",
                                dpi=300,
                                bbox_inches="tight",
                            )
                            buf_b.seek(0)

                            st.download_button(
                                label=f"Download batch Insubria plot ({pol_batch}) as PNG",
                                data=buf_b,
                                file_name=f"insubria_batch_{pol_batch}_{model_name}.png",
                                mime="image/png",
                                key=f"dl_batch_insubria_{pol_batch}_{model_name}",
                            )

# Insubria plot (training only)
with tab_insubria:
    st.subheader(f"Insubria plot – training data only – {model_name}")

    pol_for_ad = st.selectbox(
        "Select polymer for Insubria plot (training data)",
        options=AVAILABLE_POLYMERS,
        index=0,
        key=f"{model_name}_insubria_polymer_train",
    )

    if st.button("Generate Insubria plot (training)", key=f"{model_name}_btn_insubria_train"):
        try:
            df_ins, h_crit = compute_insubria_for_polymer(state, pol_for_ad)
        except Exception as e:
            st.error(f"Cannot generate Insubria plot: {e}")
        else:
            st.markdown(
                f"""
                **Insubria plot** (leverage vs predicted LogKd) for polymer **{pol_for_ad}** (training data)  
                in **{model_name}**.  

                Vertical line at **h* = 3(p+1)/n = {h_crit:.3f}** marks the applicability domain threshold.  
                Points with leverage > h* are **outside AD**.
                """
            )

            fig_t, ax_t = plt.subplots(figsize=(7, 5))

            df_in = df_ins[df_ins["AD_flag"] == "Inside AD"]
            df_out = df_ins[df_ins["AD_flag"] == "Outside AD"]

            if not df_in.empty:
                ax_t.scatter(
                    df_in["Leverage_h"],
                    df_in["LogKd_pred"],
                    label="Inside AD",
                    alpha=0.8,
                )
            if not df_out.empty:
                ax_t.scatter(
                    df_out["Leverage_h"],
                    df_out["LogKd_pred"],
                    label="Outside AD",
                    marker="s",
                    alpha=0.9,
                )

            ax_t.axvline(
                h_crit,
                linestyle="--",
                linewidth=1.5,
                label=f"h* = {h_crit:.3f}",
            )

            y_min_tr = float(df_ins["LogKd_pred"].min())
            y_max_tr = float(df_ins["LogKd_pred"].max())
            x_min, x_max = ax_t.get_xlim()
            ax_t.hlines(
                [y_min_tr, y_max_tr],
                x_min,
                x_max,
                linewidth=1.0,
            )
            ax_t.set_xlim(x_min, x_max)

            ax_t.set_xlabel("Leverage (h)")
            ax_t.set_ylabel("Predicted LogKd")
            ax_t.set_title(f"Insubria plot – training data ({pol_for_ad}) – {model_name}")
            ax_t.legend()
            ax_t.grid(True, alpha=0.3)

            st.pyplot(fig_t)

            buf_t = io.BytesIO()
            fig_t.savefig(
                buf_t, format="png", dpi=300, bbox_inches="tight"
            )
            buf_t.seek(0)

            st.download_button(
                label="Download Insubria plot (training) as PNG",
                data=buf_t,
                file_name=f"insubria_training_{pol_for_ad}_{model_name}.png",
                mime="image/png",
                key=f"{model_name}_dl_training_insubria",
            )

            csv_ins = df_ins.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Insubria data (training) as CSV",
                data=csv_ins,
                file_name=f"insubria_training_{pol_for_ad}_{model_name}.csv",
                mime="text/csv",
                key=f"{model_name}_dl_training_insubria_csv",
            )

            st.write("Data used for the plot:")
            st.dataframe(
                df_ins.sort_values("Leverage_h", ascending=False),
                use_container_width=True,
            )

st.markdown("---")
st.caption(
    "Use the model selector in the sidebar to switch between Gaussian-based and RDKit-based models. "
    "Each model uses its own data, train/test split and (optionally) saved hyperparameters."
)
