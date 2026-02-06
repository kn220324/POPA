import io

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


# =========================
# Global parameters (wspólne)
# =========================
EXPECTED_POLYMERS = ["PE", "PP", "PS"]
RANDOM_STATE = 42
Q2_TEST_METHOD = "F1"  # "F1", "F2", "F3"

# --- Colors for polymers (plots) ---
poly_colors = {
    "PE": "#1f77b4",
    "PP": "tomato",
    "PS": "forestgreen",
}

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

MODEL_CONFIGS = {
    "Gaussian descriptors": {
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

# =========================
# Helpers
# =========================
def clean_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\u00A0", "").replace("−", "-").replace(",", ".").strip()
    if s in {"", "-"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def rdkit_descriptors_from_smiles(smiles: str):
    """
    Computes descriptors needed by RDKit model:
    M' = MW / 100
    V' = LabuteASA / 100
    π  = MolMR / V'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    mw = Descriptors.MolWt(mol)
    labute = rdMolDescriptors.CalcLabuteASA(mol)
    molmr = Descriptors.MolMR(mol)

    Vp = labute / 100.0
    if Vp == 0:
        raise ValueError("LabuteASA returned zero.")

    Mp = mw / 100.0
    pi_val = molmr / Vp

    return {
        "M": Mp,
        "V'": Vp,
        "π": pi_val,
    }

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
                print(f"[WARN] [{pol}] Manual list empty/invalid — fallback to 'combined': {selected_combined}")
                return selected_combined
            raise ValueError(f"[{pol}] Manual feature list empty/invalid and fallback disabled.")
        return ok
    return selected_combined


def compute_combined_ranking(X_train_all, Y_train_all, top_k=1):
    """Zwraca (score_df, selected_combined) – ranking + selekcja jak w Twoich skryptach."""
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
                    mi = mutual_info_regression(xv.reshape(-1, 1), yv, random_state=RANDOM_STATE)
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
        return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else pd.Series(0.0, index=s.index)

    score_df["combined"] = z(score_df["pearson_abs"]) + z(score_df["spearman_abs"]) + z(score_df["mi"])
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


# =========================
# Data structure for models
# =========================
class ModelState:
    def __init__(
        self,
        name,
        config,
        X_by_comp,
        Y_wide,
        X_train_all,
        X_test_all,
        Y_train_all,
        Y_test_all,
        feature_columns,
        best_params_per_polymer,
        selected_combined,
    ):
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


# =========================
# Pipeline loading
# =========================
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

    # normalize columns
    COLMAP = {
        "q-": "q−",
        "q–": "q−",
        "q—": "q−",
        "V’": "V'",
        "V´": "V'",
        "V`": "V'",
        "Vʼ": "V'",
    }
    df = df.rename(columns={c: COLMAP.get(c, c) for c in df.columns})

    if "Polymer" not in df.columns:
        raise ValueError("Column 'Polymer' not found in dataset.")
    df["Polymer"] = df["Polymer"].apply(norm_polymer)

    # numeric cleanup
    num_cols = ["LogKd", "logD", "εα", "εβ", "π", "M", "q−", "V'"]
    num_cols = [c for c in num_cols if c in df.columns]
    for c in num_cols:
        df[c] = df[c].apply(clean_number)

    # features available
    feat_cols = [c for c in ["logD", "εα", "εβ", "π", "M", "q−", "V'"] if c in df.columns]

    # wide y
    Y_wide = df.pivot_table(index=compound_col, columns="Polymer", values="LogKd", aggfunc="first")
    Y_wide = Y_wide.reindex(columns=[p for p in EXPECTED_POLYMERS if p in Y_wide.columns])

    # X per compound
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

    # Train/test split (prefer saved split)
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

    # Hyperparameters from report (optional)
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
                f"[{model_name}][HP] Wczytano hiperparametry z '{report_file}' dla: "
                f"{list(best_params_per_polymer.keys())}"
            )
        except FileNotFoundError:
            print(f"[{model_name}][HP] Plik '{report_file}' nie istnieje — użyję domyślnych parametrów GB.")
        except Exception as e:
            print(f"[{model_name}][HP] Problem z wczytaniem hiperparametrów ({e}) — użyję domyślnych parametrów GB.")
    else:
        print(f"[{model_name}][HP] Brak report_file w config — użyję domyślnych parametrów GB.")

    return ModelState(
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


# =========================
# Model + AD basis
# =========================
def get_estimator_for_polymer(state: ModelState, polymer: str):
    """Tworzy i trenuje estymator na pełnym zbiorze (Y_wide notna), z zapisanymi hiperparametrami jeśli są."""
    if polymer not in state.Y_wide.columns:
        raise ValueError(f"Polymer {polymer} not found in Y_wide for model '{state.name}'.")

    selected_features = select_features_for_polymer(polymer, state.X_by_comp.columns, state.selected_combined)

    y_full = state.Y_wide[polymer].dropna()
    if len(y_full) < 4:
        raise ValueError(f"Not enough data to train model for polymer {polymer} (n={len(y_full)}).")

    X_full = state.X_by_comp.loc[y_full.index, selected_features]

    params = state.best_params_per_polymer.get(polymer, {})
    est = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    est.fit(X_full, y_full)

    return est, selected_features, X_full, y_full


def get_ad_basis_for_polymer(state: ModelState, polymer: str):
    """
    Zwraca bazę AD:
    - est: model (fitowany na full danych dla danego polimeru)
    - selected_feats
    - XtX_inv: (X'X)^-1 dla TRAIN (z interceptem) -> leverage dla nowych związków
    - h_crit = 3(p+1)/n_train (jak w AD)
    """
    est, selected_feats, X_full, y_full = get_estimator_for_polymer(state, polymer)

    # leverage basis liczmy na TRAIN (żeby AD było stricte treningowe)
    ytr = state.Y_train_all[polymer].dropna()
    Xtr = state.X_train_all.loc[ytr.index, selected_feats].dropna(axis=0, how="any")
    ytr = ytr.loc[Xtr.index]

    n = Xtr.shape[0]
    p = Xtr.shape[1]
    if n < (p + 2):
        raise ValueError(f"Too few TRAIN samples to compute leverage for {polymer}: n={n}, p={p}.")

    X_design = np.column_stack([np.ones(len(Xtr)), Xtr.values])
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)

    h_crit = 3.0 * (p + 1) / n
    return est, selected_feats, XtX_inv, float(h_crit), Xtr, ytr


# =========================
# TRAINING AD plot (to co miałaś jako "Insubria training")
# =========================
def compute_training_ad_plot_data(state: ModelState, polymer: str):
    """
    To NIE jest Insubria wg Twojej definicji.
    To jest training-domain plot: leverage vs predicted (training points),
    z flagą inside/outside AD.
    """
    est, selected_feats, XtX_inv, h_crit, Xtr, ytr = get_ad_basis_for_polymer(state, polymer)

    y_pred_tr = est.predict(Xtr)

    X_design_tr = np.column_stack([np.ones(len(Xtr)), Xtr.values])
    AX = X_design_tr @ XtX_inv
    h = np.sum(AX * X_design_tr, axis=1)

    df_tr = pd.DataFrame(
        {
            "Compound": Xtr.index.astype(str),
            "LogKd_true": ytr.values.astype(float),
            "LogKd_pred": y_pred_tr.astype(float),
            "Leverage_h": h.astype(float),
        }
    ).set_index("Compound")

    df_tr["AD_flag"] = np.where(df_tr["Leverage_h"] <= h_crit, "Inside AD", "Outside AD")
    return df_tr, float(h_crit)


# =========================
# INSUBRIA BASIS (zgodnie z Twoim komentarzem)
# Insubria plot pokazuje TYLKO nowe związki (spoza train i test).
# Granice Y: min/max EKSPERYMENTALNE w TRAIN (y_train^exp).
# =========================
def compute_insubria_basis_for_polymer(state: ModelState, polymer: str):
    est, selected_feats, XtX_inv, h_crit, Xtr, ytr = get_ad_basis_for_polymer(state, polymer)

    y_min_train = float(np.min(ytr.values.astype(float)))
    y_max_train = float(np.max(ytr.values.astype(float)))

    return {
        "est": est,
        "selected_feats": selected_feats,
        "XtX_inv": XtX_inv,
        "h_crit": float(h_crit),
        "y_min_train": y_min_train,
        "y_max_train": y_max_train,
        "train_index": set(Xtr.index.astype(str)),
        "test_index": set(state.X_test_all.index.astype(str)),
    }


def leverage_for_new(X_new: pd.DataFrame, XtX_inv: np.ndarray) -> np.ndarray:
    X_design_new = np.column_stack([np.ones(len(X_new)), X_new.values.astype(float)])
    AX_new = X_design_new @ XtX_inv
    h_new = np.sum(AX_new * X_design_new, axis=1)
    return h_new


def filter_new_compounds_only(df_in: pd.DataFrame, compound_col: str, train_index: set, test_index: set) -> pd.DataFrame:
    """
    Insubria: zostaw tylko te wiersze, których ID nie jest w TRAIN ani w TEST.
    """
    ids = df_in[compound_col].astype(str)
    mask_new = ~ids.isin(train_index) & ~ids.isin(test_index)
    return df_in.loc[mask_new].copy()


# =========================
# Predictions
# =========================
def predict_single_compound(state: ModelState, descriptor_dict: dict, polymers: list[str], compound_name: str = "compound") -> pd.DataFrame:
    X_new = pd.DataFrame([descriptor_dict], index=[compound_name])
    rows = []

    for pol in polymers:
        try:
            basis = compute_insubria_basis_for_polymer(state, pol)
            est = basis["est"]
            selected_feats = basis["selected_feats"]
            XtX_inv = basis["XtX_inv"]
            h_crit = basis["h_crit"]

            X_pol = X_new[selected_feats]
            y_pred = float(est.predict(X_pol)[0])

            h_new = float(leverage_for_new(X_pol, XtX_inv)[0])
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
        basis = compute_insubria_basis_for_polymer(state, pol)
        est = basis["est"]
        selected_feats = basis["selected_feats"]
        XtX_inv = basis["XtX_inv"]
        h_crit = basis["h_crit"]

        X_pol = df[selected_feats]
        preds = est.predict(X_pol)
        df[f"LogKd_{pol}"] = preds

        h_new = leverage_for_new(X_pol, XtX_inv)
        df[f"Leverage_h_{pol}"] = h_new
        df[f"AD_flag_{pol}"] = np.where(h_new <= h_crit, "Inside AD", "Outside AD")

    return df


# =========================
# STREAMLIT UI
# =========================
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
- **SMILES input** (only for RDKit model)
- **Single prediction** (with AD)
- **Batch prediction** (with AD),

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

# Renamed 3rd tab to reflect strict definition
if model_name == "RDKit descriptors":
    tab_smiles, tab_single, tab_batch = st.tabs(
        [
            "SMILES input (RDKit auto descriptors)",
            "Single prediction",
            "Batch prediction",
        ]
    )
else:
    tab_single, tab_batch = st.tabs(
        [
            "Single prediction",
            "Batch prediction",
        ]
    )

# =========================
# SMILES → RDKit descriptors
# =========================
if model_name == "RDKit descriptors":
    with tab_smiles:
        st.subheader("Prediction directly from SMILES (RDKit descriptors)")

        st.write("""
Descriptors computed automatically:

• M′
                 
• π

The model requires additional descriptors (logD), provide them manually.
""")

        mode = st.radio("Mode", ["Single SMILES", "Batch (file upload)"])

        # ---------- SINGLE ----------
        if mode == "Single SMILES":
            smiles = st.text_input("SMILES string")
            compound_name = st.text_input("Compound name", "smiles_compound")

            logD_input = st.number_input(
                "logD (if required by model)",
                value=0.0,
                format="%.6f",
            )

            if st.button("Predict from SMILES"):
                if smiles.strip() == "":
                    st.warning("Provide SMILES.")
                else:
                    try:
                        desc = rdkit_descriptors_from_smiles(smiles)
                        desc["logD"] = logD_input

                        df_res = predict_single_compound(
                            state,
                            desc,
                            selected_polymers,
                            compound_name,
                        )

                        st.dataframe(df_res, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")

        # ---------- BATCH ----------
        else:
            uploaded = st.file_uploader(
                "Upload CSV/Excel containing SMILES column",
                type=["csv", "xlsx"],
            )

            smiles_col = st.text_input("SMILES column name", value="SMILES")

            if uploaded is not None:
                if uploaded.name.endswith(".csv"):
                    df_in = pd.read_csv(uploaded)
                else:
                    df_in = pd.read_excel(uploaded)

                st.write("Uploaded data preview:")
                st.dataframe(df_in.head())

                if smiles_col not in df_in.columns:
                    st.error("SMILES column not found.")
                else:
                    if st.button("Run batch SMILES prediction"):
                        rows = []

                        for _, row in df_in.iterrows():
                            try:
                                desc = rdkit_descriptors_from_smiles(row[smiles_col])

                                if "logD" in df_in.columns:
                                    desc["logD"] = row["logD"]

                                rows.append(desc)
                            except Exception:
                                rows.append({})

                        df_desc = pd.DataFrame(rows)

                        df_pred = predict_batch(
                            state,
                            df_desc,
                            selected_polymers,
                        )

                        st.dataframe(df_pred.head(), use_container_width=True)

                        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download predictions",
                            data=csv_bytes,
                            file_name="smiles_predictions.csv",
                            mime="text/csv",
                        )

# =========================
# Single prediction
# =========================
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
            df_res = predict_single_compound(state, descriptor_values, selected_polymers, compound_name)

        st.write("Predictions with applicability domain info:")
        st.dataframe(df_res, use_container_width=True)

        st.markdown("### Insubria plots for this compound (STRICT: only NEW compound)")

        for pol in selected_polymers:
            st.markdown(f"**Polymer: {pol}**")
            poly_color = poly_colors.get(pol, "gray")

            row_pol = df_res[df_res["Polymer"] == pol]
            if row_pol.empty:
                st.warning(f"No prediction for polymer {pol}.")
                continue

            row_pol = row_pol.iloc[0]
            h_new = row_pol["Leverage_h"]
            y_new = row_pol["LogKd_pred"]

            if pd.isna(h_new) or pd.isna(y_new):
                st.warning(f"Cannot build Insubria plot for {pol} (no valid AD data for this prediction).")
                continue

            try:
                basis = compute_insubria_basis_for_polymer(state, pol)
            except Exception as e:
                st.error(f"Cannot build Insubria plot for {pol}: {e}")
                continue

            h_crit = basis["h_crit"]
            y_min_tr = basis["y_min_train"]
            y_max_tr = basis["y_max_train"]

            fig, ax = plt.subplots(figsize=(6, 4))

            # STRICT Insubria: plot only NEW compound(s)
            ax.scatter(
                h_new,
                y_new,
                marker="*",
                s=190,
                linewidths=1.2,
                label=f"New compound: {compound_name}",
                color=poly_color,
                edgecolors="black",
                alpha=0.8,
            )

            ax.axvline(
                h_crit,
                linestyle="--",
                linewidth=1.5,
                color="gray",
                label=f"h* = {h_crit:.3f}",
            )

            # y-bounds = min/max EXPERIMENTAL from TRAIN
            x_min, x_max = ax.get_xlim()
            ax.hlines([y_min_tr, y_max_tr], x_min, x_max, linewidth=1.2, color="gray")
            ax.set_xlim(x_min, x_max)

            ax.set_xlabel("Leverage (h)")
            ax.set_ylabel("Predicted LogKd")
            ax.set_title(f"Insubria plot (NEW only) – {pol} – {model_name}")
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

# =========================
# Batch prediction
# =========================
with tab_batch:
    st.subheader(f"Batch prediction from file – {model_name}")

    st.write(
        """
Upload a **CSV or Excel** file that contains at least the following columns  
(these are the descriptors used in the selected model):
"""
    )
    st.code(", ".join(FEATURE_COLUMNS), language="text")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key=f"{model_name}_upload")

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
                        st.write("Predictions with applicability domain info (first rows):")
                        st.dataframe(df_pred.head(), use_container_width=True)

                        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download full predictions as CSV",
                            data=csv_bytes,
                            file_name=f"logKd_predictions_with_AD_{model_name}.csv",
                            mime="text/csv",
                        )

                        st.markdown("### Insubria plots for this batch (STRICT: only NEW compounds)")

                        # optional: allow user to specify ID column to filter out train/test
                        compound_id_col = st.text_input(
                            "Optional: column name with compound IDs (to exclude those present in train/test). "
                            "Leave empty to treat all rows as NEW.",
                            value="",
                            key=f"{model_name}_batch_id_col",
                        ).strip()

                        df_for_plot = df_pred.copy()
                        if compound_id_col and compound_id_col in df_for_plot.columns:
                            # filter out compounds that belong to train/test for each polymer basis
                            # (train/test sets are the same per model_name, so OK to do globally)
                            # We'll filter once using polymer-independent sets from the first polymer basis.
                            try:
                                basis0 = compute_insubria_basis_for_polymer(state, selected_polymers[0])
                                train_set = basis0["train_index"]
                                test_set = basis0["test_index"]
                                df_for_plot = filter_new_compounds_only(df_for_plot, compound_id_col, train_set, test_set)
                                st.info(f"Filtered to NEW-only rows: {len(df_for_plot)} (excluded train/test IDs).")
                            except Exception as e:
                                st.warning(f"Could not filter by ID column ({e}). Plotting all rows as NEW.")

                        for pol_batch in selected_polymers:
                            st.markdown(f"**Polymer: {pol_batch}**")
                            poly_color = poly_colors.get(pol_batch, "gray")

                            col_h = f"Leverage_h_{pol_batch}"
                            col_y = f"LogKd_{pol_batch}"
                            if col_h not in df_for_plot.columns or col_y not in df_for_plot.columns:
                                st.warning(f"No prediction columns for polymer {pol_batch} in batch output.")
                                continue

                            try:
                                basis = compute_insubria_basis_for_polymer(state, pol_batch)
                            except Exception as e:
                                st.error(f"Cannot build Insubria plot for {pol_batch}: {e}")
                                continue

                            h_crit = basis["h_crit"]
                            y_min_tr = basis["y_min_train"]
                            y_max_tr = basis["y_max_train"]

                            h_new = df_for_plot[col_h].values
                            y_new = df_for_plot[col_y].values

                            fig_b, ax_b = plt.subplots(figsize=(6, 4))

                            # STRICT Insubria: plot only NEW batch points (no training points)
                            ax_b.scatter(
                                h_new,
                                y_new,
                                marker="^",
                                s=140,
                                label="New compounds (batch)",
                                alpha=0.75,
                                color=poly_color,
                                edgecolors="black",
                                linewidths=0.6,
                            )

                            ax_b.axvline(
                                h_crit,
                                linestyle="--",
                                linewidth=1.5,
                                color="gray",
                                label=f"h* = {h_crit:.3f}",
                            )

                            x_min, x_max = ax_b.get_xlim()
                            ax_b.hlines([y_min_tr, y_max_tr], x_min, x_max, linewidth=1.2, color="gray")
                            ax_b.set_xlim(x_min, x_max)

                            ax_b.set_xlabel("Leverage (h)")
                            ax_b.set_ylabel("Predicted LogKd")
                            ax_b.set_title(f"Insubria plot (NEW only) – {pol_batch} – {model_name}")
                            ax_b.legend()
                            ax_b.grid(True, alpha=0.3)

                            st.pyplot(fig_b)

                            buf_b = io.BytesIO()
                            fig_b.savefig(buf_b, format="png", dpi=300, bbox_inches="tight")
                            buf_b.seek(0)

                            st.download_button(
                                label=f"Download batch Insubria plot ({pol_batch}) as PNG",
                                data=buf_b,
                                file_name=f"insubria_batch_NEWonly_{pol_batch}_{model_name}.png",
                                mime="image/png",
                                key=f"dl_batch_insubria_{pol_batch}_{model_name}",
                            )

# =========================
# Training AD plot (training only)
# =========================
# with tab_training_ad:
#     st.subheader(f"Training AD plot – training data only – {model_name}")

#     pol_for_ad = st.selectbox(
#         "Select polymer for training AD plot",
#         options=AVAILABLE_POLYMERS,
#         index=0,
#         key=f"{model_name}_ad_polymer_train",
#     )

#     if st.button("Generate training AD plot (training)", key=f"{model_name}_btn_ad_train"):
#         try:
#             df_tr, h_crit = compute_training_ad_plot_data(state, pol_for_ad)
#         except Exception as e:
#             st.error(f"Cannot generate training AD plot: {e}")
#         else:
#             poly_color = poly_colors.get(pol_for_ad, "gray")

#             st.markdown(
#                 f"""
# **Training AD plot** (leverage vs predicted LogKd) for polymer **{pol_for_ad}** (training data) in **{model_name}**.

# Vertical line at **h* = 3(p+1)/n = {h_crit:.3f}** marks the applicability domain threshold.  
# Points with leverage > h* are **outside AD**.

# *(Note: this plot is training-domain visualization; strict Insubria should show only NEW compounds.)*
# """
#             )

#             fig_t, ax_t = plt.subplots(figsize=(7, 5))

#             df_in = df_tr[df_tr["AD_flag"] == "Inside AD"]
#             df_out = df_tr[df_tr["AD_flag"] == "Outside AD"]

#             if not df_in.empty:
#                 ax_t.scatter(
#                     df_in["Leverage_h"],
#                     df_in["LogKd_pred"],
#                     label="Training – inside AD",
#                     alpha=0.8,
#                     color=poly_color,
#                 )
#             if not df_out.empty:
#                 ax_t.scatter(
#                     df_out["Leverage_h"],
#                     df_out["LogKd_pred"],
#                     label="Training – outside AD",
#                     marker="s",
#                     alpha=0.9,
#                     facecolors="none",
#                     edgecolors=poly_color,
#                     linewidths=1.2,
#                 )

#             ax_t.axvline(
#                 h_crit,
#                 linestyle="--",
#                 color="gray",
#                 linewidth=1.5,
#                 label=f"h* = {h_crit:.3f}",
#             )

#             # (opcjonalnie) y-limits z exp TRAIN – jeśli chcesz też tu spójność z opisem
#             ytr_exp = df_tr["LogKd_true"].astype(float)
#             y_min_exp = float(ytr_exp.min())
#             y_max_exp = float(ytr_exp.max())
#             x_min, x_max = ax_t.get_xlim()
#             ax_t.hlines([y_min_exp, y_max_exp], x_min, x_max, linewidth=1.0, color="gray")
#             ax_t.set_xlim(x_min, x_max)

#             ax_t.set_xlabel("Leverage (h)")
#             ax_t.set_ylabel("Predicted LogKd")
#             ax_t.set_title(f"Training AD plot – training data ({pol_for_ad}) – {model_name}")
#             ax_t.legend()
#             ax_t.grid(True, alpha=0.3)

#             st.pyplot(fig_t)

#             buf_t = io.BytesIO()
#             fig_t.savefig(buf_t, format="png", dpi=300, bbox_inches="tight")
#             buf_t.seek(0)

#             st.download_button(
#                 label="Download training AD plot as PNG",
#                 data=buf_t,
#                 file_name=f"training_AD_plot_{pol_for_ad}_{model_name}.png",
#                 mime="image/png",
#                 key=f"{model_name}_dl_training_ad",
#             )

#             csv_tr = df_tr.reset_index().to_csv(index=False).encode("utf-8")
#             st.download_button(
#                 label="Download training AD data as CSV",
#                 data=csv_tr,
#                 file_name=f"training_AD_data_{pol_for_ad}_{model_name}.csv",
#                 mime="text/csv",
#                 key=f"{model_name}_dl_training_ad_csv",
#             )

#             st.write("Data used for the plot:")
#             st.dataframe(df_tr.sort_values("Leverage_h", ascending=False), use_container_width=True)


st.markdown("---")
st.caption(
    "Use the model selector in the sidebar to switch between Gaussian-based and RDKit-based models. "
    "Each model uses its own data, train/test split and (optionally) saved hyperparameters."
)
