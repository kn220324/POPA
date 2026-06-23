import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


# =========================
# Global parameters (wspólne)
# =========================
EXPECTED_POLYMERS = ["PE", "PP", "PS"]
RANDOM_STATE = 42
Q2_TEST_METHOD = "F1"  # "F1", "F2", "F3"

# --- Colors for polymers ---
poly_colors = {
    "PE": "#1f77b4",
    "PP": "#e85d4e",
    "PS": "#2e8b57",
}

# --- Full polymer names (display only; internal data still uses PE/PP/PS) ---
POLYMER_NAMES = {
    "PE": "Polyethylene",
    "PP": "Polypropylene",
    "PS": "Polystyrene",
}

# --- Contact / feedback (EDIT THESE) ---
CONTACT_EMAIL = "kinga.nimz@pg.edu.pl"        # <-- podmień na swój adres
ISSUES_URL = ""                                  # <-- opcjonalnie: link do repo/Issues (np. GitHub), zostaw "" jeśli brak

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
    "Quantum mechanical (QM) descriptors": {
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
# AD basis (leverage / applicability domain) used in predictions
# =========================
def compute_ad_basis_for_polymer(state: ModelState, polymer: str):
    est, selected_feats, XtX_inv, h_crit, Xtr, ytr = get_ad_basis_for_polymer(state, polymer)

    return {
        "est": est,
        "selected_feats": selected_feats,
        "XtX_inv": XtX_inv,
        "h_crit": float(h_crit),
        "train_index": set(Xtr.index.astype(str)),
        "test_index": set(state.X_test_all.index.astype(str)),
    }


def leverage_for_new(X_new: pd.DataFrame, XtX_inv: np.ndarray) -> np.ndarray:
    X_design_new = np.column_stack([np.ones(len(X_new)), X_new.values.astype(float)])
    AX_new = X_design_new @ XtX_inv
    h_new = np.sum(AX_new * X_design_new, axis=1)
    return h_new


# =========================
# Predictions
# =========================
def predict_single_compound(state: ModelState, descriptor_dict: dict, polymers: list[str], compound_name: str = "compound") -> pd.DataFrame:
    X_new = pd.DataFrame([descriptor_dict], index=[compound_name])
    rows = []

    for pol in polymers:
        try:
            basis = compute_ad_basis_for_polymer(state, pol)
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
        basis = compute_ad_basis_for_polymer(state, pol)
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
# UI helpers
# =========================
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ---- typography ---- */
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

        /* ---- main container ---- */
        .block-container { padding-top: 1.6rem; max-width: 1150px; }

        /* ---- hero header ---- */
        .app-hero {
            background: linear-gradient(135deg, #0f766e 0%, #155e75 100%);
            padding: 1.6rem 2rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 32px rgba(15, 118, 110, 0.28);
        }
        .app-hero h1 { color: #ffffff; margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; }
        .app-hero p { color: rgba(255, 255, 255, 0.88); margin: 0.45rem 0 0; font-size: 0.98rem; }

        /* ---- buttons ---- */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            background: #0f766e;
            color: #ffffff;
            padding: 0.5rem 1.3rem;
            transition: all 0.15s ease;
        }
        .stButton > button:hover { background: #0d655d; transform: translateY(-1px); }
        .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid #0f766e;
            background: transparent;
            color: #0f766e;
        }
        .stDownloadButton > button:hover { background: #0f766e; color: #ffffff; }

        /* ---- tabs ---- */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.4rem 1rem;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] { background: rgba(15, 118, 110, 0.10); }

        /* ---- polymer badge ---- */
        .poly-badge {
            color: #ffffff;
            padding: 0.4rem 0.8rem;
            border-radius: 9px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.55rem;
            letter-spacing: 0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_cards(df_res: pd.DataFrame):
    """Renderuje wynik pojedynczej predykcji jako kolumny z metrykami i statusem AD."""
    if df_res.empty:
        st.info("No predictions to show.")
        return

    cols = st.columns(len(df_res))
    for col, (_, row) in zip(cols, df_res.iterrows()):
        with col:
            pol = str(row["Polymer"])
            color = poly_colors.get(pol, "#64748b")
            display_name = POLYMER_NAMES.get(pol, pol)
            st.markdown(
                f"<div class='poly-badge' style='background:{color};'>{display_name}</div>",
                unsafe_allow_html=True,
            )

            val = row["LogKd_pred"]
            flag = str(row["AD_flag"])

            if pd.isna(val):
                st.error("Prediction failed")
                st.caption(flag)
                continue

            st.metric("Predicted LogKd", f"{val:.3f}")
            if pd.notna(row["Leverage_h"]) and pd.notna(row["h_crit"]):
                st.caption(f"leverage h = {row['Leverage_h']:.3f}  ·  h* = {row['h_crit']:.3f}")

            if flag == "Inside AD":
                st.success("Inside applicability domain")
            elif flag.startswith("Error"):
                st.error(flag)
            else:
                st.warning("Outside applicability domain")


# --- Short descriptor definitions (used for tooltips and the guide) ---
DESCRIPTOR_INFO = {
    "logD": "n-octanol/water distribution coefficient at a given pH (lipophilicity, accounting for ionization).",
    "M": "M′ — molecular mass of the compound (used in a scaled form).",
    "π": "Polarizability-to-molecular-volume ratio: π = α / V′ (α = polarizability, V′ = molecular volume).",
    "V'": "V′ — molecular volume of the compound.",
}


def descriptor_help(feat: str):
    return DESCRIPTOR_INFO.get(feat)


# =========================
# Prediction tabs (reusable per model)
# =========================
def render_single_prediction(state: ModelState, model_name: str, polymers: list[str]):
    st.subheader("Single compound prediction")

    if not polymers:
        st.warning("Select at least one polymer in the sidebar to run predictions.")
        return

    feature_columns = list(state.feature_columns)

    compound_name = st.text_input(
        "Compound ID / name", value="your_compound", key=f"{model_name}_single_name"
    )

    st.write("Provide descriptor values (same scale as the training data for this model).")

    descriptor_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_columns):
        with cols[i % 3]:
            descriptor_values[feat] = st.number_input(
                label=f"{feat}",
                value=0.0,
                format="%.6f",
                help=descriptor_help(feat),
                key=f"{model_name}_single_{feat}",
            )

    if st.button("Predict LogKd for this compound", key=f"{model_name}_btn_single"):
        with st.spinner("Running predictions..."):
            df_res = predict_single_compound(state, descriptor_values, polymers, compound_name)

        st.markdown("#### Predictions with applicability-domain info")
        render_prediction_cards(df_res)

        with st.expander("Show full results table"):
            df_show = df_res.copy()
            df_show["Polymer"] = df_show["Polymer"].map(lambda p: POLYMER_NAMES.get(str(p), p))
            st.dataframe(df_show, use_container_width=True)


def render_batch_prediction(state: ModelState, model_name: str, polymers: list[str]):
    st.subheader("Batch prediction from file")

    if not polymers:
        st.warning("Select at least one polymer in the sidebar to run predictions.")
        return

    feature_columns = list(state.feature_columns)

    st.write(
        "Upload a **CSV or Excel** file containing at least the following descriptor columns:"
    )
    st.code(", ".join(feature_columns), language="text")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx"], key=f"{model_name}_upload"
    )

    if uploaded_file is None:
        return

    if uploaded_file.name.lower().endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df_input.head(), use_container_width=True)

    missing_cols = [c for c in feature_columns if c not in df_input.columns]
    if missing_cols:
        st.error(f"Missing required feature columns: {missing_cols}")
        return

    if st.button("Run batch prediction", key=f"{model_name}_btn_batch"):
        with st.spinner("Running batch predictions..."):
            try:
                df_pred = predict_batch(state, df_input, polymers)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return

        st.markdown("#### Applicability-domain summary")
        summary_cols = st.columns(len(polymers))
        for s_col, pol in zip(summary_cols, polymers):
            flag_col = f"AD_flag_{pol}"
            with s_col:
                color = poly_colors.get(pol, "#64748b")
                st.markdown(
                    f"<div class='poly-badge' style='background:{color};'>{POLYMER_NAMES.get(pol, pol)}</div>",
                    unsafe_allow_html=True,
                )
                if flag_col in df_pred.columns:
                    inside = int((df_pred[flag_col] == "Inside AD").sum())
                    total = int(df_pred[flag_col].notna().sum())
                    st.metric("Inside AD", f"{inside} / {total}")
                else:
                    st.caption("n/a")

        st.markdown("#### Predictions (first rows)")
        st.dataframe(df_pred.head(), use_container_width=True)

        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full predictions as CSV",
            data=csv_bytes,
            file_name=f"logKd_predictions_with_AD_{model_name}.csv",
            mime="text/csv",
            key=f"{model_name}_dl_batch",
        )


def render_smiles_prediction(state: ModelState, model_name: str, polymers: list[str]):
    st.subheader("Prediction directly from SMILES (RDKit descriptors)")

    if not polymers:
        st.warning("Select at least one polymer in the sidebar to run predictions.")
        return

    st.write(
        "Descriptors computed automatically from SMILES: **M′** and **π**. "
        "The model also needs **logD**, which you provide manually."
    )

    mode = st.radio(
        "Mode",
        ["Single SMILES", "Batch (file upload)"],
        horizontal=True,
        key=f"{model_name}_smiles_mode",
    )

    # ---------- SINGLE ----------
    if mode == "Single SMILES":
        smiles = st.text_input("SMILES string", key=f"{model_name}_smiles_str")
        compound_name = st.text_input(
            "Compound name", "smiles_compound", key=f"{model_name}_smiles_name"
        )
        logD_input = st.number_input(
            "logD (if required by model)",
            value=0.0,
            format="%.6f",
            help=descriptor_help("logD"),
            key=f"{model_name}_smiles_logD",
        )

        if st.button("Predict from SMILES", key=f"{model_name}_btn_smiles_single"):
            if smiles.strip() == "":
                st.warning("Provide SMILES.")
            else:
                try:
                    desc = rdkit_descriptors_from_smiles(smiles)
                    desc["logD"] = logD_input

                    df_res = predict_single_compound(state, desc, polymers, compound_name)

                    st.markdown("#### Predictions")
                    render_prediction_cards(df_res)

                    with st.expander("Show full results table"):
                        df_show = df_res.copy()
                        df_show["Polymer"] = df_show["Polymer"].map(lambda p: POLYMER_NAMES.get(str(p), p))
                        st.dataframe(df_show, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ---------- BATCH ----------
    else:
        uploaded = st.file_uploader(
            "Upload CSV/Excel containing SMILES column",
            type=["csv", "xlsx"],
            key=f"{model_name}_smiles_upload",
        )
        smiles_col = st.text_input(
            "SMILES column name", value="SMILES", key=f"{model_name}_smiles_col"
        )

        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)

            st.write("Uploaded data preview:")
            st.dataframe(df_in.head(), use_container_width=True)

            if smiles_col not in df_in.columns:
                st.error("SMILES column not found.")
            else:
                if st.button("Run batch SMILES prediction", key=f"{model_name}_btn_smiles_batch"):
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
                    df_pred = predict_batch(state, df_desc, polymers)

                    st.write("Predictions (first rows):")
                    st.dataframe(df_pred.head(), use_container_width=True)

                    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv_bytes,
                        file_name="smiles_predictions.csv",
                        mime="text/csv",
                        key=f"{model_name}_dl_smiles_batch",
                    )


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="MP-AdsorbNet",
    page_icon="🔬",
    layout="wide",
)

inject_custom_css()

st.markdown(
    """
    <div class="app-hero">
        <h1>🔬 MP-AdsorbNet</h1>
        <p>Predicting <b>LogKd</b> (log₁₀ microplastic/water partition coefficient) for adsorption of organic
        pollutants onto three common microplastics: polyethylene (PE), polypropylene (PP) and polystyrene (PS),
        using multi-output GradientBoosting models with applicability-domain (leverage) checks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Sidebar
# =========================
st.sidebar.markdown("## ⚙️ MP-AdsorbNet")
st.sidebar.caption("Microplastic adsorption (LogKd) predictor")
st.sidebar.markdown("---")
st.sidebar.subheader("Prediction settings")

selected_polymers = st.sidebar.multiselect(
    "Select polymers for prediction",
    options=EXPECTED_POLYMERS,
    default=EXPECTED_POLYMERS,
    format_func=lambda p: f"{POLYMER_NAMES.get(p, p)} ({p})",
)
if not selected_polymers:
    st.sidebar.warning("Select at least one polymer to enable predictions.")

st.sidebar.markdown("---")
st.sidebar.subheader("📬 Contact & feedback")
st.sidebar.markdown(
    "Found a bug, have a question, or want to share feedback? Get in touch:"
)
st.sidebar.markdown(
    f"- ✉️ **Email:** [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}"
    f"?subject=MP-AdsorbNet%20feedback)"
)
if ISSUES_URL:
    st.sidebar.markdown(f"- 🐞 **Report an issue:** [open a ticket]({ISSUES_URL})")
st.sidebar.caption("Your feedback helps improve the models and the app.")

# =========================
# Top-level navigation: Home + one tab per model
# =========================
tab_home, tab_qm, tab_rdkit = st.tabs(
    ["🏠 Home & guide", "⚛️ QM model", "🧬 RDKit model"]
)

# ---------------- HOME ----------------
with tab_home:
    st.header("About this app")
    st.markdown(
        """
**MP-AdsorbNet** predicts **LogKd** (logarithm of the microplastic/water partition coefficient) *Kd* for the **adsorption of organic pollutants onto microplastics**. *Kd* (typical units **L/kg**) describes how strongly a compound partitions from water onto a
microplastic particle; a **higher LogKd means a stronger adsorption affinity** for that polymer.
"""
    )

    st.subheader("Polymers covered")
    st.markdown(
        """
| Code | Polymer |
|------|---------|
| **PE** | Polyethylene |
| **PP** | Polypropylene |
| **PS** | Polystyrene |
"""
    )

    st.subheader("Molecular descriptors")
    st.markdown(
        """
The models use a small set of physicochemical descriptors of the organic compound:

- **logD** — *n*-octanol/water distribution coefficient at a given pH. A lipophilicity measure
  that, unlike logP, accounts for ionization of the compound at that pH.
- **M′** — molecular mass of the compound (used in a scaled form).
- **π** — polarizability-to-molecular-volume ratio, defined as **π = α / V′**, where *α* is the
  molecular polarizability and *V′* the molecular volume. It reflects the electronic
  polarizability density of the molecule.
"""
    )

    st.subheader("The two models")
    st.markdown(
        """
Both models are **multi-output GradientBoosting regressors** (one model per polymer) and share the
same descriptors, but differ in how those descriptors are obtained:

- **⚛️ QM model** — descriptors computed at the **quantum-mechanical level using *Gaussian 09***
  (e.g. the polarizability *α* comes from the QM calculation).
- **🧬 RDKit model** — descriptors computed with **RDKit**, which also allows prediction
  **directly from a SMILES string**. Here *M′* = MW/100, *V′* = LabuteASA/100, and
  *π* = MolMR / *V′* (molar refractivity used as a polarizability surrogate); logD is still
  provided by the user.

Open the **⚛️ QM model** or **🧬 RDKit model** tab above to make predictions.
"""
    )

    st.subheader("Applicability domain (AD)")
    st.markdown(
        r"""
Every prediction reports whether the compound falls **inside the applicability domain** of the
model, based on the **leverage** *h* compared with a warning threshold *h**. Predictions for compounds
**outside the AD** are extrapolations and should be treated with caution.
"""
    )

    st.subheader("How to use")
    st.markdown(
        """
1. In the **sidebar**, choose which **polymers** to predict for.
2. Open a model tab: **⚛️ QM model** or **🧬 RDKit model**.
3. Pick a sub-tab:
   - **🔹 Single prediction** — type descriptor values for one compound.
   - **📦 Batch prediction** — upload a CSV/Excel with the descriptor columns.
   - **🧪 SMILES input** (RDKit only) — paste a SMILES; M′ and π are computed automatically.
"""
    )

# ---------------- QM MODEL ----------------
with tab_qm:
    qm_name = "Quantum mechanical (QM) descriptors"
    st.header("⚛️ Quantum mechanical (QM) descriptors model")
    st.caption("Descriptors computed at the quantum-mechanical level using Gaussian 09.")

    try:
        with st.spinner("Loading QM model…"):
            qm_state = load_model_state(qm_name)
    except Exception as e:
        st.error(f"Could not load the QM model: {e}")
    else:
        qm_polymers = [p for p in selected_polymers if p in qm_state.Y_wide.columns]
        sub_single, sub_batch = st.tabs(["🔹 Single prediction", "📦 Batch prediction"])
        with sub_single:
            render_single_prediction(qm_state, qm_name, qm_polymers)
        with sub_batch:
            render_batch_prediction(qm_state, qm_name, qm_polymers)

# ---------------- RDKIT MODEL ----------------
with tab_rdkit:
    rk_name = "RDKit descriptors"
    st.header("🧬 RDKit descriptors model")
    st.caption("Descriptors computed automatically with RDKit — including directly from SMILES.")

    try:
        with st.spinner("Loading RDKit model…"):
            rk_state = load_model_state(rk_name)
    except Exception as e:
        st.error(f"Could not load the RDKit model: {e}")
    else:
        rk_polymers = [p for p in selected_polymers if p in rk_state.Y_wide.columns]
        sub_smiles, sub_single, sub_batch = st.tabs(
            ["🧪 SMILES input", "🔹 Single prediction", "📦 Batch prediction"]
        )
        with sub_smiles:
            render_smiles_prediction(rk_state, rk_name, rk_polymers)
        with sub_single:
            render_single_prediction(rk_state, rk_name, rk_polymers)
        with sub_batch:
            render_batch_prediction(rk_state, rk_name, rk_polymers)

st.markdown("---")
st.caption(
    "MP-AdsorbNet · QM descriptors via Gaussian 09 · RDKit descriptors via RDKit. "
    "Each model uses its own data, train/test split and (optionally) saved hyperparameters."
)