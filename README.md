# 🔬 POPA (Polymer–Organic Pollutant Adsorption)

**Predict the adsorption capacity (LogKd) of organic pollutants onto microplastics.**

POPA is a simple Streamlit app that predicts **LogKd** — the base-10 logarithm of the
sorption (distribution) coefficient *Kd* — for three common microplastics:
**polyethylene (PE)**, **polypropylene (PP)** and **polystyrene (PS)**.
A higher LogKd means a stronger adsorption affinity of the compound for that polymer.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://popa-mp.streamlit.app)

---

## ✨ Features

- Two alternative prediction models (QM-based and RDKit-based).
- **Single** prediction (one compound) and **batch** prediction (CSV/Excel upload).
- Prediction **directly from SMILES** (RDKit model only).
- **Applicability-domain (AD)** check for every prediction, based on leverage.
- Downloadable results (CSV).

---

## 🧠 The two models

Both models are multi-output GradientBoosting regressors (one model per polymer) and use the same
descriptors — they differ only in **how the descriptors are obtained**:

| Model | How descriptors are computed |
|-------|------------------------------|
| ⚛️ **Quantum-mechanical (QM)** | Descriptors (π, M′) computed at the quantum-mechanical level with **Gaussian 09**. |
| 🧬 **RDKit-based** | Descriptors (M′, π) computed automatically with **RDKit**, so you can predict straight from a SMILES string. |

---

## 🧪 What you need

For every organic compound the models use three physicochemical descriptors:

| Descriptor | Meaning |
|-----------|---------|
| **logD** | *n*-octanol/water distribution coefficient at a given pH (lipophilicity, accounting for ionization). |
| **M′** | Molecular mass of the compound (used in a scaled form). |
| **π** | Ratio of molecular polarizability to molecular volume: **π = α / V′** (*α* = polarizability, *V′* = molecular volume). |

> **Note:** In the RDKit model, **M′** and **π** are computed automatically from the SMILES string,
> so you only need to provide **logD** manually. In the QM model you provide all descriptor values.

---

## 🚀 How to run it

### Option 1 — Use the hosted app (no install)

Just click the badge above, or open:
👉 **https://popa-mp.streamlit.app**

### Option 2 — Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/popa.git
cd popa

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app expects the model data files in a `data/` folder (Excel files with the training data,
train/test split and saved hyperparameters). Make sure that folder is present before running.

---

## 📖 How to use the app

1. **Choose the polymers** in the sidebar (PE, PP and/or PS).
2. **Open a model tab** at the top: **⚛️ QM model** or **🧬 RDKit model**.
3. **Pick a prediction mode** (sub-tab):

   | Mode | What it does |
   |------|--------------|
   | 🔹 **Single prediction** | Type descriptor values for one compound and get its LogKd per polymer. |
   | 📦 **Batch prediction** | Upload a CSV/Excel with descriptor columns to predict many compounds at once. |
   | 🧪 **SMILES input** *(RDKit only)* | Paste a SMILES string — M′ and π are computed automatically; you add logD. |

4. **Read the results.** Each polymer shows the predicted **LogKd** and whether the compound is
   **inside / outside the applicability domain** (see below).
5. For batch jobs, **download the full results** as a CSV.

### 📄 Batch file format

Your CSV/Excel must contain one column per descriptor used by the selected model
(for example: `logD`, `M`, `π`). One row = one compound. The app tells you if a required
column is missing.

---

## 🎯 Applicability domain (AD)

Every prediction reports whether the compound lies **inside the applicability domain** of the
model, using the **leverage** *h* compared with a warning threshold **h\* = 3(p + 1) / n**
(*p* = number of descriptors, *n* = number of training compounds).
Predictions for compounds **outside the AD** are extrapolations and should be treated with caution.

---

## 📬 Contact & feedback

Found a bug, have a question, or want to share feedback? Get in touch via the **Contact & feedback**
section in the app sidebar, or open an issue in this repository.

---

*MP-AdsorbNet · QM descriptors via Gaussian 09 · RDKit descriptors via RDKit.*
