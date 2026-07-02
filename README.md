# 📦 Executive Decision Intelligence Dashboard — 3PL D2C Logistics

A 10-page **decision-support system** for a UAE third-party-logistics (3PL) business serving
direct-to-consumer (D2C) merchants. Upload survey data → the app **auto-cleans it**, runs
descriptive → diagnostic → predictive → prescriptive analytics, and turns each page into an
answer to one executive question.

Built to be **deploy-safe on Streamlit Community Cloud**: a single `app.py`, no matplotlib,
pure-Python table styling, per-page error isolation (one failing page never blanks the app),
and optional heavy libraries that activate only if installed.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload a CSV/XLSX in the sidebar, or use the bundled `sample_data.csv` (1,178 raw records).

### Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `.streamlit/config.toml`, and (optionally) `sample_data.csv` to a GitHub repo.
2. On share.streamlit.io, point a new app at `app.py`.
3. Done — no extra configuration needed.

---

## The 10 pages (each answers one executive question)

| # | Page | Question it answers |
|---|------|---------------------|
| 1 | 🏠 Executive Overview | How healthy is the business, and where is value concentrated? |
| 2 | 👥 Customer Intelligence | Who are our merchants and how do they behave? |
| 3 | 🔬 Diagnostic Analytics | *Why* are churn, spend and satisfaction moving? |
| 4 | 🧩 Customer Segmentation | What natural customer groups exist (K-Means + LCA)? |
| 5 | 🔮 Predictive Analytics | Who will churn / adopt premium, and what drives it? |
| 6 | 📈 Regression Analytics | What drives Customer Lifetime Value, and by how much? |
| 7 | 🛒 Service Basket Analysis | Which services are bought together — what should we bundle? |
| 8 | ⏱️ Forecasting | What does the acquisition pipeline look like ahead? |
| 9 | 🎯 Recommendation Engine | What's the next-best service to offer each merchant? |
| 10 | 🧠 AI Business Advisor | Given everything, what should management do next? |

Every page includes **Business Question → Key Insight → Visual → Managerial Interpretation → Recommended Action**.

---

## Feasibility gating (the "don't force bad models" requirement)

The app statistically vets each technique against the uploaded data and **hides or adapts**
anything inappropriate, with a professional on-screen explanation. For this B2B 3PL survey:

- **Demographics → Firmographics.** No Age/Gender/Income/Occupation exist in a B2B dataset,
  so Customer Intelligence uses Industry, Company Size, Emirate and tenure instead.
- **Market Basket → Service-bundle affinity.** No product baskets exist; association-rule
  mining (custom Apriori: support / confidence / lift / conviction) runs on the 9 logistics
  service flags — the real bundling question for a 3PL.
- **Trend/Forecast is partial.** Only signup & last-order dates exist, so trends are
  *acquisition* cohorts (not transactional sales). ARIMA is enabled; **SARIMA/Prophet are
  intentionally disabled** (short synthetic series, no reliable seasonality).
- **LCA** runs as a Bernoulli mixture (pure NumPy) on the binary service flags, with the
  number of latent classes chosen by **BIC**, and is compared to K-Means.
- **LightFM** (recommender) is documented as an upgrade path only — it needs a C build that
  doesn't run reliably on Streamlit Cloud. NMF + item-item collaborative filtering are used instead.

---

## Auto data pipeline

On every upload the app removes duplicates, normalises text/categories, coerces types, parses
numbers from currency/free-text, fixes impossible values, rescales fractional rates, imputes
missing values (median/mode), engineers features (`Service_Count`, `Spend_per_Order`, tenure,
cohort month) and produces a **Data Quality Report** (completeness, missing, duplicates,
outlier summary, variable types) — shown in the Executive Overview.

---

## Optional models (auto-detected)

`XGBoost` and `SHAP` are **not required**. If you add them to `requirements.txt` they light up
automatically on the Predictive page; if absent, the app runs fine and shows a small "not
installed" badge (falling back to permutation importance). This keeps the base deploy bulletproof.

---

## Architecture note

The brief suggested a multi-folder layout (`pages/`, `components/`, …). This build ships as a
**single robust `app.py`** on purpose: multi-file uploads and hidden dependencies were the two
things that broke earlier deploys. Everything is organised into clearly-marked sections
(theme, cleaning, feasibility, analytics, pages, router) and can be split into modules later
if desired — the logic is already sectioned for it.

## Tech
Streamlit · pandas · NumPy · scikit-learn · statsmodels · SciPy · Plotly.
No matplotlib. Pure-Python gradient table styling. Light/Dark mode. Cached compute.
