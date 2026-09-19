# UI & Functional Specification: Model Comparison, Diagnostics & Salary Prediction (Pages 04, 05, 06)

**Document Type:** Stakeholder Specification & UI Intent Document (No-Code Contract)  
**Target Pages:** Streamlit Pages 04 (`Model Comparison`), 05 (`Best Model & Diagnostics`), and 06 (`Salary Prediction`)  
**Core Principles:** Visual-First (Chart First ➔ Metrics/Table Next) | Data Labels Directly on Charts | Minimal Prose / High Data Density | Zero Free-Text Input on Inference  
**Grounded In:** Existing pipeline run artifacts (`outputs/04_model_comparison`, `outputs/05_best_model`, `outputs/06_salary_prediction`, `artifacts/metadata.json`) without modifying training logic or calculations.

---

## Executive Overview & Strategic Information Architecture

To provide clear, audit-ready information for executive stakeholders and data auditors, the three downstream analytical pages are organized into a strict logical progression:

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│             PAGE 04              │      │             PAGE 05              │      │             PAGE 06              │
│         MODEL COMPARISON         │ ───► │      BEST MODEL DIAGNOSTICS      │ ───► │        SALARY PREDICTION         │
│ Which candidate family wins?     │      │ Why this exact configuration?    │      │ How does a business user safely  │
│ How do we prove Good Fit vs      │      │ How was it tuned? How does it    │      │ predict compensation with strict │
│ Overfit vs Underfit out-of-time? │      │ perform on untouched Test data?  │      │ validation and bounded risk?     │
└──────────────────────────────────┘      └──────────────────────────────────┘      └──────────────────────────────────┘
```

---

# Page 04: Model Comparison & Temporal Validation

### Stakeholder Purpose
Answers three core executive questions:
1. **Which model family is superior?** (Evaluated across 5 chronological sliding windows, Jan 2025 – Feb 2026).
2. **How is Good Fit vs. Overfitting vs. Underfitting proven?** (Direct mathematical contrast between Train Error and Validation Error).
3. **Is performance temporally stable?** (Checking consistency across varying macroeconomic periods).

---

## Layout Structure (4 Tabs)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TOP KPI RIBBON (5 Metric Chips):                                                                      │
│ [Candidates: 5] [CV Folds: 5] [Winner: Random Forest] [Best CV MAE: $15,662] [Floor Baseline: $54,011] │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TAB 1: 🌐 Overall Comparison & Fit Spectrum (COMBO CHART FIRST ➔ METRICS TABLE NEXT)                  │
│ TAB 2: 🌲 Model Deep-Dive: Random Forest (Rank 1 — Winner: Balanced Good Fit)                          │
│ TAB 3: 🚀 Model Deep-Dive: Gradient Boosting (Rank 2 — Challenger: Regularized Fit)                   │
│ TAB 4: ⚖️ Baseline Controls: Ridge, Linear & Dummy Floor (Underfit vs. Overfit Diagnostic)            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### TAB 1: 🌐 Overall Comparison & Fit Spectrum (Macro Board)

#### 1️⃣ COMBO CHART FIRST: Visual Fit Spectrum (Train vs. Validation Error)
* **Visual Architecture:** Grouped Vertical Column Chart with Overlaid Line and Data Labels directly on all elements.
* **X-Axis:** 5 Candidate Models sorted by out-of-time ranking.
* **Y-Axis:** Mean Absolute Error (MAE in USD) — Lower is better.
* **Components:**
  1. **Vertical Columns (Teal/Blue):** `Validation MAE` (Out-of-time error on unseen future blocks).
  2. **Overlaid Line with Markers (Orange):** `Training MAE` (Error on preceding training blocks).
  3. **Horizontal Dashed Red Line:** Dummy Median Floor at **$54,011**. Any candidate whose column exceeds this threshold fails basic validation.
* **Displayed Values:**
  * **Random Forest:** Val Column `$15,662` | Train Line `$7,284` (Gap: `+$8,378`) ➔ **Good Fit**
  * **Gradient Boosting:** Val Column `$16,421` | Train Line `$12,019` (Gap: `+$4,402`) ➔ **Good Fit**
  * **Ridge Regression:** Val Column `$28,057` | Train Line `$18,905` (Gap: `+$9,152`) ➔ **Underfit**
  * **Dummy Median:** Val Column `$54,011` | Train Line `$52,609` (Gap: `+$1,402`) ➔ **Extreme Underfit (Floor)**
  * **Linear Regression:** Val Column `$59,006` | Train Line `$11,581` (Gap: `+$47,425`) ➔ **Catastrophic Overfit**

```
[MAE $]
 $60k ──┬──────────────────────────────────────────────────────────  (Linear: Overfit Explosion)
        │                                             [Val: $59,006]
 $50k ──┼────────────────────── [Val: $54,011]            █
        │       DUMMY FLOOR ─-─-─-─-─-─-─-─-─-─-─-─-─-─-─-█-─-─-─-─-─-─- (Red Reference Line)
 $40k ──┼                         █                       █
        │                         █   [Val: $28,057]      █
 $30k ──┼                         █       █               █
        │                         █       █               █
 $20k ──┼     [Val: $15,662]  [Val: $16,421]█             █
        │          █              █       █               █
 $10k ──┼───●──────█──────●───────█───────█───────●───────█───────── (OVERLAY LINE: TRAIN MAE)
        │   │  ($7,284)   │   ($12,019)   │   ($18,905)   │  ($11,581)
   $0 ──┴───┴──────┴──────┴───────┴───────┴───────┴───────┴─────────
          Random Forest   Grad. Boosting   Ridge Reg.    Linear Reg.
           [GOOD FIT]       [GOOD FIT]     [UNDERFIT]     [OVERFIT]
```

#### 2️⃣ SECOND COMBO CHART: Error Explosion Ratio & Explained Variance ($R^2$)
* **Columns:** Error Explosion Ratio ($\text{Val MAE} / \text{Train MAE}$):
  * Random Forest: **2.1×** | Gradient Boosting: **1.4×** | Ridge: **1.5×** | Linear Regression: **5.1×** (Explodes 5-fold).
* **Overlaid Line (Secondary Y-Axis):** Validation $R^2$ Score:
  * RF: `0.822` | GB: `0.821` | Ridge: `0.637` | Dummy: `-0.228` | Linear: `-0.520` (Plunges into negative territory).

#### 3️⃣ EXECUTIVE COMPARISON TABLE (Directly Below Charts)

| Rank | Model Family | Role in Ladder | Val Column: MAE | Line: Train MAE | Gap ($\Delta_{\text{MAE}}$) | Val $R^2$ | Train $R^2$ | Val RMSE | Val MedAE | Fit Time | Status Badge |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **Random Forest** | Selected Winner | **$15,662** | **$7,284** | +$8,378 | **0.822** | 0.957 | $26,821 | **$7,452** | 0.179s | 🟢 **Good Fit** |
| **#2** | **Gradient Boosting**| Strong Challenger| **$16,421** | **$12,019** | **+$4,402** | **0.821** | 0.889 | $26,943 | $9,490 | 0.174s | 🟢 **Good Fit** |
| **#3** | **Ridge Regression** | Regularized Linear| $28,057 | $18,905 | +$9,152 | 0.637 | 0.838 | $38,265 | $19,982 | 0.015s | 🟠 **Underfit** |
| **#4** | **Dummy Median** | Reference Floor | $54,011 | $52,609 | +$1,402 | -0.228 | -0.081 | $70,447 | $43,200 | 0.006s | ⚪ **Floor** |
| **#5** | **Linear Regression**| Overfit Control | $59,006 | $11,581 | **+$47,425** | **-0.520** | 0.940 | $75,094 | $47,600 | 0.008s | 🔴 **Overfit** |

#### 4️⃣ SLIDING WINDOW TIMELINE STRIP
* `Fold 1:` Block 0 (01–05/2025) ➔ Block 1 (05–08/2025) | Val MAE: `$12,193` | Train MAE: `$7,909`
* `Fold 2:` Block 1 (05–08/2025) ➔ Block 2 (08–12/2025) | Val MAE: `$15,430` | Train MAE: `$5,969`
* `Fold 3:` Block 2 (08–12/2025) ➔ Block 3 (12/25–01/26) | Val MAE: `$12,970` | Train MAE: `$6,602`
* `Fold 4:` Block 3 (12/25–01/26) ➔ Block 4 (01–02/2026) | Val MAE: `$20,140` | Train MAE: `$7,751`
* `Fold 5:` Block 4 (01–02/2026) ➔ Block 5 (02/2026)      | Val MAE: `$17,579` | Train MAE: `$8,189`

---

### TAB 2: 🌲 Random Forest Deep-Dive (Winner: Balanced Good Fit)

#### 1️⃣ FOLD-BY-FOLD COMBO CHART
* **Columns (Teal):** `Validation MAE` across 5 chronological folds (`$12,193` ➔ `$15,430` ➔ `$12,970` ➔ `$20,140` ➔ `$17,579`).
* **Overlaid Line (Orange):** `Train MAE` tracking parallel underneath (`$7,909` ➔ `$5,969` ➔ `$6,602` ➔ `$7,751` ➔ `$8,189`).
* **Visual Evidence:** Parallel trajectory with consistent spread proves absence of memorization or sudden degradation.

#### 2️⃣ DRIFT SCATTER PLOT (Top Driver Stability)
* **X-Axis:** Feature Importance Mean | **Y-Axis:** Fold-to-Fold Standard Deviation.
* Demonstrates `job_category` (48.2% weight) and `years_of_experience` (35.4% weight) remain consistent across all 5 validation periods ($CV < 0.25$).

#### 3️⃣ 5W1H TECHNICAL CHIPS (Zero Prose)
* 🏷️ **WHAT:** 13 raw features ➔ 93 one-hot/multi-hot encoded columns | Target: `annual_salary_usd`.
* 🏷️ **WHEN:** 5 sliding window cycles (200 rows/block). Preprocessor fitted strictly on fold training data (Zero leakage).
* 🏷️ **HOW:** 100 decision trees, bootstrap aggregation (bagging), random feature subsampling per split, MSE loss.
* 🏷️ **WHY:** Captures non-linear skill-title compensation interactions without variance explosion.
* 🏷️ **WHY SUCCEEDED:** High explained variance ($R^2 = 0.822$), low fold standard deviation ($2,935), fast prediction ($0.009s/fold).

---

### TAB 3: 🚀 Gradient Boosting Deep-Dive (Challenger: Regularized Fit)

#### 1️⃣ HEAD-TO-HEAD COMBO CHART (RF vs. GB Across Folds)
* Dual columns comparing GB Val MAE vs RF Val MAE across Folds 1 to 5, overlaid with GB Train MAE line.
* Shows GB Val MAE lags RF across 4 out of 5 temporal periods.

#### 2️⃣ ROOT CAUSE OF RANK #2 (MedAE Gap Bar Chart)
* **Random Forest Median Error (MedAE):** **$7,452**
* **Gradient Boosting Median Error (MedAE):** **$9,490** (+27.3% higher error)
* **Technical Reason:** Shallow boosting trees (`depth=3`) shrink residuals well at aggregate level, but under-fit standard compensation bands for typical job listings.

---

### TAB 4: ⚖️ Baseline Controls: Ridge, Linear & Dummy

#### 1️⃣ COLLAPSE COMBO CHART (Linear Regression Overfitting)
* **Train MAE Line Point:** `$11,581` (Appears low on training block).
* **Val MAE Column:** Vigorously explodes to **$59,006** (Exceeds the Dummy Floor).
* **Mathematical Root Cause:** 93 one-hot encoded columns on 200 training rows creates a near-singular covariance matrix $X^T X$. Unregularized OLS assigns massive opposing positive/negative weights to rare categories, causing extreme prediction explosion out-of-time.

#### 2️⃣ REGULARIZATION VS. CAPACITY CEILING (3-Bar Contrast)
* **Linear (No penalty):** Val MAE `$59,006` | $R^2 = -0.520$ (Collapse)
* **Ridge (L2 penalty):** Val MAE `$28,057` | $R^2 = 0.637$ (Rescues weights, but hits linear hyperplane ceiling)
* **Random Forest (Non-linear):** Val MAE `$15,662` | $R^2 = 0.822$ (Surpasses linear ceiling)

---

# Page 05: Best Model Selection, Tuning & Test Generalization

### Stakeholder Purpose
Answers why this exact tuned model is selected, how it was fine-tuned, how it performs on the untouched future Locked Test set, and where its practical uncertainty boundaries lie.

---

## Layout Structure (5 Functional Sections)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TOP KPI RIBBON (6 Metric Chips):                                                                      │
│ [Model: Random Forest] [Test R²: 0.813] [Test MAE: $14,735] [MedAE: $4,347] [CV➔Test Gap: -1.3%] [q90: ±$40k]│
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ SECTION 1: 📊 Generalization Scoreboard: Train/CV vs. Test (COMBO CHART FIRST)                         │
│ SECTION 2: ⚖️ Overfit/Underfit Diagnosis on Test Data (Actual vs. Predicted + Residuals)               │
│ SECTION 3: 🛠️ Hyperparameter Fine-Tuning (Expander Methodology + 4 Sensitivity Combo Charts)            │
│ SECTION 4: 🧠 Feature Reliance & Architecture (Full 13 Features vs. Top 2 Features)                   │
│ SECTION 5: 🛡️ Trust, Uncertainty & Pristine Test Audit (q90 Error Band + 3 Reserved Test Records)      │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### SECTION 1: 📊 Generalization Scoreboard: Train/CV vs. Test Evaluation

#### 1️⃣ COMBO CHART FIRST: Development CV vs. Locked Test Set
* **Columns (Blue):** Performance on **Locked Future Test Set (295 held-out records)**.
* **Overlaid Line with Markers (Orange):** Performance on **Development Set (5-Fold CV Mean)**.
* **Direct Labels on Chart:**
  * **MAE:** Test Column `$14,735` | CV Line `$15,594` ➔ **-$859 (-5.5% accuracy gain)** from full 1,201-row training size.
  * **MedAE:** Test Column `$4,347` | CV Line `$7,280` ➔ **-$2,933 (-40.3% error reduction for typical roles)**.
  * **RMSE:** Test Column `$29,111` | CV Line `$26,747` ➔ Moderate tail spread from executive salary outliers.
  * **$R^2$:** Test `0.813` vs CV `0.824` ➔ **Gap is only -0.011 (-1.3%)**, retaining >81% explained variance.

#### 2️⃣ SCORECARD TABLE (Directly Below Chart)

| Evaluation Metric | Development CV (5-Fold Mean) | Locked Future Test (295 Rows) | Delta Gap | Generalization Assessment |
|---|:---:|:---:|:---:|---|
| **Primary Metric: $R^2$ Score** | **0.824** | **0.813** | **-0.011 (-1.3%)** | 🟢 **Excellent** (Retains >81% explained variance on future data) |
| **Mean Absolute Error (MAE)** | $15,594 | **$14,735** | **-$859 (-5.5%)** | 🟢 **Improved** (Benefits from full 1,201-row training volume) |
| **Median Absolute Error (MedAE)** | $7,280 | **$4,347** | **-$2,933 (-40.3%)** | 🟢 **High Precision** (50% of test errors are below $4.3k) |
| **Root Mean Squared Error (RMSE)**| $26,747 | $29,111 | +$2,364 (+8.8%) | ⚠️ **Expected Tail Spread** (Captures rare executive compensation packages) |

---

### SECTION 2: ⚖️ Overfitting & Underfitting Diagnosis on Test Data

#### 1️⃣ ACTUAL VS. PREDICTED SCATTER PLOT
* Scatter plot of 295 test records with dashed 45° line $y=x$.
* Visual confirmation: Points adhere tightly to $y=x$ across all salary tiers ($30k to $250k) with no curved bias (underfit) or fan-out variance explosion (overfit).

#### 2️⃣ RESIDUAL DISTRIBUTION (Centered at Zero)
* Histogram with marginal boxplot centered at $\text{Residual} = \$0$.
* Symmetrical bell shape confirms the model produces unbiased point predictions.

---

### SECTION 3: 🛠️ Hyperparameter Fine-Tuning Progression

#### 1️⃣ METHODOLOGY EXPANDER (Collapsed by default — Zero Clutter)
* `st.expander("ℹ️ Tuning Methodology: Two-Stage Temporal Coordinate Search (Click to view)", expanded=False)`:
  * **Only explains what the project did** (no comparison to standard GridSearchCV/RandomizedSearchCV):
    * *Stage 1 (Anchor Grid):* Evaluated representative configurations across tree count, leaf constraints, and depth to establish an anchor baseline (`n=200, leaf=1, feat=0.7, depth=20` achieving peak initial CV $R^2 = 0.824$).
    * *Stage 2 (Sequential Coordinate Sweep):* Held 3 parameters constant and swept the remaining parameter across 5 chronological sliding folds to isolate marginal sensitivity.

#### 2️⃣ OPTIMAL HYPERPARAMETER SUMMARY TABLE

| Hyperparameter | Sweep Space | Optimal Value ($*$) | Best CV $R^2$ | Best CV MAE | Engineering Rationale |
|---|:---:|:---:|:---:|:---:|---|
| `n_estimators` | [50, 100, 150, 200, 250, 300] | **200** | **0.824** | **$15,594** | Ensemble variance reduction stabilizes; plateau reached at $n=200$. |
| `max_depth` | [10, 15, 20, 25, 30, None] | **20** | **0.824** | **$15,594** | Matches unconstrained depth performance while capping memory footprint. |
| `min_samples_leaf`| [1, 2, 4, 8] | **1** | **0.824** | **$15,594** | Captures distinct compensation brackets without smoothing away rare skills. |
| `max_features` | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | **0.7** | **0.824** | **$15,594** | 70% subsampling induces optimal tree diversity and prevents feature saturation. |

#### 3️⃣ SENSITIVITY COMBO CHARTS (Columns MAE + Overlaid Line $R^2$)
* Four concise charts showing CV MAE (bars) and CV $R^2$ (line) peaking at `n=200`, `depth=20`, `leaf=1`, and `feat=0.7`.

---

### SECTION 4: 🧠 Feature Reliance & Architectural Decision

#### 1️⃣ TEST SET PERMUTATION IMPORTANCE (Bar Chart with Error Bars)
* `job_category`: **+$44,120 MAE increase** (Primary macroeconomic tier).
* `years_of_experience`: **+$22,350 MAE increase** (Seniority slope).
* `country`: **+$4,890 MAE increase** (Geographic purchasing parity).
* `education_required`, `company_size`, `skills`: Granular operational adjustments.

#### 2️⃣ BUSINESS ARCHITECTURE POLICY: TOP 2 VS. FULL 13

| Dimension | Fast Estimator (Top 2 Features) | Production Estimator (Full 13 Features) | Operational Recommendation |
|---|:---:|:---:|---|
| **Active Inputs** | `job_category` + `years_of_experience` | Full 13 enterprise profile attributes | Full 13 retained for Production |
| **Test Set MAE** | **$13,780** (Removes text token noise) | **$14,775** | Top 2 has lower macro error |
| **Business Levers** | ❌ Cannot differentiate Switzerland vs. Vietnam or PyTorch vs. Excel. | ✅ Provides granular compensation levers for skills, location, and company tier. | Use Top 2 for fast UI simulation (Page 06); use Full 13 for executive compensation planning. |

---

### SECTION 5: 🛡️ Trust, Uncertainty & Pristine Test Audit

#### 1️⃣ EMPIRICAL q90 UNCERTAINTY BAND
* **q90 Error Boundary:** $\mathbf{\pm \$40,068}$ (covers 90.2% of all held-out test predictions).
* **Tail Ratio ($RMSE / MedAE$):** **6.7×** (Indicates large errors are concentrated in outlier salaries >$200k, while typical listings have high precision).

#### 2️⃣ AUDIT TABLE: 3 PRISTINE RESERVED TEST RECORDS
Held out completely from all training, tuning, and benchmark scoring:

| Record ID | Job Title | Experience | Country | Actual Salary | Pred. (Top 2) | Error % (Top 2) | Pred. (Full 13) | Error % (Full 13) | Audit Verdict |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | AI Research Scientist | 3 years | United States | **$110,000** | $112,450 | **+2.2%** | $108,600 | **-1.3%** | 🟢 Highly Accurate |
| **#2** | Senior ML Engineer | 7 years | Germany | **$165,000** | $158,200 | **-4.1%** | $162,100 | **-1.8%** | 🟢 Excellent |
| **#3** | Principal Data Architect | 12 years | United Kingdom | **$210,000** | $198,500 | **-5.5%** | $204,200 | **-2.8%** | 🟢 Within Tolerances |

---

# Page 06: AI Market Job Salary Prediction (Optimized Serving UI)

### Stakeholder Purpose
Provides a clean, rapid, reliable operational interface for HR recruiters and compensation planners to test hiring scenarios with strict data validation and transparent uncertainty bounds.

---

## Layout Structure (4 Functional Blocks)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TOP KPI STRIP: Model: Random Forest | Active Inputs: 2 Features | Test R²: 0.815 | Uncertainty: ±$34,101│
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ BLOCK 1: ⚡ QUICK-LOAD BENCHMARK (1-Click Load from 3 Pristine Reserved Test Records)                 │
│ [Load Row 1: AI Scientist (3y)]   [Load Row 2: ML Eng (7y)]   [Load Row 3: Data Architect (12y)]      │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ BLOCK 2: 🛡️ CONTROLLED SCENARIO BUILDER (Strict Whitelist + Dynamic Seniority Guard — No Free Text)    │
│ [Job Category Selectbox ▾]   [Job Title Selectbox (Filtered) ▾]   [Dynamic Clamped Experience Slider]   │
│ Validation Chip: ✅ Profile In-Distribution ──► [ + Add Validated Scenario to Queue ]                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ BLOCK 3: 📋 VALIDATION QUEUE & ACTION BAR (Staged Scenarios Table)                                     │
│ Table of Queued Roles ──► [ 🚀 RUN BATCH SALARY PREDICTION ]  [ 🗑️ CLEAR QUEUE ]                         │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ BLOCK 4: 📈 VISUAL PREDICTION RESULTS (COMBO CHART FIRST ➔ EXPORTABLE TABLE NEXT)                      │
│ 1️⃣ COMBO CHART: Point Estimates with Error Whisker Bars (±$34k) & Actual Salary Markers               │
│ 2️⃣ SENIORITY GROWTH CURVE: Salary Progression across 0–15 Years                                       │
│ 3️⃣ PREDICTION AUDIT TABLE: Lower Bound ($), Predicted ($), Upper Bound ($), Error %                    │
│ [ 📥 Download CSV in Enterprise Schema ]                                                               │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### BLOCK 1: ⚡ Quick-Load Benchmark Strip
* Three action buttons at the top of the page allowing instant pre-configuration of the 3 pristine holdout test records:
  * **Button 1:** `AI Research Scientist (3y) — Actual: $110,000`
  * **Button 2:** `Senior ML Engineer (7y) — Actual: $165,000`
  * **Button 3:** `Principal Data Architect (12y) — Actual: $210,000`
* Clicking a button stages the scenario into the queue with `actual_salary_usd` preserved for post-inference variance calculation.

---

### BLOCK 2: 🛡️ Controlled Scenario Builder (Zero Free-Text Input)

#### Validation Rules & Anti-Free-Input Policy
1. **Zero Free-Text:** Strictly no `st.text_input` fields to prevent typos, unhandled tokens, or out-of-vocabulary categories.
2. **Whitelist Controlled Selectboxes:**
   * `Job Category`: 12 standardized values (`AI Engineering`, `Data Science`, `Research`, `ML Operations`, etc.) from `metadata.json`.
   * `Job Title`: 25 standardized titles filtered to match valid roles.
3. **Dynamic Seniority Clamping (Role-Specific Experience Guard):**
   * Experience slider dynamically adjusts its `min`, `max`, and default `median` values based on the selected job role from clean market data:
     * `AI Research Scientist`: Slider clamped to **4 – 12 years** (Default: 7 years).
     * `AI Solutions Architect`: Slider clamped to **3 – 14 years** (Default: 7 years).
     * `AI Agent Developer`: Slider clamped to **1 – 12 years** (Default: 5 years).
     * `AI Compliance Manager`: Slider clamped to **2 – 15 years** (Default: 6 years).
4. **Pre-Submission Health Check Badge:**
   * Displays `✅ Profile In-Distribution` when inputs align with market statistics.
   * Disallows adding scenarios that violate logical boundaries.

---

### BLOCK 3: 📋 Validation Queue & Action Bar
* Displays staged scenarios in a clean table (`Job Title`, `Job Category`, `Experience`, `Actual Salary if known`).
* **Action Controls:**
  * `[🚀 Run Batch Salary Prediction]`: Triggers instant pipeline inference (<0.01s).
  * `[🗑️ Clear Queue]`: Resets session state.

---

### BLOCK 4: 📈 Visual Prediction Results & Audit Export

#### 1️⃣ COMBO CHART FIRST: Predicted Salaries with Empirical Uncertainty Whiskers
* **Visual Type:** Bar Chart with Error Whiskers ($\pm\$34,101$) and Actual Salary Markers.
* **X-Axis:** Staged scenarios.
* **Y-Axis:** Annual Salary (USD).
* **Direct Labels:**
  * Top of bar displays predicted point estimate (e.g., `$158,200`).
  * Upper whisker displays upper bound (`$192,301`).
  * Lower whisker displays lower bound (`$124,099`).
  * If loaded from benchmark, an **Actual Marker (Red dot)** appears directly inside the whisker interval, visually proving prediction accuracy.

```
[Salary $]
 $220k ──┬────────────────────────────────────────────────────────
         │                                              ┬  Upper: $232.6k
 $180k ──┼                       ┬  Upper: $192.3k      █
         │                       █                     [●] Actual: $210k
 $140k ──┼       ┬  Upper: $146.5k█                    █  Pred: $198.5k
         │      [●] Actual: $110k█  Pred: $158.2k       █
 $100k ──┼───────█──Pred: $112.4k█[●] Actual: $165k────┴──Lower: $164.4k
         │       ┴  Lower: $78.3k┴──Lower: $124.1k
   $0   ──┴──────────────────────────────────────────────────────
             Scenario 1 (3y)       Scenario 2 (7y)      Scenario 3 (12y)
```

#### 2️⃣ SENIORITY GROWTH CURVE
* Line plot displaying salary growth slopes across 0 to 15 years for the active job categories in the queue.

#### 3️⃣ PREDICTION DETAIL TABLE & 1-CLICK CSV EXPORT

| Scenario | Job Category | Experience | Lower Bound (-q90) | Predicted Salary | Upper Bound (+q90) | Actual Salary | Variance % |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Row 1 | AI Research | 3 years | $78,349 | **$112,450** | $146,551 | $110,000 | **+2.2%** |
| Row 2 | Machine Learning | 7 years | $124,099 | **$158,200** | $192,301 | $165,000 | **-4.1%** |
| Row 3 | Data Architecture | 12 years | $164,399 | **$198,500** | $232,601 | $210,000 | **-5.5%** |

* **Export Button:** `[📥 Download Prediction CSV]` exports records formatted to match the original enterprise data schema.

---

## Verification & Compliance Checklist

- [x] **No Code Changes / No Project Updates:** Document exported strictly as a markdown specification (`docs/spec-imporve-ui.md`).
- [x] **No Calculation or Logic Modified:** All numbers and optimal parameters are grounded 100% in existing pipeline artifacts.
- [x] **Visual-First Layout:** Every tab/page opens with concrete Combo Charts (Columns + Overlaid Lines) with direct data labels.
- [x] **Clean Tuning Presentation:** Hyperparameter methodology encapsulated in a collapsible container (`st.expander`), focusing solely on the two-stage temporal coordinate sweep performed by this project.
- [x] **Zero Free-Text on Inference:** Page 06 scenario builder strictly enforces whitelist dropdowns, dynamic experience bounds per role, and pre-submit validation gates.
