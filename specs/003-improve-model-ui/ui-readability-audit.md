# UI Readability Audit: Pages 04–06

**Date**: 2026-09-19  
**Trigger**: Stakeholder reports that the report UI overlaps and important numbers cannot be read.  
**Scope**: Audit originated as specification/research. T056–T064 and T066–T067 were subsequently implemented after explicit approval; T065 browser acceptance remains open.

## Method and limitations

The audit combines:

1. direct source inspection of `src/pages/common.py`, `model_evidence.py` and Pages 04–06;
2. Streamlit AppTest through the real `streamlit.py` entrypoint and active pack `ui-79dccd400d7b8c59`;
3. Plotly protobuf inspection for traces, direct-text counts and final layout margins;
4. local Streamlit UI guidance in `developing-with-streamlit/references/{design,layouts,dashboards,data-display,best-practices}.md`;
5. Graphify traversal from `show_plot` through Pages 04–06.

Chrome DevTools MCP is not configured, so this is not a screenshot, DOM bounding-box or contrast audit. Browser acceptance remains mandatory after implementation.

## Root findings

### F1 — One global margin profile overrides every chart

`show_plot` applies `margin={l:20,r:20,t:55,b:25}` to every Plotly figure. It does so after each builder's layout, preventing charts from reserving space for long feature/model labels. No shared `automargin`, chart-specific height or dense-label policy exists.

**Likely user effect**: outside bar labels clip at the top; long x labels collide with the bottom; horizontal importance labels are compressed or cut off.

### F2 — Compact width was added without reducing label density

The visual-priority amendment correctly bounded supporting charts, but several figures still label every mark:

| View | Runtime evidence | Readability risk |
| --- | --- | --- |
| Page 04 overall | 5 categories, 10 direct labels | long candidate names plus paired labels with only 25 px bottom margin |
| Page 04 ratio/R² | 5 categories, 10 labels in 820 px | dual axes and paired labels compete vertically |
| Page 04 RF drift | 12 labels, longest 19 characters in 680 px | severe scatter-text overlap likely |
| Page 04 GB folds | 3 traces, 15 labels | two bars and a line label per fold |
| Page 05 tuning | four half-row charts, 8–12 labels each | each plot is approximately half content width; optimum is not visually dominant |
| Page 05 encoded importance | 15 horizontal categories in 680 px | left margin is only 20 px |
| Page 06 prediction | up to 10 scenarios/page by contract | prediction, actual and interval geometry can collide; whisker endpoints are not directly labelled |

### F3 — The report lacks a novice reading path

Page captions assume knowledge of `CV`, `R²`, `MedAE`, `q90`, `residual`, `frozen configuration` and `historical exposure`. The pages provide warnings and limitations but no compact metric glossary or plain-language “what question does this section answer?” guidance.

### F4 — Conclusions are incomplete

Current copy often says what evidence does **not** prove. It rarely summarizes:

- the observed winner and size of advantage;
- whether test error increased or decreased versus CV, by how much and on which metric;
- which fold/period is weakest;
- what q90 means for a typical reader;
- whether Top-2 and full evidence differ materially;
- what a scenario result can safely support.

A third party therefore has to infer the report's conclusion from multiple plots and raw tables.

### F5 — Tables do not consistently prioritize readable values

Page 04 formats the main comparison table, but Page 05 scorecard and variant tables expose technical column names and mixed precision. Wide audit tables require horizontal scanning without pinned identity columns or a summary-first projection.

### F6 — Runtime warnings add review noise

AppTest emits inherited `use_container_width` warnings from untouched shared/auth paths and an Arrow conversion warning from a mixed-type table elsewhere in the routed app. They do not explain the chart overlap, but they make a clean browser-console standard impossible until separately addressed or isolated.

## Recommended information architecture

Each page follows this order:

1. **Page brief** — one sentence naming the business question and evidence scope.
2. **KPI ribbon** — values plus short interpretation-oriented labels.
3. **How to read this page** — collapsed glossary, not a wall of prose.
4. **Primary chart** — one question, one conclusion.
5. **Conclusion card** — Finding / Why it matters / Limit / Decision-use.
6. **Supporting evidence** — compact chart(s), then formatted table/download.
7. **Methodology** — collapsed unless essential to interpreting risk.

## Chart remediation contract

- Figure builders own margins and height; shared rendering must preserve them.
- Use `automargin=True` on categorical axes.
- Use whole-dollar labels or compact `$15.7k` labels consistently; exact cents/full precision remain in downloads.
- Preserve full model/feature names in hover/table; wrap or abbreviate display ticks with an explicit mapping.
- Label all primary bars. On constrained paired charts, label only selected/important line points.
- Keep scatter text off dense points; use hover and labelled summary annotations.
- Horizontal importance charts reserve left margin based on label length and use a view-level Top N.
- Prediction charts use stable short scenario labels (`S001`, etc.); profile details remain directly above/below or in hover/table.
- Every chart declares a deterministic empty/missing state and a readable y-axis range including relevant zero/negative values.

## Required derived conclusions

### Page 04

- Candidate winner, runner-up and absolute/percentage MAE separation.
- Winner's improvement versus Dummy.
- Fold variability and worst fold/period.
- Clear statement that CV family ranking does not replace the saved-model identity.
- Baseline conclusion distinguishing observed error behavior from unproven root cause.

### Page 05

- CV-to-test delta for MAE, MedAE, RMSE and R² in correct units/directions.
- Test population and historical-exposure status.
- q90 plain-language statement and observed same-population coverage.
- Tail-risk statement from RMSE/MedAE without claiming a cause not in evidence.
- Full versus Top-2 comparison on matched rows.
- Applied tuning settings versus sweep evidence and non-nested limitation.

### Page 06

- Active two-input model and strict observed-domain statement before input.
- After prediction: number of scenarios, prediction range and q90 basis.
- Known-actual count and largest absolute variance when benchmark actuals exist.
- Explicit statement when actuals are unavailable.
- Extrapolation count/warning when experience exceptions are used.
- One safe-use conclusion: scenario comparison, not compensation guarantee.

## Implemented remediation evidence

- Shared rendering now preserves builder-specific geometry instead of overwriting margins.
- Page 04 runtime direct-label counts changed from 10→7 on the candidate chart and 12→3 on reliance drift; all Page 04 charts now declare 430–470 pixel heights and 65–105 pixel bottom margins as appropriate.
- Page 05 charts now declare 390–590 pixel heights. Tuning charts are stacked and show all MAE bars plus only the optimum R² label; horizontal importance reserves 220–250 pixel left margins.
- Pages 04–06 now include page briefs, metric glossaries, section questions and evidence-derived conclusion cards.
- Page 06 conclusions bind to the current result revision and disappear after queue mutation.
- Tables use reader-facing labels/units while downloads retain their original contracts.
- Page 04–05 compact decision tables now use one visible emphasis grammar: bold for governing/selected evidence and italics for reference/context; wide detail stays in typed interactive dataframes without raw Markdown.
- Page 04 now prioritizes a collapsed actual-method guide describing adjacent chronological row blocks, fold-local preprocessing, MAE family ranking and the non-random/non-expanding limitations.
- Page 05 now prioritizes a collapsed tuning guide describing the four anchor trials, 6/6/4/6 coordinate sweeps, R² ranking, carried/fixed values, initial-table selection provenance and non-nested limitation. Unsupported legacy tuning rationale is omitted.
- Active fold/tuning CSV downloads are byte-exact. Compatible historical primary-pipeline manifests, relevant CSVs and complete JSONL are optional exports with source/path/checksum gates and explicit run identities.
- The pre-amendment automated result was 111 passed, 1 opt-in skip before final saved-pack check; current amendment results are recorded in `verification.md`.
- Active regenerated evidence pack: `ui-b35971c9f8bcd166`; unchanged invocation returned `reused`.

## Acceptance evidence still needed

- Before/after screenshots at 1280×800 and 1440×900 for every tab/section.
- Browser DOM bounding-box check showing no text-label collisions or clipping.
- Console review and accessibility tree inspection.
- A fresh third-party walkthrough: identify each page's main finding and limitation without opening raw CSVs.
- Automated figure tests for margins, heights, label counts, number formatting and conclusion values.
