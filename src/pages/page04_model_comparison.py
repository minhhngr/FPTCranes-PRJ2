from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .common import *


def render(st, root, role="admin"):
    style_page(st)
    st.title("4. Model Comparison")
    st.caption("Branch B — B4 Model Training & Sliding Window Temporal Cross-Validation | 200-row blocks + baseline ladder + runtime + family ablation + feature-reliance stability")
    comp=read_csv(root,"04_model_comparison/model_comparison.csv")
    folds=read_csv(root,"04_model_comparison/cv_fold_metrics.csv")
    fam=read_csv(root,"04_model_comparison/09_feature_family_ablation.csv") if (root/"outputs/04_model_comparison/09_feature_family_ablation.csv").exists() else pd.DataFrame()
    drift=read_csv(root,"04_model_comparison/09_feature_importance_drift.csv") if (root/"outputs/04_model_comparison/09_feature_importance_drift.csv").exists() else pd.DataFrame()
    imp_fold=read_csv(root,"04_model_comparison/09_feature_importance_by_fold.csv") if (root/"outputs/04_model_comparison/09_feature_importance_by_fold.csv").exists() else pd.DataFrame()

    best=comp.sort_values("MAE_mean").iloc[0]
    metric_cols(st,[("Candidate models",len(comp)),("Temporal folds",int(best.folds)),("Selected family",best.model),("Best mean CV MAE",money(best.MAE_mean)),("Best mean CV R²",f"{best.R2_mean:.3f}")])
    st.success("All candidate models use identical 200-row sliding window temporal folds and TRAIN-only preprocessing. The future locked period remains unopened during family comparison and tuning.")

    with st.expander("Interactive model filters",expanded=True):
        c1,c2,c3=st.columns(3)
        model_opts=comp.model.tolist()
        models=c1.multiselect("Models",model_opts,default=model_opts,key="p4_models")
        metric=c2.selectbox("Comparison metric",["MAE_mean","RMSE_mean","R2_mean","MedAE_mean","fit_time_mean_s","predict_time_mean_s"],key="p4_metric")
        fold_metric=c3.selectbox("Fold trend metric",["MAE","RMSE","R2","MedAE","fit_time_s","predict_time_s"],key="p4_fold_metric")
    if not models:
        st.warning("Select at least one model.")
        return
    c=comp[comp.model.isin(models)].copy(); f=folds[folds.model.isin(models)].copy()

    with st.expander("📌 Temporal Fold Partition Scheme: Chronological Order + 200-row Blocks + Sliding Window", expanded=True):
        fold_spec = pd.DataFrame([
            {"Fold": 1, "Train Block": "Block 0 (rows 0–199)", "Train Period": "2025-01 .. 2025-05", "Train Size": 200, "Validation Block": "Block 1 (rows 200–399)", "Validation Period": "2025-05 .. 2025-08", "Val Size": 200, "Mechanism": "Sliding Window"},
            {"Fold": 2, "Train Block": "Block 1 (rows 200–399)", "Train Period": "2025-05 .. 2025-08", "Train Size": 200, "Validation Block": "Block 2 (rows 400–599)", "Validation Period": "2025-08 .. 2025-12", "Val Size": 200, "Mechanism": "Sliding Window"},
            {"Fold": 3, "Train Block": "Block 2 (rows 400–599)", "Train Period": "2025-08 .. 2025-12", "Train Size": 200, "Validation Block": "Block 3 (rows 600–799)", "Validation Period": "2025-12 .. 2026-01", "Val Size": 200, "Mechanism": "Sliding Window"},
            {"Fold": 4, "Train Block": "Block 3 (rows 600–799)", "Train Period": "2025-12 .. 2026-01", "Train Size": 200, "Validation Block": "Block 4 (rows 800–999)", "Validation Period": "2026-01 .. 2026-02", "Val Size": 200, "Mechanism": "Sliding Window"},
            {"Fold": 5, "Train Block": "Block 4 (rows 800–999)", "Train Period": "2026-01 .. 2026-02", "Train Size": 200, "Validation Block": "Block 5 (rows 1000–1200)", "Validation Period": "2026-02", "Val Size": 201, "Mechanism": "Sliding Window"},
        ])
        st.dataframe(fold_spec, use_container_width=True, hide_index=True)
        st.caption("Principle: Development set (1,201 rows) is sorted chronologically by posting_year and posting_month, then partitioned into 200-row blocks. Each fold uses the preceding 200-row block as Train and the immediate subsequent 200-row block as Validation.")

    tabs=st.tabs(["Model Ranking","Temporal Fold Stability","Runtime","Feature-family Ablation","Feature Importance Drift","Candidate Ladder","Evidence Tables"])

    with tabs[0]:
        asc=metric!="R2_mean"
        d=c.sort_values(metric,ascending=asc)
        fig=px.bar(d.sort_values(metric,ascending=not asc),y="model",x=metric,orientation="h",text=metric,title=f"Candidate Comparison — {metric}")
        if metric in {"MAE_mean","RMSE_mean","MedAE_mean"}: fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
        elif "time" in metric: fig.update_traces(texttemplate="%{text:.4f}s",textposition="outside")
        else: fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
        show_plot(st,fig,"p4_rank")
        runner=comp.sort_values("MAE_mean").iloc[1] if len(comp)>1 else best
        dummy=comp.loc[comp.model=="Dummy Median","MAE_mean"]
        dummy_gain=100*(float(dummy.iloc[0])-float(best.MAE_mean))/float(dummy.iloc[0]) if len(dummy) else np.nan
        gap=100*(float(runner.MAE_mean)-float(best.MAE_mean))/float(runner.MAE_mean) if float(runner.MAE_mean) else 0
        interpretation_card(st,f"{best.model} ranks first with mean temporal-CV MAE {money(best.MAE_mean)} and R² {best.R2_mean:.3f}; it improves MAE by {gap:.1f}% versus {runner.model}"+(f" and {dummy_gain:.1f}% versus Dummy Median." if np.isfinite(dummy_gain) else "."),"The selected family wins on out-of-time accuracy under the same preprocessing and folds, so the comparison isolates model-family behavior rather than split differences.","Advance only the selected family to bounded tuning; keep the runner-up as a stability benchmark.","success")

    with tabs[1]:
        fig=px.line(f.sort_values(["model","fold"]),x="validation_period",y=fold_metric,color="model",markers=True,text=fold_metric,title=f"{fold_metric} Across Sliding Window Temporal Folds")
        fig.update_traces(textposition="top center"); show_plot(st,fig,"p4_fold")
        heat=f.pivot(index="model",columns="validation_period",values=fold_metric)
        fig=px.imshow(heat,text_auto=".2f",aspect="auto",title=f"Fold-by-Fold Heatmap — {fold_metric}",labels=dict(x="Validation period",y="Model",color=fold_metric)); show_plot(st,fig,"p4_heat")
        stab=f.groupby("model").agg(mean=(fold_metric,"mean"),std=(fold_metric,"std"),worst=(fold_metric,"max" if fold_metric!="R2" else "min")).reset_index()
        b=stab.loc[stab.model==best.model].iloc[0]
        interpretation_card(st,f"For {best.model}, {fold_metric} mean is {b['mean']:.3f} with fold-to-fold SD {0 if pd.isna(b['std']) else b['std']:.3f}.","A low average error is more credible when performance is not driven by a single fold. Sliding 200-row windows preserve chronology and expose temporal sensitivity without compounding historical volume.","Inspect any validation period where the selected model degrades sharply before trusting the aggregate average.","info")

    with tabs[2]:
        rt=c.melt(id_vars=["model"],value_vars=["fit_time_mean_s","predict_time_mean_s"],var_name="runtime",value_name="seconds")
        fig=px.bar(rt,x="seconds",y="model",color="runtime",orientation="h",barmode="stack",text="seconds",title="Mean Fit & Predict Time per Temporal Fold")
        fig.update_traces(texttemplate="%{text:.4f}s",textposition="inside"); show_plot(st,fig,"p4_runtime")
        slow=comp.sort_values("fit_time_mean_s",ascending=False).iloc[0]
        interpretation_card(st,f"Slowest mean fit time is {slow.model} at {slow.fit_time_mean_s:.3f}s/fold; selected {best.model} predicts in {best.predict_time_mean_s:.4f}s/fold on this dataset.","At this scale all serious candidates are operationally lightweight; temporal accuracy, not runtime, should drive model selection.","Keep runtime evidence for deployment sizing, but do not trade away substantial out-of-time accuracy for millisecond-level gains here.","info")

    with tabs[3]:
        st.markdown("### Branch-B Feature-family Ablation (from FPTCranes-PRJ2-main design)")
        if fam.empty:
            st.info("Family ablation artifact is unavailable; re-run pipeline.py.")
        else:
            metric2=st.selectbox("Ablation metric",["MAE_mean","RMSE_mean","R2_mean","improvement_vs_A0_pct"],key="p4_fam_metric")
            asc=metric2!="R2_mean" and metric2!="improvement_vs_A0_pct"
            d=fam.sort_values(metric2,ascending=asc)
            fig=px.bar(d.sort_values(metric2,ascending=not asc),y="experiment",x=metric2,orientation="h",text=metric2,hover_data=["feature_count","features"],title=f"Feature-family Ablation — {metric2}")
            if metric2 in {"MAE_mean","RMSE_mean"}: fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
            elif metric2=="improvement_vs_A0_pct": fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
            else: fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
            show_plot(st,fig,"p4_family_ablation")
            best_ab=fam.sort_values("MAE_mean").iloc[0]
            base=fam.loc[fam.experiment=="A0_CONSERVATIVE_CORE"].iloc[0] if (fam.experiment=="A0_CONSERVATIVE_CORE").any() else fam.iloc[-1]
            interpretation_card(st,f"Best family experiment is {best_ab.experiment} at {money(best_ab.MAE_mean)} mean MAE, {best_ab.improvement_vs_A0_pct:.1f}% better than A0.","This isolates the incremental contribution of years, bucketed experience and skills under the same temporal-validation design.","Keep a feature family for accuracy only when the gain is material and its semantics are defensible; otherwise treat it as optional scenario detail.","warning")
            st.dataframe(fam.style.format({"MAE_mean":"${:,.0f}","MAE_std":"${:,.0f}","RMSE_mean":"${:,.0f}","R2_mean":"{:.3f}","improvement_vs_A0_pct":"{:.1f}%"}),use_container_width=True,hide_index=True)

    with tabs[4]:
        st.markdown("### Feature-reliance Stability Across Temporal Folds")
        if drift.empty or imp_fold.empty:
            st.info("Feature-importance drift evidence is unavailable; re-run pipeline.py.")
        else:
            topn=st.slider("Top encoded features by mean importance",5,40,20,key="p4_drift_topn")
            top=drift.nlargest(topn,"importance_mean")
            fig=px.scatter(top,x="importance_mean",y="importance_std",size="importance_mean",hover_name="encoded_feature",color="importance_cv",title="Importance Mean vs Temporal Drift",labels={"importance_mean":"Mean importance","importance_std":"Fold SD"})
            show_plot(st,fig,"p4_drift_scatter")
            focus=st.selectbox("Inspect feature across folds",top.encoded_feature.tolist(),key="p4_drift_feature")
            fd=imp_fold[imp_fold.encoded_feature==focus].sort_values("fold")
            fig=px.line(fd,x="validation_period",y="importance",markers=True,text="importance",title=f"{focus} — Importance Across Folds")
            fig.update_traces(texttemplate="%{text:.3f}",textposition="top center"); show_plot(st,fig,"p4_drift_line")
            row=drift.loc[drift.encoded_feature==focus].iloc[0]
            tone="warning" if row.importance_cv>0.5 else "info"
            interpretation_card(st,f"{focus}: mean importance {row.importance_mean:.3f}, fold SD {row.importance_std:.3f}, coefficient of variation {row.importance_cv:.2f}.","Large variation means the model's reliance on this encoded signal changes across validation months even if aggregate accuracy is good.","Treat unstable importance as a monitoring target and avoid causal interpretation.",tone)

    with tabs[5]:
        ladder=pd.DataFrame([
            ["Dummy Median","Non-ML floor","Every promoted model must materially outperform a training-fold median predictor."],
            ["Linear Regression","Transparent linear baseline","Tests whether salary is explained by a simple additive relationship."],
            ["Ridge Regression","Regularized linear baseline","Uses L2 regularization after one-hot expansion."],
            ["Random Forest","Nonlinear bagging candidate","Captures nonlinear interactions in mixed tabular data and supports importance review."],
            ["Gradient Boosting","Nonlinear boosting challenger","Sequentially corrects residuals and provides a strong stability comparison."],
        ],columns=["Candidate","Role","Technical rationale"])
        st.dataframe(ladder,use_container_width=True,hide_index=True)
        rank=comp.sort_values("MAE_mean").reset_index(drop=True).copy(); rank["Rank"]=rank.index+1
        rank["Decision"]=["Selected" if i==0 else ("Strong challenger" if i==1 else ("Reference floor" if r.model=="Dummy Median" else "Not selected")) for i,r in rank.iterrows()]
        st.dataframe(rank[["Rank","model","MAE_mean","MAE_std","R2_mean","Decision"]].style.format({"MAE_mean":"${:,.0f}","MAE_std":"${:,.0f}","R2_mean":"{:.3f}"}),use_container_width=True,hide_index=True)
        interpretation_card(st,f"The ladder contains {len(comp)} candidates from non-ML floor through linear, regularized and nonlinear families.","A baseline ladder prevents a complex model from being promoted merely because it produces a plausible score.","Require every promoted family to materially beat Dummy Median on the same temporal folds.","info")

    with tabs[6]:
        downloadable_table(st,comp,"Model Comparison Summary","p4_comp")
        downloadable_table(st,folds,"Per-fold Temporal CV Evidence","p4_folds",height=450)
        if not fam.empty: downloadable_table(st,fam,"Feature-family Ablation","p4_fam")
        if not drift.empty: downloadable_table(st,drift,"Encoded Feature Importance Drift","p4_drift",height=450)
