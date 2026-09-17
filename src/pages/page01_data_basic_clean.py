from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .common import *
from ai_job_market.core import dynamic_insights


def _filters(st, df):
    with st.expander("Interactive filters", expanded=True):
        c1,c2,c3,c4=st.columns(4)
        cats=c1.multiselect("Job category", sorted(df.job_category.dropna().astype(str).unique()), key="p1_cat")
        levels=c2.multiselect("Experience level", sorted(df.experience_level.dropna().astype(str).unique()), key="p1_exp")
        countries=c3.multiselect("Country", sorted(df.country.dropna().astype(str).unique()), key="p1_country")
        remote=c4.multiselect("Remote work", sorted(df.remote_work.dropna().astype(str).unique()), key="p1_remote")
    return apply_filters(df,{"job_category":cats,"experience_level":levels,"country":countries,"remote_work":remote})


def render(st, root, role="admin"):
    style_page(st)
    st.title("1. Data Basic Clean")
    st.caption("Common Foundation — Stage 1 Project Scope & Raw Data Ingestion → Stage 2 Basic Clean → Stage 3 Contradictory-Feature Investigation")
    audit=read_json(root,"01_data_basic_clean/basic_clean_audit.json")
    target=read_json(root,"01_data_basic_clean/target_summary.json")
    clean=read_csv(root,"01_data_basic_clean/basic_clean.csv")
    findings=read_csv(root,"01_data_basic_clean/contradiction_summary.csv")
    flags=read_csv(root,"01_data_basic_clean/integrity_row_flags.csv") if (root/"outputs/01_data_basic_clean/integrity_row_flags.csv").exists() else pd.DataFrame()
    metric_cols(st,[("Raw records",f"{audit['raw_rows']:,}"),("Raw columns",audit['raw_columns']),("Clean records",f"{audit['clean_rows']:,}"),("Clean columns",audit['clean_columns']),("Removed corrupted rows",audit['invalid_category_rows_removed'])])

    filtered=_filters(st,clean)
    if filtered.empty:
        st.warning("No records match the current filters. Clear one or more filters to continue.")
        return

    tabs=st.tabs(["Stage 1 · Scope & Profiling","Stage 2 · Basic Clean","Stage 3 · Contradictions","Interactive Chart Explorer","Evidence Tables"])
    with tabs[0]:
        st.markdown("### Stage 1 — Project Scope & Raw Data Ingestion")
        st.markdown('<div class="stage-note">Target <b>annual_salary_usd</b>. The raw file is parsed and profiled before model decisions. Salary extremes are retained unless there is evidence that they are invalid.</div>',unsafe_allow_html=True)
        c1,c2=st.columns([1.7,1])
        with c1:
            fig=px.histogram(filtered,x="annual_salary_usd",nbins=24,marginal="box",title="Observed Annual Salary Distribution",labels={"annual_salary_usd":"Annual salary (USD)"})
            fig.add_vline(x=float(filtered.annual_salary_usd.mean()),line_dash="dash",annotation_text=f"Mean {money(filtered.annual_salary_usd.mean())}")
            fig.add_vline(x=float(filtered.annual_salary_usd.median()),line_dash="dot",annotation_text=f"Median {money(filtered.annual_salary_usd.median())}")
            show_plot(st,fig,"p1_target_hist")
        with c2:
            q=filtered.annual_salary_usd.quantile([.25,.5,.75])
            st.dataframe(pd.DataFrame({"Metric":["Rows","Min","Q1","Median","Mean","Q3","Max","Std"],"Value":[f"{len(filtered):,}",money(filtered.annual_salary_usd.min()),money(q.loc[.25]),money(q.loc[.5]),money(filtered.annual_salary_usd.mean()),money(q.loc[.75]),money(filtered.annual_salary_usd.max()),money(filtered.annual_salary_usd.std())]}),hide_index=True,use_container_width=True)
        skew_note="right-skewed" if filtered.annual_salary_usd.mean()>filtered.annual_salary_usd.median() else "left-skewed or symmetric"
        interpretation_card(st,f"Filtered salary mean is {money(filtered.annual_salary_usd.mean())} versus median {money(filtered.annual_salary_usd.median())}; range is {money(filtered.annual_salary_usd.min())}–{money(filtered.annual_salary_usd.max())}.",f"The target is {skew_note}. Extreme salaries are visible but are not removed automatically because the chart alone does not prove data corruption.","Retain target extremes for now; handle any outlier policy on TRAIN only and validate impact on temporal performance.","info")
        with st.expander("Raw schema / profiling evidence"):
            st.dataframe(read_csv(root,"01_data_basic_clean/raw_profile.csv"),use_container_width=True,hide_index=True)
            if (root/"outputs/01_data_basic_clean/numeric_descriptive_summary.csv").exists():
                st.dataframe(read_csv(root,"01_data_basic_clean/numeric_descriptive_summary.csv"),use_container_width=True,hide_index=True)

    with tabs[1]:
        st.markdown("### Stage 2 — Basic Clean")
        st.success(f"Basic clean removed {audit['invalid_category_rows_removed']} corrupted category row(s) and the unique `job_id` identifier. The output is **{audit['clean_rows']:,} rows × {audit['clean_columns']} columns**. `skill_count` is created later during feature preparation, not during basic cleaning.")
        compare=pd.DataFrame({"Aspect":["Records","Columns","Missing cells","Duplicate rows","Unique identifier"],"Raw":[audit['raw_rows'],audit['raw_columns'],0,audit['duplicate_rows_removed'],"job_id present"],"Basic clean":[audit['clean_rows'],audit['clean_columns'],0,0,"job_id removed"]})
        st.dataframe(compare,use_container_width=True,hide_index=True)
        c1,c2=st.columns(2)
        with c1:
            prof=read_csv(root,"01_data_basic_clean/raw_profile.csv").sort_values("unique",ascending=False).head(15)
            fig=px.bar(prof.sort_values("unique"),y="column",x="unique",orientation="h",text="unique",title="Top Feature Cardinalities — Raw Data")
            fig.update_traces(textposition="outside")
            show_plot(st,fig,"p1_cardinality")
        with c2:
            # all missing counts are visible, including zeros
            prof=read_csv(root,"01_data_basic_clean/raw_profile.csv")
            miss=prof.assign(total_missing=prof.missing+prof.hidden_missing).sort_values("total_missing",ascending=False)
            fig=px.bar(miss,y="column",x="total_missing",orientation="h",text="total_missing",title="Missing / Hidden-Missing Audit by Feature")
            show_plot(st,fig,"p1_missing")
        total_missing=int((read_csv(root,"01_data_basic_clean/raw_profile.csv").missing+read_csv(root,"01_data_basic_clean/raw_profile.csv").hidden_missing).sum())
        interpretation_card(st,f"Basic Clean changed {audit['raw_rows']-audit['clean_rows']} row(s) and {audit['raw_columns']-audit['clean_columns']} column(s); raw missing/hidden-missing count is {total_missing}.","Structural cleanliness alone is not enough: the dataset can still contain semantic contradictions even when missingness and duplication are low.","Proceed to Stage 3 contradiction checks before trusting any feature for modeling.","success" if total_missing==0 else "warning")

    with tabs[2]:
        st.markdown("### Stage 3 — Contradictory-Feature Investigation")
        fig=px.bar(findings.sort_values("affected_pct"),y="issue",x="affected_pct",orientation="h",text="affected_pct",title="Data Logic & Integrity Issues — Ranked by Severity",labels={"affected_pct":"Affected records (%)","issue":"Issue"})
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        show_plot(st,fig,"p1_issue_rank")
        insight_box(st,dynamic_insights(root/"outputs").get("data_quality",""),"warning")
        st.dataframe(findings.style.format({"affected_pct":"{:.2f}%"}),use_container_width=True,hide_index=True)
        top_issue=findings.sort_values("affected_pct",ascending=False).iloc[0]
        interpretation_card(st,f"Largest logic finding: {top_issue.issue} affects {top_issue.affected_pct:.1f}% ({int(top_issue.affected_rows):,} rows).","This is a semantic-integrity problem, not a missing-value problem; the affected field family must be governed before preprocessing.",str(top_issue.required_action),"warning")

        s1,s2,s3=st.tabs(["Experience contradiction","Salary range inconsistency","Salary tier inconsistency"])
        with s1:
            c1,c2=st.columns(2)
            with c1:
                fig=px.box(filtered,x="experience_level",y="years_of_experience",points="outliers",title="Years of Experience by Experience Level")
                show_plot(st,fig,"p1_exp_box")
            with c2:
                g=filtered.groupby("experience_level",observed=True).agg(mean_salary=("annual_salary_usd","mean"),median_salary=("annual_salary_usd","median"),records=("annual_salary_usd","size")).reset_index()
                m=g.melt(id_vars=["experience_level","records"],value_vars=["mean_salary","median_salary"],var_name="metric",value_name="salary")
                fig=px.bar(m,x="experience_level",y="salary",color="metric",barmode="group",text="salary",title="Annual Salary by Experience Level")
                fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
                show_plot(st,fig,"p1_exp_salary")
            byy=filtered.groupby("years_of_experience",observed=True).agg(mean_salary=("annual_salary_usd","mean"),median_salary=("annual_salary_usd","median"),records=("annual_salary_usd","size")).reset_index()
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=byy.years_of_experience,y=byy.mean_salary,mode="lines+markers+text",text=[f"${x:,.0f}" for x in byy.mean_salary],textposition="top center",name="Mean"))
            fig.add_trace(go.Scatter(x=byy.years_of_experience,y=byy.median_salary,mode="lines+markers",name="Median"))
            fig.update_layout(title="Annual Salary by Raw Years of Experience",xaxis_title="Years of experience",yaxis_title="Salary (USD)")
            show_plot(st,fig,"p1_year_salary")
            corr=float(filtered[["years_of_experience","annual_salary_usd"]].corr().iloc[0,1]) if len(filtered)>2 else float("nan")
            spread=float(g.mean_salary.max()-g.mean_salary.min()) if len(g)>1 else 0.0
            interpretation_card(st,f"Experience-level mean-salary spread is {money(spread)} while raw years-of-experience correlation with salary is {corr:+.2f}.","If bucket labels and numeric years tell materially different stories, they should not both be trusted as interchangeable experience signals.","Keep the contradiction visible and let ablation decide which signal adds stable out-of-time value.","warning")
        with s2:
            if not flags.empty:
                d=flags.copy()
                d["range_status"]=np.where(d.salary_range_mismatch.astype(bool),"Outside stated range","Inside stated range")
                c1,c2=st.columns(2)
                with c1:
                    g=d.range_status.value_counts().rename_axis("status").reset_index(name="records")
                    fig=px.bar(g,x="status",y="records",text="records",title="Salary Range Audit — Status")
                    fig.update_traces(textposition="outside"); show_plot(st,fig,"p1_range_status")
                with c2:
                    fig=px.histogram(d,x="annual_salary_usd",color="range_status",nbins=25,barmode="overlay",title="Actual Salary Distribution by Range Status")
                    show_plot(st,fig,"p1_range_hist")
                sample=d.sort_values("annual_salary_usd").copy().reset_index(drop=True); sample["row"]=np.arange(len(sample))
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=sample.row,y=sample.salary_min_usd,mode="lines",name="Stated min"))
                fig.add_trace(go.Scatter(x=sample.row,y=sample.salary_max_usd,mode="lines",name="Stated max"))
                fig.add_trace(go.Scatter(x=sample.row,y=sample.annual_salary_usd,mode="markers",name="Actual salary",marker=dict(size=5)))
                fig.update_layout(title="Actual Salary vs Stated Salary Bounds",xaxis_title="Records sorted by actual salary",yaxis_title="USD")
                show_plot(st,fig,"p1_range_detail")
                rate=100*d.salary_range_mismatch.astype(bool).mean()
                interpretation_card(st,f"{rate:.1f}% of filtered records fall outside the stated salary_min_usd / salary_max_usd range.","The range fields are target-adjacent metadata and cannot be treated as clean predictors when they disagree with the target this often.","Block salary_min_usd and salary_max_usd from the primary prediction feature set; retain them only for audit.","warning")
        with s3:
            if not flags.empty:
                d=flags.copy(); d["tier_status"]=np.where(d.salary_tier_mismatch.astype(bool),"Mismatch","Match")
                c1,c2=st.columns(2)
                with c1:
                    g=d.tier_status.value_counts().rename_axis("status").reset_index(name="records")
                    fig=px.bar(g,x="status",y="records",text="records",title="Salary Tier Consistency — Status")
                    show_plot(st,fig,"p1_tier_status")
                with c2:
                    ct=pd.crosstab(d.salary_tier,d.expected_salary_tier)
                    fig=px.imshow(ct,text_auto=True,aspect="auto",title="Stated Tier vs Expected Tier from Annual Salary",labels=dict(x="Expected tier",y="Stated tier",color="Records"))
                    show_plot(st,fig,"p1_tier_heat")
                fig=px.box(d,x="salary_tier",y="annual_salary_usd",color="tier_status",points=False,title="Observed Salary Distribution by Stated Tier")
                show_plot(st,fig,"p1_tier_box")
                rate=100*d.salary_tier_mismatch.astype(bool).mean()
                interpretation_card(st,f"{rate:.1f}% of filtered records have a stated salary tier that disagrees with the tier implied by annual_salary_usd.","salary_tier behaves like a noisy derivative of the target, creating both leakage risk and semantic inconsistency.","Exclude salary_tier from model X and use it only as a diagnostic field.","warning")

    with tabs[3]:
        st.markdown("### Interactive Chart Explorer")
        chart=chart_selector(st,"Choose chart",["Salary by job domain","Salary by country","Salary by city","Salary by years of experience","Demand score vs salary","Benefits score vs salary"],"p1_chart_select")
        focus=st.slider("Top categories to display",5,30,12,key="p1_topn")
        if chart=="Salary by job domain":
            g=filtered.groupby("job_category").agg(mean_salary=("annual_salary_usd","mean"),records=("annual_salary_usd","size")).reset_index().nlargest(focus,"mean_salary")
            fig=px.bar(g.sort_values("mean_salary"),y="job_category",x="mean_salary",orientation="h",text="mean_salary",hover_data=["records"],title="Mean Salary by Job Domain"); fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
        elif chart=="Salary by country":
            g=filtered.groupby("country").agg(mean_salary=("annual_salary_usd","mean"),records=("annual_salary_usd","size")).reset_index().nlargest(focus,"mean_salary")
            fig=px.bar(g.sort_values("mean_salary"),y="country",x="mean_salary",orientation="h",text="mean_salary",hover_data=["records"],title="Mean Salary by Country"); fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
        elif chart=="Salary by city":
            g=filtered.groupby(["city","country"]).agg(mean_salary=("annual_salary_usd","mean"),records=("annual_salary_usd","size")).reset_index().nlargest(focus,"mean_salary")
            fig=px.bar(g.sort_values("mean_salary"),y="city",x="mean_salary",orientation="h",color="country",text="mean_salary",hover_data=["records"],title="Mean Salary by City"); fig.update_traces(texttemplate="$%{text:,.0f}",textposition="outside")
        elif chart=="Salary by years of experience":
            g=filtered.groupby("years_of_experience").agg(mean_salary=("annual_salary_usd","mean"),median_salary=("annual_salary_usd","median"),records=("annual_salary_usd","size")).reset_index()
            fig=px.line(g,x="years_of_experience",y=["mean_salary","median_salary"],markers=True,title="Salary by Years of Experience",labels={"value":"Salary (USD)","variable":"Metric"})
        elif chart=="Demand score vs salary":
            fig=px.scatter(filtered,x="demand_score",y="annual_salary_usd",color="job_category",hover_data=["job_title","country","years_of_experience"],title="Demand Score vs Annual Salary",trendline=None)
        else:
            fig=px.scatter(filtered,x="benefits_score_10",y="annual_salary_usd",color="job_category",hover_data=["job_title","country","years_of_experience"],title="Benefits Score vs Annual Salary")
        show_plot(st,fig,"p1_explorer")
        st.caption(f"Current chart uses {len(filtered):,} filtered records. Hover over marks to inspect exact values.")
        if chart in {"Salary by job domain","Salary by country","Salary by city"}:
            top=g.sort_values("mean_salary",ascending=False).iloc[0]
            label_col="job_category" if "job_category" in g.columns else ("country" if chart=="Salary by country" else "city")
            interpretation_card(st,f"Highest mean salary in the current chart is {top[label_col]} at {money(top.mean_salary)} across {int(top.records):,} record(s).","This is a descriptive filtered comparison; unequal sample sizes and synthetic data structure can drive apparent gaps.","Use hover counts and filters to check support before interpreting category differences.","info")
        elif chart=="Salary by years of experience":
            corr=float(filtered[["years_of_experience","annual_salary_usd"]].corr().iloc[0,1]) if len(filtered)>2 else float("nan")
            interpretation_card(st,f"Current-filter Pearson correlation between years_of_experience and salary is {corr:+.2f}.","The direction is diagnostic only and may reflect the known synthetic-looking experience artifact.","Compare this chart with Stage 5 ablation and temporal CV rather than treating it as a labor-market law.","warning")
        else:
            xcol="demand_score" if chart=="Demand score vs salary" else "benefits_score_10"
            corr=float(filtered[[xcol,"annual_salary_usd"]].corr().iloc[0,1]) if len(filtered)>2 else float("nan")
            interpretation_card(st,f"Current-filter correlation between {xcol} and salary is {corr:+.2f}.","Scatter structure shows association strength and heterogeneity, not causality.","Use this as exploratory evidence only; model selection remains based on temporal validation.","info")

    with tabs[4]:
        downloadable_table(st,findings,"Contradiction Summary","p1_findings")
        if not flags.empty:
            downloadable_table(st,flags,"Row-level Integrity Flags","p1_flags","integrity_row_flags.csv",height=420)
        if (root/"outputs/01_data_basic_clean/categorical_frequency_summary.csv").exists():
            downloadable_table(st,read_csv(root,"01_data_basic_clean/categorical_frequency_summary.csv"),"Categorical Frequency Summary","p1_catfreq",height=420)
