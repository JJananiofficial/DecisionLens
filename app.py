from __future__ import annotations
import io, json, hashlib, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve
from decisionlens import data as D, decision as dec, evaluation as ev, explain as ex, models as M, pipeline as pl

st.set_page_config(page_title="DecisionLens | Decision Intelligence", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  color-scheme: light !important;
  --ink:#17243a; --muted:#64748b; --blue:#3768e8; --teal:#0f9f91;
  --line:#e7edf5; --canvas:#f5f7fb; --surface:#ffffff;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
  background-color:var(--canvas) !important;
  color:var(--ink) !important;
  color-scheme:light !important;
  font-family:'DM Sans',sans-serif !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
  background:#ffffff !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
.stMarkdown, .stMarkdown p, .stMarkdown span, .stCaption,
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color:var(--ink) !important;
}
h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  color:var(--ink) !important;
  font-family:'Manrope',sans-serif !important;
  letter-spacing:-.035em;
}
p, li, label, span { color:inherit; }
.block-container {max-width:1450px;padding-top:1.7rem;padding-bottom:3rem;}
h1 {font-size:2.2rem!important;font-weight:800!important;}
h2 {font-size:1.4rem!important;font-weight:750!important;}
[data-testid="stSidebar"] {border-right:1px solid var(--line)!important;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
[data-testid="stMetric"] {
  background:#fff!important;border:1px solid var(--line)!important;
  padding:16px 18px;border-radius:15px;
  box-shadow:0 3px 12px rgba(22,42,74,.035);
}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"],[data-testid="stMetricDelta"] * {color:var(--ink)!important;}
[data-testid="stMetricLabel"] {color:#687891!important;font-size:.78rem;font-weight:600;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-weight:800;}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color:var(--line)!important;border-radius:16px!important;background:#fff!important;
}
.stButton>button,.stDownloadButton>button {
  border-radius:10px;font-weight:650;border:1px solid #dce5f1;
  background:#fff!important;color:#17243a!important;
}
.stButton>button:hover,.stDownloadButton>button:hover {border-color:#3768e8!important;color:#2454ce!important;}
[data-testid="stDataFrame"],[data-testid="stTable"] {
  border:1px solid var(--line);border-radius:12px;overflow:hidden;
}
[data-testid="stDataFrame"] *, [data-testid="stTable"] * {color:#17243a!important;}
[data-baseweb="input"] input,[data-baseweb="select"] > div,
[data-baseweb="textarea"] textarea,[data-testid="stNumberInput"] input {
  background:#fff!important;color:#17243a!important;
  border-color:#dce5f1!important;
}
[data-baseweb="select"] *, [data-baseweb="popover"] * {color:#17243a!important;}
[data-testid="stSlider"] *,[data-testid="stRadio"] *,
[data-testid="stCheckbox"] *,[data-testid="stToggle"] * {color:#17243a!important;}
[data-testid="stFileUploader"] *,[data-testid="stExpander"] * {color:#17243a!important;}

.hero {
  background: linear-gradient(115deg, #14213d, #1d3761 58%, #3159a6);
  padding: 30px 34px;
  border-radius: 22px;
  color: white !important;
  margin-bottom: 22px;
  box-shadow: 0 12px 30px rgba(28,54,96,.12);
}

.hero * {
  color: inherit;
}

/* Hero subtitle */
.hero-kicker {
  font-size: .72rem;
  letter-spacing: .14em;
  font-weight: 700;
  color: #9dbbff !important;
  text-transform: uppercase;
}

/* Main heading */
.hero-title {
  font-family: 'Manrope', sans-serif;
  font-size: 2.15rem;
  font-weight: 800;
  line-height: 1.16;
  letter-spacing: -.04em;
  color: #fff !important;
  margin: 9px 0;
}

.hero-title span {
  color: #8FB5FF !important;
}

.hero-copy {
  color: #dfeeff !important;
  max-width: 760px;
  font-size: .96rem;
  line-height: 1.65;
}

.eyebrow {
  font-size: .72rem;
  letter-spacing: .12em;
  color: #71819a !important;
  text-transform: uppercase;
  font-weight: 700;
}

.good {
  background: #eaf8f4;
  border-left: 3px solid #16a085;
  padding: 12px 15px;
  border-radius: 8px;
  color: #24564d !important;
}

.small-muted {
  font-size: .82rem;
  color: #728198 !important;
}

</style>""", unsafe_allow_html=True)
BLUE,TEAL,AMBER,PURPLE,RED="#3768e8","#0f9f91","#e39a35","#8066d5","#d65a68"
COLORS={M.SIMPLE:BLUE,M.DECISION_TREE:AMBER,M.RANDOM_FOREST:TEAL,
        M.RANDOM_FOREST_CAL:"#16877c",M.EXTRA_TREES:"#d28b2d",
        M.EXPLAINABLE:PURPLE,M.EXPLAINABLE_CAL:"#5d4bb7",
        M.HIST_GB:"#d65a68",M.HIST_GB_CAL:"#b84b59",M.SVM:"#476b9b",
        M.BALANCED:AMBER,M.BALANCED_CAL:RED,"Oracle (true probabilities)":"#17243a"}
LAYOUT=dict(template="plotly_white",paper_bgcolor="white",plot_bgcolor="white",font=dict(family="DM Sans",color="#52627a",size=12),margin=dict(l=18,r=18,t=55,b=25),legend=dict(orientation="h",y=-.22))
def polish(fig,height=370):
    fig.update_layout(**LAYOUT,height=height);fig.update_xaxes(showgrid=True,gridcolor="#edf1f6",zeroline=False);fig.update_yaxes(showgrid=True,gridcolor="#edf1f6",zeroline=False);return fig
@st.cache_resource(show_spinner="Training models and preparing the experiment…")
def get_experiment(source,n,seed,stress,csv_bytes,target,positive):
    if source=="Simulated campaign data": df=D.generate_campaign_data(n=n,seed=seed);num,cat,tgt=D.NUMERIC,D.CATEGORICAL,D.TARGET
    else: df=D.load_csv(io.BytesIO(csv_bytes),target,positive);num,cat=D.infer_feature_types(df,target);tgt=target
    return df,pl.run_experiment(df,tgt,num,cat,seed,stress)
@st.cache_resource(show_spinner="Computing SHAP explanations…")
def cached_shap(_exp,key,model_name):
    X=_exp.data['test'][0].iloc[:min(100,len(_exp.data['test'][0]))]
    bg=_exp.data['train'][0].sample(min(80,len(_exp.data['train'][0])),random_state=0)
    explanation=ex.explanation_for_model(_exp.models[model_name],bg,X,model_name)
    return X,explanation
@st.cache_resource(show_spinner="Running robustness stress tests…")
def cached_robustness(_exp,key):
    return pl.robustness_table(_exp)

def report_text(df,exp,cm,metrics,summary,seed,source):
    digest=hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()).hexdigest()
    return "\n".join(["# DecisionLens — Experiment Report","",f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",f"Source: {source}",f"Rows: {len(df):,}",f"Target: {exp.target}",f"Seed: {seed}",f"Contact cost: {cm.cost_per_contact}",f"Conversion value: {cm.value_per_conversion}","","## Methodology","Stratified 50/15/15/20 train, calibration, validation and test partitions. Model fitting, probability calibration and threshold selection are separated; final metrics are calculated on the untouched test partition.","", "## Test metrics",metrics.to_markdown(),"","## Model fit and calibration time",
pd.DataFrame([{"Model":m,"Seconds":exp.fit_seconds.get(m)} for m in exp.model_names]).to_markdown(index=False),
"","## Decision outcomes",summary.to_markdown(index=False),"","## Limitations","- Offline conversion prediction does not estimate incremental campaign uplift.","- SHAP explains model behaviour, not causality.","- Subgroup metrics are descriptive and may be unstable for small groups.","- Economic outputs depend on the assumed contact cost and conversion value.","","## Reproducibility",f"Dataset SHA-256: {digest}",f"Python: {platform.python_version()}",f"NumPy: {np.__version__}",f"Pandas: {pd.__version__}"])

with st.sidebar:
    st.markdown("## ◈ DecisionLens");st.caption("DECISION INTELLIGENCE");st.markdown("---")
    section=st.radio("WORKSPACE",["Executive overview","Model performance","Calibration","Explainability","Business simulator","Fairness & subgroups","Research report"])
    st.markdown("---");st.markdown("**EXPERIMENT SETUP**")
    source=st.radio("Dataset",["Simulated campaign data","Upload CSV"]);csv_bytes=target=positive=None;n=20000
    if source=="Upload CSV":
        up=st.file_uploader("Upload binary-target CSV",type=["csv"])
        if up is None: st.info("Upload a CSV or switch to simulated data.");st.stop()
        csv_bytes=up.getvalue();cols=list(pd.read_csv(io.BytesIO(csv_bytes),nrows=100).columns);target=st.selectbox("Target column",cols,index=len(cols)-1)
        vals=pd.read_csv(io.BytesIO(csv_bytes),usecols=[target])[target].dropna().unique().tolist()
        if len(vals)!=2: st.error("Target must contain exactly two classes.");st.stop()
        positive=st.selectbox("Which value means positive?",vals,index=1)
    else:n=st.select_slider("Synthetic customer records",[5000,10000,20000,40000],value=20000)
    seed=int(st.number_input("Random seed",0,999999,42));stress=st.toggle("Include calibration stress test",value=True)
    st.markdown("---");st.markdown("**BUSINESS ASSUMPTIONS**")
    cost=st.number_input("Contact cost",.1,500.0,6.0,.5);value=st.number_input("Value per conversion",1.0,5000.0,40.0,1.0)
    cm=dec.CostModel(float(cost),float(value));st.metric("Break-even probability",f"{cm.breakeven:.1%}")
try: df,exp=get_experiment(source,n,seed,stress,csv_bytes,target,positive)
except Exception as e: st.error(f"Experiment could not run: {e}");st.info("Check target classes, dataset size and feature quality.");st.stop()
y=exp.data['test'][1];P={m:exp.probs[m]['test'] for m in exp.model_names};core=[m for m in exp.model_names if not m.startswith('Stress test:')]
metrics=ev.metrics_table(y,P,n_boot=200);summary=pl.decision_summary(exp,cm);tuned=summary[(summary.policy=="tuned on validation")&summary.model.isin(exp.model_names)].set_index('model')
y_val=exp.data['val'][1]
validation_policy_profit={m:dec.evaluate_policy(y_val,exp.probs[m]['val'],dec.best_threshold(y_val,exp.probs[m]['val'],cm),cm)['profit_per_1000'] for m in exp.model_names}
best=max(validation_policy_profit,key=validation_policy_profit.get);key=(source,n,seed,stress,target,positive)
st.markdown('<div class="hero"><div class="hero-kicker">BUSINESS ANALYTICS · MODEL EVALUATION · DECISION SCIENCE</div><div class="hero-title">Better predictions.<br><span style="color:#9dbbff">Smarter decisions.</span></div><div class="hero-copy">Evaluate whether predictive models create measurable business value — combining model quality, probability calibration, explainability and cost-sensitive decisions in one reproducible workspace.</div></div>',unsafe_allow_html=True)
st.caption(f"{'Synthetic experiment' if source=='Simulated campaign data' else 'Uploaded dataset'} · {len(df):,} records · Seed {seed} · Target: {exp.target}")

if section=="Executive overview":
    st.markdown('<div class="eyebrow">01 / EXECUTIVE BRIEF</div>',unsafe_allow_html=True);st.title("Experiment at a glance")
    c=st.columns(4);c[0].metric("Customers",f"{len(df):,}");c[1].metric("Conversion rate",f"{df[exp.target].mean():.1%}");c[2].metric("Features",len(exp.num)+len(exp.cat));c[3].metric("Leading validation policy",best.split('(')[0].strip())
    a,b=st.columns([1.35,1]);a.markdown("#### Model scorecard");a.dataframe(metrics.style.format({"ROC AUC":"{:.3f}","PR AUC":"{:.3f}","Brier":"{:.4f}","Log loss":"{:.4f}","ECE":"{:.4f}"}),use_container_width=True)
    b.markdown("#### Test-set outcome · validation-selected model");f=go.Figure();
    for m in core:f.add_bar(name=m,x=[m],y=[tuned.loc[m,'profit_per_1000']],marker_color=COLORS.get(m,BLUE))
    polish(f,300);f.update_layout(showlegend=False,yaxis_title="Profit per 1,000",xaxis_title="");b.plotly_chart(f,use_container_width=True)
    st.markdown(f'<div class="good"><b>Selected using validation only:</b> {best}. Final performance is reported on held-out test customers.</div>',unsafe_allow_html=True)
    st.markdown("#### Evaluation protocol");st.dataframe(pd.DataFrame([{"Partition":s.title(),"Rows":len(exp.data[s][1]),"Purpose":p} for s,p in zip(pl.SPLITS,["Fit model parameters","Fit probability calibration","Select decision thresholds","Final untouched evaluation"])]),hide_index=True,use_container_width=True)
elif section=="Model performance":
    st.markdown('<div class="eyebrow">02 / PREDICTIVE PERFORMANCE</div>',unsafe_allow_html=True);st.title("Compare model quality");st.dataframe(metrics,use_container_width=True)
    st.markdown("#### Training and calibration time")
    timing_df=pd.DataFrame([{"Model":m,"Fit + calibration seconds":exp.fit_seconds.get(m,np.nan)} for m in exp.model_names]).sort_values("Fit + calibration seconds")
    st.dataframe(timing_df.style.format({"Fit + calibration seconds":"{:.2f}s"}),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2);roc=go.Figure();pr=go.Figure()
    for m in exp.model_names:
        fpr,tpr,_=roc_curve(y,P[m]);precision,recall,_=precision_recall_curve(y,P[m]);roc.add_scatter(x=fpr,y=tpr,name=m,line=dict(color=COLORS.get(m,BLUE),width=2.5));pr.add_scatter(x=recall,y=precision,name=m,line=dict(color=COLORS.get(m,BLUE),width=2.5))
    roc.add_scatter(x=[0,1],y=[0,1],name="Chance",line=dict(dash="dot",color="#9aa7b8"));polish(roc,390);roc.update_layout(title="ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate");polish(pr,390);pr.update_layout(title="Precision–recall curve",xaxis_title="Recall",yaxis_title="Precision");c1.plotly_chart(roc,use_container_width=True);c2.plotly_chart(pr,use_container_width=True)
    ma=st.selectbox("Model A",core,index=1);mb=st.selectbox("Model B",core,index=0)
    if ma!=mb:
        cmp=pl.compare_models(exp,cm,ma,mb,n_boot=300);x=st.columns(2);x[0].metric("ROC-AUC difference (A − B)",f"{cmp['auc_diff']['mean']:+.3f}",f"95% CI {cmp['auc_diff']['lo']:+.3f} to {cmp['auc_diff']['hi']:+.3f}",delta_color="off");x[1].metric("Profit difference / 1,000",f"{cmp['profit_diff_per_1000']['mean']:+,.0f}",f"95% CI {cmp['profit_diff_per_1000']['lo']:+,.0f} to {cmp['profit_diff_per_1000']['hi']:+,.0f}",delta_color="off")
    st.caption("ROC-AUC measures ranking; PR-AUC reflects precision/recall; Brier and log loss measure probability quality; ECE measures calibration error.")
    with st.expander("Advanced research: robustness under input perturbations"):
        st.write("Descriptive held-out diagnostics under random missingness and measurement noise. These scenarios are not used to select models.")
        if st.button("Run robustness stress tests",key="run_robustness"):
            robust=pl.robustness_table(exp)
            st.session_state["decisionlens_robustness"]=robust
        if "decisionlens_robustness" in st.session_state:
            st.dataframe(st.session_state["decisionlens_robustness"].style.format({"ROC AUC":"{:.3f}","AUC change vs clean":"{:+.3f}","Brier":"{:.4f}"}),use_container_width=True,hide_index=True)
elif section=="Calibration":
    st.markdown('<div class="eyebrow">03 / PROBABILITY RELIABILITY</div>',unsafe_allow_html=True);st.title("Are the probabilities trustworthy?");f=go.Figure();f.add_scatter(x=[0,1],y=[0,1],name="Perfect calibration",line=dict(color="#8c9aab",dash="dash"))
    for m in exp.model_names:
        t=ev.calibration_table(y,P[m],10);f.add_scatter(x=t.mean_predicted,y=t.observed_rate,name=m,mode="lines+markers",line=dict(color=COLORS.get(m,BLUE),width=2.5))
    polish(f,440);f.update_layout(xaxis_title="Mean predicted probability",yaxis_title="Observed conversion rate",xaxis_range=[0,1],yaxis_range=[0,1]);st.plotly_chart(f,use_container_width=True)
    calibration_rows=[{"Model":m,"ECE":ev.expected_calibration_error(y,P[m]),"Brier":np.mean((P[m]-y)**2)} for m in exp.model_names]
    st.dataframe(pd.DataFrame(calibration_rows).style.format({"ECE":"{:.4f}","Brier":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.info(f"Break-even probability under current economics: {cm.breakeven:.1%}. Miscalibration can cause economically poor decisions even when ranking is strong.")
elif section=="Explainability":
    st.markdown('<div class="eyebrow">04 / MODEL INTERPRETABILITY</div>',unsafe_allow_html=True)
    st.title("Understand the model's reasoning")
    st.caption("TreeSHAP is used for supported tree pipelines; model-agnostic permutation-SHAP is used for calibrated models, SVM and histogram boosting.")
    explain_model=st.selectbox("Choose a model to explain",core,index=core.index(M.EXPLAINABLE) if M.EXPLAINABLE in core else 0)
    X,shap_exp=cached_shap(exp,key,explain_model)
    vals=np.asarray(shap_exp.values)
    if vals.ndim==3: vals=vals[..., -1]
    names=list(shap_exp.feature_names)
    importance=pd.Series(np.abs(vals).mean(axis=0),index=names).sort_values(ascending=False)
    st.markdown("#### Global feature importance")
    top=importance.head(18).sort_values()
    fig=go.Figure(go.Bar(x=top.values,y=top.index,orientation="h",marker_color=TEAL))
    polish(fig,430);fig.update_layout(title=f"{explain_model} · mean absolute SHAP",xaxis_title="Mean absolute SHAP value")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("For TreeSHAP, values are in the model's native output space. For model-agnostic explanations, values describe positive-class probability. Magnitudes should not be compared directly across explanation methods.")
    import matplotlib.pyplot as plt, shap
    feature=st.selectbox("Inspect feature dependence",names)
    plt.figure(figsize=(8,4));shap.plots.scatter(shap_exp[:,feature],show=False);st.pyplot(plt.gcf(),clear_figure=True)
    idx=st.slider("Test customer index",0,len(X)-1,0)
    st.markdown("#### Individual explanation")
    st.dataframe(X.iloc[[idx]],use_container_width=True)
    st.metric("Predicted positive-class probability",f"{exp.probs[explain_model]['test'][idx]:.1%}")
    plt.figure(figsize=(8,4));shap.plots.waterfall(shap_exp[idx],max_display=10,show=False);st.pyplot(plt.gcf(),clear_figure=True)
elif section=="Business simulator":
    st.markdown('<div class="eyebrow">05 / BUSINESS DECISION LAB</div>',unsafe_allow_html=True);st.title("Simulate the economics of a decision");st.write("Change commercial assumptions and operating threshold. Results use held-out customers to illustrate trade-offs; this is an offline simulation, not a live campaign.")
    c=st.columns(3);sc=c[0].slider("Cost per contact",.5,50.0,float(cost),.5);sv=c[1].slider("Value per conversion",5.0,300.0,float(value),1.0);th=c[2].slider("Contact probability threshold",0.0,1.0,float(np.clip(cm.breakeven,0,1)),.01)
    model=st.selectbox("Model",core,index=1);simcm=dec.CostModel(sc,sv);r=dec.evaluate_policy(y,P[model],th,simcm);k=st.columns(4);k[0].metric("Customers contacted",f"{r['contacted']:,}",f"{r['share_contacted']:.1%} of test set");k[1].metric("Conversions captured",f"{r['conversions']:,}");k[2].metric("Precision",f"{r['precision']:.1%}" if not np.isnan(r['precision']) else "—");k[3].metric("Profit per 1,000",f"{r['profit_per_1000']:,.0f}")
    curve=dec.profit_curve(y,P[model],simcm);f=go.Figure();f.add_scatter(x=curve.share_contacted,y=curve.profit/len(y)*1000,mode='lines',name='Profit frontier',line=dict(color=BLUE,width=3));f.add_vline(x=r['share_contacted'],line_dash='dash',line_color=TEAL);polish(f,390);f.update_layout(title="Profit frontier",xaxis_title="Share contacted",yaxis_title="Profit per 1,000");st.plotly_chart(f,use_container_width=True);st.dataframe(pl.decision_summary(exp,simcm),use_container_width=True,hide_index=True)
    st.caption("The target is observed conversion, not treatment uplift. This does not estimate incremental conversions caused by contact.")
elif section=="Fairness & subgroups":
    st.markdown('<div class="eyebrow">06 / SUBGROUP ROBUSTNESS</div>',unsafe_allow_html=True);st.title("Inspect performance across groups");features=[c for c in exp.num+exp.cat if 2<=df[c].nunique(dropna=True)<=30]
    if not features:st.warning("No suitable subgroup features. Choose a dataset with a categorical or low-cardinality feature.")
    else:
        group=st.selectbox("Subgroup feature",features);model=st.selectbox("Model",exp.model_names,index=1);threshold=dec.best_threshold(exp.data['val'][1],exp.probs[model]['val'],cm);values=exp.data['test'][0][group].reset_index(drop=True).fillna('Missing').astype(str);rows=[]
        for v in values.unique():
            mask=(values==v).to_numpy();yy=y[mask];pp=P[model][mask];pred=pp>=threshold;tp=((pred==1)&(yy==1)).sum();fp=((pred==1)&(yy==0)).sum();fn=((pred==0)&(yy==1)).sum()
            rows.append({"Group":v,"N":len(yy),"Positive rate":yy.mean(),"Selection rate":pred.mean(),"Precision":tp/(tp+fp) if tp+fp else np.nan,"Recall / TPR":tp/(tp+fn) if tp+fn else np.nan,"Mean predicted risk":pp.mean(),"Brier":np.mean((pp-yy)**2)})
        table=pd.DataFrame(rows).sort_values('N',ascending=False);st.caption(f"Threshold {threshold:.3f} selected on validation data. Interpret small groups cautiously.");st.dataframe(table.style.format({c:'{:.1%}' for c in ['Positive rate','Selection rate','Precision','Recall / TPR']}).format({'Mean predicted risk':'{:.3f}','Brier':'{:.3f}'}),use_container_width=True,hide_index=True)
        a,b=st.columns(2)
        for col,metric,title in [(a,'Selection rate','Selection rate by subgroup'),(b,'Recall / TPR','Recall by subgroup')]:
            f=px.bar(table,x='Group',y=metric,color='Group',text_auto='.1%');polish(f,330);f.update_layout(title=title,showlegend=False);col.plotly_chart(f,use_container_width=True)
        st.info("Descriptive subgroup audit, not a legal fairness certification. Metrics depend on label quality, group definitions, sample sizes and decision context. Do not infer protected attributes.")
        if table.N.min()<50:st.warning("At least one subgroup has fewer than 50 test records; its rates may be highly uncertain.")
elif section=="Research report":
    st.markdown('<div class="eyebrow">07 / RESEARCH OUTPUT</div>',unsafe_allow_html=True);st.title("Report and reproduce this experiment");st.write("Export the method, test metrics, decision outcomes and experiment fingerprint.")
    st.download_button("Download research report (.md)",report_text(df,exp,cm,metrics,summary,seed,source).encode(),"DecisionLens_Experiment_Report.md","text/markdown",type="primary");st.download_button("Download decision results (.csv)",summary.to_csv(index=False).encode(),"DecisionLens_Decision_Results.csv","text/csv")
    digest=hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()).hexdigest();manifest={"project":"DecisionLens","created_utc":datetime.now(timezone.utc).isoformat(),"data_source":source,"dataset_sha256":digest,"rows":len(df),"target":exp.target,"positive_class":"1 (internal encoding)","random_seed":seed,"stress_test":stress,"contact_cost":cost,"value_per_conversion":value,"split_sizes":{s:len(exp.data[s][1]) for s in pl.SPLITS},"models":exp.model_names,"fit_seconds":exp.fit_seconds,"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
    st.markdown("#### Reproducibility manifest");st.json(manifest);st.download_button("Download experiment manifest (.json)",json.dumps(manifest,indent=2).encode(),"DecisionLens_Experiment_Manifest.json","application/json")
    st.markdown("#### Research limitations\n- Simulated data demonstrates methodology, not real-world performance.\n- Offline prediction does not measure causal campaign uplift.\n- SHAP is explanatory, not causal.\n- Subgroup metrics are descriptive and require adequate sample sizes.\n- The economic model assumes constant contact cost and conversion value.")
    with st.expander("Generated findings"):
        for finding in pl.build_findings(exp,cm):st.markdown(f"- {finding}")
st.markdown("---");st.markdown('<div class="small-muted">DECISIONLENS · Experimental decision-support prototype · Results are analytical estimates, not operational recommendations.</div>',unsafe_allow_html=True)
