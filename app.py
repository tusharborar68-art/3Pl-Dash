"""Executive Decision Intelligence Dashboard — 3PL D2C (single-file, deploy-safe).

Upload survey data -> auto-clean -> analytics -> insights -> strategic recommendations.
Sidebar page navigation; each page is rendered in isolation (one failure never blanks
the app). Pure-Python styling (no matplotlib). Optional libs (xgboost, shap) are used
only if installed; otherwise the relevant module is gated with a professional message.

Run:  streamlit run app.py
"""
import io
import warnings
from itertools import combinations
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from types import SimpleNamespace

from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, r2_score, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso,
                                  ElasticNet, lasso_path)

# ---- optional heavy libs (never required for the app to boot) ----
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

st.set_page_config(page_title="3PL Decision Intelligence", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

# =========================================================================================
# THEME
# =========================================================================================
NAVY="#102A43"; INK="#1B2A3A"; TEAL="#14B8A6"; INDIGO="#6366F1"; AMBER="#F59E0B"
SKY="#0EA5E9"; VIOLET="#8B5CF6"; PINK="#EC4899"; GREEN="#10B981"; RED="#EF4444"; SLATE="#64748B"
SEQ=[TEAL,INDIGO,AMBER,SKY,VIOLET,PINK,GREEN,RED,"#0891B2","#9333EA"]

def apply_plotly_theme(dark=False):
    ink = "#E6EDF6" if dark else INK
    grid = "#24314D" if dark else "#E6EBF2"
    pio.templates["exec"] = go.layout.Template(layout=dict(
        font=dict(family="Inter,Segoe UI,sans-serif", color=ink, size=13),
        colorway=SEQ, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=15, color=ink)),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=ink)),
        margin=dict(l=40,r=20,t=48,b=40)))
    pio.templates.default = "exec"

def inject_css(dark=False):
    if dark:
        bg,card,ink,muted,line,side="#0E1726","#16213A","#E6EDF6","#93A4BD","#233149","#0B1220"
    else:
        bg,card,ink,muted,line,side="#F4F6FA","#FFFFFF","#1B2A3A","#64748B","#E3E9F1","#0E2438"
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {{ --bg:{bg}; --card:{card}; --ink:{ink}; --muted:{muted}; --line:{line};
             --teal:{TEAL}; --navy:{NAVY}; }}
    html,body,[class*="css"] {{ font-family:'Inter',sans-serif; }}
    .stApp {{ background:var(--bg); color:var(--ink); }}
    .block-container {{ padding-top:1.4rem; padding-bottom:2rem; max-width:1560px; }}
    h1,h2,h3,h4,p,span,label,div {{ color:var(--ink); }}

    .hero {{ background:linear-gradient(120deg,{NAVY} 0%,#173d63 55%,{TEAL} 145%);
        border-radius:18px; padding:20px 26px; margin-bottom:14px;
        box-shadow:0 10px 30px rgba(16,42,67,.20); }}
    .hero h1 {{ color:#fff!important; font-size:1.5rem; font-weight:800; margin:0; letter-spacing:-.02em; }}
    .hero p {{ color:#cfe6f0!important; margin:.25rem 0 0; font-size:.9rem; }}

    .phead {{ font-size:1.28rem; font-weight:800; color:var(--navy); margin:.2rem 0 .1rem; }}
    .pq {{ background:{TEAL}14; border-left:4px solid {TEAL}; color:var(--ink);
           padding:8px 14px; border-radius:8px; font-size:.9rem; margin:.2rem 0 .8rem; font-weight:600; }}
    .sec {{ color:var(--navy); font-size:1.02rem; font-weight:700; margin:1rem 0 .2rem;
            padding-bottom:.3rem; border-bottom:2px solid var(--line); }}
    .hint {{ color:var(--muted); font-size:.82rem; margin:.1rem 0 .6rem; }}

    .callout {{ border-radius:10px; padding:10px 14px; margin:.5rem 0 .2rem; font-size:.88rem;
                line-height:1.5; border:1px solid var(--line); background:var(--card); }}
    .ci {{ border-left:4px solid {AMBER}; }} .ca {{ border-left:4px solid {GREEN}; }}
    .cm {{ border-left:4px solid {INDIGO}; }}

    .kpi-row {{ display:flex; gap:13px; flex-wrap:wrap; margin:.2rem 0 1rem; }}
    .kpi {{ flex:1; min-width:150px; background:var(--card); border-radius:14px; padding:14px 16px;
        box-shadow:0 4px 14px rgba(16,42,67,.07); border-left:5px solid {TEAL};
        transition:transform .15s, box-shadow .15s; }}
    .kpi:hover {{ transform:translateY(-3px); box-shadow:0 12px 24px rgba(16,42,67,.14); }}
    .kpi .lbl {{ color:var(--muted); font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
    .kpi .val {{ color:var(--navy); font-size:1.4rem; font-weight:800; margin-top:3px; }}
    .kpi .sub {{ font-size:.73rem; font-weight:600; margin-top:2px; }}
    .kpi.t-teal{{border-left-color:{TEAL}}} .kpi.t-indigo{{border-left-color:{INDIGO}}}
    .kpi.t-amber{{border-left-color:{AMBER}}} .kpi.t-violet{{border-left-color:{VIOLET}}}
    .kpi.t-sky{{border-left-color:{SKY}}} .kpi.t-red{{border-left-color:{RED}}}
    .kpi.t-green{{border-left-color:{GREEN}}}
    .up{{color:{GREEN}}} .down{{color:{RED}}}
    .kpi .val, .kpi.t-teal .val {{ color:var(--navy); }}

    .badge {{ display:inline-block; background:{TEAL}1a; color:{TEAL}; font-weight:700;
        padding:3px 11px; border-radius:20px; font-size:.78rem; margin:2px 6px 2px 0; }}
    .badge.best {{ background:{GREEN}1a; color:{GREEN}; }}
    .badge.warn {{ background:{AMBER}1a; color:{AMBER}; }}
    .badge.off {{ background:{SLATE}22; color:{SLATE}; }}

    .gate {{ background:var(--card); border:1px dashed {SLATE}66; border-radius:12px;
        padding:18px 20px; color:var(--muted); font-size:.9rem; }}
    .gate b {{ color:var(--ink); }}

    section[data-testid="stSidebar"] {{ background:{side}; }}
    section[data-testid="stSidebar"] * {{ color:#dbe7f0!important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-weight:600; }}
    [data-testid="stMetricValue"] {{ color:var(--navy); }}
    </style>""", unsafe_allow_html=True)

def hero(t,s): st.markdown(f"<div class='hero'><h1>{t}</h1><p>{s}</p></div>", unsafe_allow_html=True)
def page_header(icon,title,q):
    st.markdown(f"<div class='phead'>{icon} {title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pq'>❓ Business question: {q}</div>", unsafe_allow_html=True)
def section(t,hint=None):
    st.markdown(f"<div class='sec'>{t}</div>", unsafe_allow_html=True)
    if hint: st.markdown(f"<div class='hint'>{hint}</div>", unsafe_allow_html=True)
def insight(t): st.markdown(f"<div class='callout ci'>💡 <b>Key insight</b> — {t}</div>", unsafe_allow_html=True)
def action(t): st.markdown(f"<div class='callout ca'>✅ <b>Recommended action</b> — {t}</div>", unsafe_allow_html=True)
def interp(t): st.markdown(f"<div class='callout cm'>🧭 <b>Managerial interpretation</b> — {t}</div>", unsafe_allow_html=True)
def gate(msg): st.markdown(f"<div class='gate'>🔒 <b>Module unavailable for this dataset.</b><br>{msg}</div>", unsafe_allow_html=True)

def kpi_cards(cards):
    html="<div class='kpi-row'>"
    for c in cards:
        tone=c.get("tone","teal"); sub=""
        if c.get("sub"):
            cls="up" if c.get("up") is True else ("down" if c.get("up") is False else "")
            arrow="▲ " if c.get("up") is True else ("▼ " if c.get("up") is False else "")
            sub=f"<div class='sub {cls}'>{arrow}{c['sub']}</div>"
        html+=f"<div class='kpi t-{tone}'><div class='lbl'>{c['label']}</div><div class='val'>{c['value']}</div>{sub}</div>"
    st.markdown(html+"</div>", unsafe_allow_html=True)

# ---- pure-python gradient styler (no matplotlib) ----
def _hex_lerp(c1,c2,t):
    c1,c2=c1.lstrip("#"),c2.lstrip("#")
    a=[int(c1[i:i+2],16) for i in (0,2,4)]; b=[int(c2[i:i+2],16) for i in (0,2,4)]
    r,g,bl=(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3)); return f"#{r:02x}{g:02x}{bl:02x}"
def _scol(t,stops):
    t=0.0 if t<0 else 1.0 if t>1 else t; n=len(stops)-1
    if n==0: return stops[0]
    seg=t*n; i=min(int(seg),n-1); return _hex_lerp(stops[i],stops[i+1],seg-i)
def style_grad(data,stops,axis=0,subset=None,precision=1,na_rep="—"):
    df=data.copy(); cols=list(subset) if subset is not None else list(df.columns)
    num=df[cols].apply(pd.to_numeric,errors="coerce")
    def css(d):
        out=pd.DataFrame("",index=d.index,columns=d.columns)
        if axis is None:
            vals=num.to_numpy(dtype=float)
            if np.isfinite(vals).any():
                vmin,vmax=np.nanmin(vals),np.nanmax(vals); rng=vmax-vmin
                for c in cols:
                    for idx in d.index:
                        v=num.at[idx,c]
                        if pd.notna(v) and rng>0: out.at[idx,c]=f"background-color:{_scol((v-vmin)/rng,stops)}"
        else:
            for c in cols:
                col=num[c]; vmin,vmax=col.min(),col.max(); rng=vmax-vmin
                for idx in d.index:
                    v=col.at[idx]
                    if pd.notna(v) and rng and rng>0: out.at[idx,c]=f"background-color:{_scol((v-vmin)/rng,stops)}"
        return out
    sty=df.style.apply(css,axis=None)
    fmt={c:(lambda x,p=precision: x if isinstance(x,str) else f"{x:,.{p}f}") for c in cols}
    return sty.format(fmt,na_rep=na_rep)

def aed(x):
    try: x=float(x)
    except: return "—"
    if abs(x)>=1e9: return f"AED {x/1e9:.2f}B"
    if abs(x)>=1e6: return f"AED {x/1e6:.2f}M"
    if abs(x)>=1e3: return f"AED {x/1e3:.1f}K"
    return f"AED {x:,.0f}"

# =========================================================================================
# CLEANING + DATA QUALITY
# =========================================================================================
SNAPSHOT=pd.Timestamp("2026-06-27")
FLAG_COLS=["Uses_Warehousing","Uses_Same_Day_Delivery","Uses_Next_Day_Delivery","Uses_Reverse_Logistics",
    "Uses_COD_Collection","Uses_International_Shipping","Uses_Qcommerce_Fulfillment",
    "Uses_Inventory_Management","Uses_Custom_Packaging"]
NUMERIC_COLS=["Company_Age_Years","Number_of_Sales_Channels","Monthly_Order_Volume","Avg_Order_Value_AED",
    "Number_of_SKUs","Total_Lifetime_Orders","Return_Rate_Pct","COD_Order_Pct","Avg_Delivery_Time_Days",
    "Peak_Season_Multiplier","Monthly_Logistics_Spend_AED","Recency_Days","Order_Frequency_Per_Month",
    "Customer_Lifetime_Value_AED","On_Time_Delivery_Rate_Pct","Order_Accuracy_Pct","Avg_Support_Response_Hours",
    "Complaints_Last_Quarter","Damaged_Shipment_Pct","Satisfaction_Score","NPS_Rating","Willingness_To_Switch"]
CAT_COLS=["Industry_Category","Company_Size","Emirate","Primary_Sales_Channel","Current_Provider",
    "Contract_Type","Preferred_Billing_Cycle","Tech_Integration_Level","Price_Sensitivity","Primary_Pain_Point"]
BOUNDS={"Satisfaction_Score":(1,10),"NPS_Rating":(0,10),"Willingness_To_Switch":(1,5),
    "On_Time_Delivery_Rate_Pct":(0,100),"Order_Accuracy_Pct":(0,100),"Return_Rate_Pct":(0,100),
    "COD_Order_Pct":(0,100),"Damaged_Shipment_Pct":(0,100)}
NON_NEG=["Monthly_Order_Volume","Company_Age_Years","Number_of_SKUs","Avg_Order_Value_AED",
    "Monthly_Logistics_Spend_AED","Total_Lifetime_Orders"]
EMIRATE_MAP={"dubai":"Dubai","dxb":"Dubai","abu dhabi":"Abu Dhabi","auh":"Abu Dhabi","sharjah":"Sharjah",
    "shj":"Sharjah","ras al khaimah":"Ras Al Khaimah","rak":"Ras Al Khaimah","ajman":"Ajman",
    "fujairah":"Fujairah","umm al quwain":"Umm Al Quwain"}
PROVIDER_MAP={"aramex":"Aramex","imile":"iMile","i-mile":"iMile","in house":"In-house","inhouse":"In-house",
    "in-house":"In-house","fetchr":"Fetchr","quiqup":"Quiqup","smsa express":"SMSA Express","shipa":"Shipa","none":"None"}
BOOL_TRUE={"1","1.0","yes","y","true","t"}; BOOL_FALSE={"0","0.0","no","n","false","f"}
import re
def _to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)): return float(x)
    s=str(x).lower().replace("aed","").replace(",","").strip()
    m=re.search(r"-?\d+\.?\d*",s); return float(m.group()) if m else np.nan
def _to_bool(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().lower()
    return 1 if s in BOOL_TRUE else 0 if s in BOOL_FALSE else np.nan
def _norm(x,mp=None):
    if pd.isna(x): return np.nan
    s=re.sub(r"\s+"," ",str(x)).strip()
    return (mp.get(s.lower(), s.title() if s else np.nan)) if mp is not None else s
def _dates(s): return pd.to_datetime(s,errors="coerce",dayfirst=True,format="mixed")

def clean_data(raw):
    df=raw.copy(); rep=[]
    if "Customer_ID" in df: df["Customer_ID"]=df["Customer_ID"].astype(str).str.strip()
    b=len(df); df=df.drop_duplicates()
    if "Customer_ID" in df: df=df.drop_duplicates(subset="Customer_ID",keep="first")
    rep.append(("Removed duplicate rows / IDs", b-len(df)))
    if "Emirate" in df: df["Emirate"]=df["Emirate"].apply(lambda v:_norm(v,EMIRATE_MAP))
    if "Current_Provider" in df: df["Current_Provider"]=df["Current_Provider"].apply(lambda v:_norm(v,PROVIDER_MAP))
    for c in ["Industry_Category","Company_Size","Primary_Sales_Channel","Contract_Type",
              "Preferred_Billing_Cycle","Tech_Integration_Level","Price_Sensitivity","Primary_Pain_Point"]:
        if c in df: df[c]=df[c].apply(_norm)
    rep.append(("Standardised category labels & trimmed text","applied"))
    fx=0
    for c in FLAG_COLS:
        if c in df:
            o=df[c].copy(); df[c]=df[c].apply(_to_bool); fx+=int((o.astype(str)!=df[c].astype(str)).sum())
    rep.append(("Normalised service flags to 0/1",fx))
    for c in NUMERIC_COLS:
        if c in df: df[c]=df[c].apply(_to_num)
    rep.append(("Parsed numbers from text/currency","applied"))
    if "Return_Rate_Pct" in df:
        fr=df["Return_Rate_Pct"].between(0,1,inclusive="right") & (df["Return_Rate_Pct"]>0)
        df.loc[fr,"Return_Rate_Pct"]*=100; rep.append(("Rescaled fractional Return_Rate (0.xx→%)",int(fr.sum())))
    nf=0
    for c in NON_NEG:
        if c in df:
            neg=df[c]<0; nf+=int(neg.sum()); df.loc[neg,c]=df.loc[neg,c].abs()
    rep.append(("Fixed impossible negatives",nf))
    oor=0
    for c,(lo,hi) in BOUNDS.items():
        if c in df:
            bad=~df[c].between(lo,hi)&df[c].notna(); oor+=int(bad.sum()); df.loc[bad,c]=np.nan
    rep.append(("Out-of-range values set to NaN",oor))
    for c in ["Signup_Date","Last_Order_Date"]:
        if c in df: df[c]=_dates(df[c])
    ni=0
    for c in NUMERIC_COLS:
        if c in df and df[c].isna().any(): ni+=int(df[c].isna().sum()); df[c]=df[c].fillna(df[c].median())
    for c in FLAG_COLS:
        if c in df and df[c].isna().any():
            df[c]=df[c].fillna(df[c].mode(dropna=True).iloc[0] if df[c].notna().any() else 0).astype(int)
    ci=0
    for c in CAT_COLS:
        if c in df and df[c].isna().any():
            ci+=int(df[c].isna().sum()); m=df[c].mode(dropna=True); df[c]=df[c].fillna(m.iloc[0] if len(m) else "Unknown")
    rep.append(("Imputed missing (median / mode)",f"{ni} num / {ci} cat"))
    if "Signup_Date" in df:
        df["Signup_Month"]=df["Signup_Date"].dt.to_period("M").dt.to_timestamp()
        df["Tenure_Months"]=((SNAPSHOT-df["Signup_Date"]).dt.days/30.44).round(1)
    if "Signup_Date" in df and "Last_Order_Date" in df:
        df["Active_Until_Months"]=((df["Last_Order_Date"]-df["Signup_Date"]).dt.days/30.44).clip(lower=0)
    for c in ["Churned","Premium_Tier_Adoption"]:
        if c in df: df[c]=df[c].astype(str).str.strip().str.title()
    # engineered features
    if {"Monthly_Logistics_Spend_AED","Monthly_Order_Volume"}.issubset(df.columns):
        df["Spend_per_Order"]=(df["Monthly_Logistics_Spend_AED"]/df["Monthly_Order_Volume"].replace(0,np.nan)).round(2)
        df["Spend_per_Order"]=df["Spend_per_Order"].fillna(df["Spend_per_Order"].median())
    if set(FLAG_COLS).issubset(df.columns):
        df["Service_Count"]=df[FLAG_COLS].sum(axis=1).astype(int)
    rep.append(("Engineered features (Spend_per_Order, Service_Count)","added"))
    return df.reset_index(drop=True), rep

def data_quality(raw, clean):
    n=len(raw)
    dtypes=pd.DataFrame({"Column":clean.columns,
        "Type":[str(clean[c].dtype) for c in clean.columns],
        "Missing_raw":[int(raw[c].isna().sum()) if c in raw else 0 for c in clean.columns],
        "Unique":[int(clean[c].nunique()) for c in clean.columns]})
    completeness=100*(1-raw.isna().sum().sum()/(raw.shape[0]*raw.shape[1]))
    dups=int(raw.duplicated().sum())
    # outlier summary (IQR) on numeric
    outs=[]
    for c in NUMERIC_COLS:
        if c in clean:
            q1,q3=clean[c].quantile([.25,.75]); iqr=q3-q1
            k=int(((clean[c]<q1-1.5*iqr)|(clean[c]>q3+1.5*iqr)).sum())
            if k>0: outs.append({"Column":c,"Outliers":k})
    return dict(completeness=completeness,dups=dups,missing=int(raw.isna().sum().sum()),
                dtypes=dtypes,outliers=pd.DataFrame(outs))

# =========================================================================================
# FEASIBILITY (drives module gating)
# =========================================================================================
def feasibility(df):
    n=len(df); f={}
    has_dates="Signup_Month" in df and df["Signup_Month"].notna().sum()>12
    f["kpi"]=("ok","")
    f["demographics"]=("adapt","No personal demographics (Age/Gender/Income/Occupation) in a B2B dataset — "
                       "adapted to firmographics: Industry, Company Size, Emirate, tenure.")
    f["trend"]=("ok" if has_dates else "gate",
                "" if has_dates else "No usable date field, so acquisition/cohort trends can't be built.")
    f["cohort"]=("ok" if has_dates else "gate","" if has_dates else "Requires signup & last-order dates.")
    f["diagnostic"]=("ok","")
    f["kmeans"]=("ok" if n>=50 else "gate","" if n>=50 else "Too few rows for stable clusters.")
    f["lca"]=("ok" if set(FLAG_COLS).issubset(df.columns) else "gate",
              "" if set(FLAG_COLS).issubset(df.columns) else "Needs binary indicator variables.")
    has_churn = "Churned" in df and df["Churned"].nunique()==2
    f["classification"]=("ok" if has_churn else "gate",
              "" if has_churn else "Needs a binary target (e.g., Churned / Premium adoption).")
    f["regression"]=("ok" if "Customer_Lifetime_Value_AED" in df else "gate",
              "" if "Customer_Lifetime_Value_AED" in df else "Needs a continuous target such as CLV.")
    f["basket"]=("adapt" if set(FLAG_COLS).issubset(df.columns) else "gate",
              "No product-level baskets in this survey — adapted to service-bundle affinity across the 9 logistics services."
              if set(FLAG_COLS).issubset(df.columns) else "No item/basket data available.")
    f["forecast"]=("ok" if has_dates else "gate","" if has_dates else "Needs a time field.")
    f["reco"]=("ok" if set(FLAG_COLS).issubset(df.columns) else "gate",
              "" if set(FLAG_COLS).issubset(df.columns) else "Needs a customer×item usage matrix.")
    f["advisor"]=("ok","")
    return f

# =========================================================================================
# ANALYTICS HELPERS
# =========================================================================================
SCALERS={"Standard":StandardScaler,"MinMax":MinMaxScaler,"Robust":RobustScaler}
CLUSTER_FEATURES=["Monthly_Order_Volume","Avg_Order_Value_AED","Number_of_SKUs","Monthly_Logistics_Spend_AED",
    "Number_of_Sales_Channels","Recency_Days","Order_Frequency_Per_Month","Return_Rate_Pct","Satisfaction_Score"]

def scale_X(df,feats,scaler): return SCALERS[scaler]().fit_transform(df[list(feats)].astype(float).values)

@st.cache_data(show_spinner=False)
def kmeans_sweep(df,feats,scaler):
    X=scale_X(df,feats,scaler); ks=list(range(2,9)); inertia=[];sil=[];dbi=[]
    for k in ks:
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X,km.labels_)); dbi.append(davies_bouldin_score(X,km.labels_))
    return ks,inertia,sil,dbi

def rfm(df):
    r=df.copy()
    r["R"]=pd.qcut(r["Recency_Days"].rank(method="first"),5,labels=[5,4,3,2,1]).astype(int)
    r["F"]=pd.qcut(r["Order_Frequency_Per_Month"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
    r["M"]=pd.qcut(r["Monthly_Logistics_Spend_AED"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
    r["RFM_Score"]=r["R"]+r["F"]+r["M"]
    def seg(s):
        s=s["RFM_Score"]
        return ("Champions" if s>=13 else "Loyal" if s>=11 else "Potential Loyalist" if s>=9
                else "Needs Attention" if s>=7 else "At Risk" if s>=5 else "Hibernating")
    r["RFM_Segment"]=r.apply(seg,axis=1); return r

def cohort_matrix(df,max_c=18,max_m=12):
    d=df.dropna(subset=["Signup_Month","Tenure_Months","Active_Until_Months"])
    if d.empty: return None
    cs=sorted(d["Signup_Month"].unique())[-max_c:]; rows={}
    for c in cs:
        sub=d[d["Signup_Month"]==c]; vals=[]
        for m in range(max_m+1):
            den=(sub["Tenure_Months"]>=m).sum()
            vals.append(np.nan if den==0 else round(((sub["Tenure_Months"]>=m)&(sub["Active_Until_Months"]>=m)).sum()/den*100,0))
        rows[pd.Timestamp(c).strftime("%Y-%m")]=vals
    return pd.DataFrame(rows,index=[f"M{m}" for m in range(max_m+1)]).T

# ---- LCA: Bernoulli mixture (pure numpy) for binary indicators ----
def lca_fit(X,k,iters=150,seed=0):
    rng=np.random.default_rng(seed); n,d=X.shape
    pi=np.ones(k)/k; p=rng.uniform(0.25,0.75,(k,d)); ll_old=-np.inf; r=None
    for _ in range(iters):
        logp=X@np.log(p.T+1e-9)+(1-X)@np.log(1-p.T+1e-9)+np.log(pi+1e-9)
        m=logp.max(1,keepdims=True); ex=np.exp(logp-m); r=ex/ex.sum(1,keepdims=True)
        ll=(m+np.log(ex.sum(1,keepdims=True))).sum()
        Nk=r.sum(0); pi=Nk/n; p=np.clip((r.T@X)/Nk[:,None],1e-3,1-1e-3)
        if abs(ll-ll_old)<1e-4: break
        ll_old=ll
    npar_=k-1+k*d; bic=-2*ll_old+npar_*np.log(n); aic=-2*ll_old+2*npar_
    return dict(pi=pi,p=p,resp=r,ll=ll_old,bic=bic,aic=aic,labels=r.argmax(1))

@st.cache_data(show_spinner=False)
def lca_select(df,krange=(2,3,4,5)):
    X=df[FLAG_COLS].astype(int).values.astype(float); out=[]
    for k in krange:
        best=min((lca_fit(X,k,seed=s) for s in range(4)),key=lambda m:m["bic"])
        out.append((k,best["bic"],best["aic"]))
    bestk=min(out,key=lambda t:t[1])[0]
    model=min((lca_fit(X,bestk,seed=s) for s in range(6)),key=lambda m:m["bic"])
    return out,bestk,model

# ---- association rules over service flags (custom apriori) ----
@st.cache_data(show_spinner=False)
def assoc_rules(df,min_support=0.05,min_conf=0.3,max_ante=2):
    items=FLAG_COLS; M=df[items].astype(int).values; n=len(df)
    lab={c:c.replace("Uses_","").replace("_"," ") for c in items}
    def sup(idx): 
        return (M[:,idx].sum(1)==len(idx)).mean() if len(idx)>1 else M[:,idx[0]].mean()
    single={(i,):M[:,i].mean() for i in range(len(items))}
    rules=[]
    for a in range(1,max_ante+1):
        for ante in combinations(range(len(items)),a):
            sa=sup(list(ante))
            if sa<min_support: continue
            for cons in range(len(items)):
                if cons in ante: continue
                sab=sup(list(ante)+[cons])
                if sab<min_support: continue
                conf=sab/sa; 
                if conf<min_conf: continue
                sb=single[(cons,)]; lift=conf/sb if sb>0 else np.nan
                conv=(1-sb)/(1-conf) if conf<1 else np.inf
                rules.append({"Antecedent":" + ".join(lab[items[i]] for i in ante),
                    "Consequent":lab[items[cons]],"Support":round(sab,3),"Confidence":round(conf,3),
                    "Lift":round(lift,3),"Conviction":round(conv,2) if np.isfinite(conv) else 99.0})
    rd=pd.DataFrame(rules)
    if not rd.empty: rd=rd.sort_values("Lift",ascending=False).reset_index(drop=True)
    return rd

def build_prep(df,target,scaler,num_drop=()):
    num=[c for c in NUMERIC_COLS+FLAG_COLS if c in df and c not in num_drop and c!=target]
    if "Service_Count" in df and "Service_Count" not in num: num.append("Service_Count")
    if "Spend_per_Order" in df and "Spend_per_Order" not in num: num.append("Spend_per_Order")
    cat=[c for c in CAT_COLS if c in df and c!=target]
    pre=ColumnTransformer([("num",SCALERS[scaler](),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
    return pre,num,cat

def auto_scaler(df,feats):
    # choose scaler by average absolute skew: heavy skew -> Robust, moderate -> Standard, else MinMax
    sk=np.nanmean([abs(pd.Series(df[c]).skew()) for c in feats if c in df])
    return "Robust" if sk>2 else "Standard" if sk>0.75 else "MinMax"

@st.cache_data(show_spinner=True)
def run_classification(df,target,scaler,tune,test_size,do_cv):
    y=(df[target]=="Yes").astype(int).values
    pre,num,cat=build_prep(df,target,scaler); Xdf=df[num+cat]
    Xtr,Xte,ytr,yte=train_test_split(Xdf,y,test_size=test_size,stratify=y,random_state=42)
    pre.fit(Xtr); Xtr_t=pre.transform(Xtr); Xte_t=pre.transform(Xte)
    if hasattr(Xtr_t,"toarray"): Xtr_t,Xte_t=Xtr_t.toarray(),Xte_t.toarray()
    feat_names=list(pre.get_feature_names_out())
    models={"KNN":(KNeighborsClassifier(),{"n_neighbors":[5,11,21]}),
        "Decision Tree":(DecisionTreeClassifier(random_state=42),{"max_depth":[4,6,10]}),
        "Random Forest":(RandomForestClassifier(random_state=42,n_estimators=200),{"max_depth":[None,10]}),
        "Gradient Boosting":(GradientBoostingClassifier(random_state=42),{"learning_rate":[0.05,0.1]}),
        "Logistic Regression":(LogisticRegression(max_iter=2000),{"C":[0.1,1,10]})}
    if HAS_XGB:
        models["XGBoost"]=(XGBClassifier(eval_metric="logloss",use_label_encoder=False,random_state=42,verbosity=0),
                           {"max_depth":[3,5]})
    rows=[];roc={};fitted={}
    for name,(mdl,grid) in models.items():
        est=GridSearchCV(mdl,grid,cv=3,scoring="f1",n_jobs=-1).fit(Xtr_t,ytr).best_estimator_ if tune else mdl.fit(Xtr_t,ytr)
        fitted[name]=est
        ptr,pte=est.predict(Xtr_t),est.predict(Xte_t); proba=est.predict_proba(Xte_t)[:,1]
        cv=cross_val_score(est,Xtr_t,ytr,cv=5,scoring="f1").mean() if do_cv else np.nan
        rows.append({"Model":name,"Train Acc":round(accuracy_score(ytr,ptr),3),"Test Acc":round(accuracy_score(yte,pte),3),
            "Precision":round(precision_score(yte,pte,zero_division=0),3),"Recall":round(recall_score(yte,pte,zero_division=0),3),
            "F1":round(f1_score(yte,pte,zero_division=0),3),"ROC-AUC":round(roc_auc_score(yte,proba),3),
            "CV-F1":round(cv,3) if do_cv else None})
        fpr,tpr,_=roc_curve(yte,proba); roc[name]=(fpr.tolist(),tpr.tolist())
    res=pd.DataFrame(rows).sort_values("ROC-AUC",ascending=False).reset_index(drop=True)
    best=res.iloc[0]["Model"]; est=fitted[best]; pte=est.predict(Xte_t); proba=est.predict_proba(Xte_t)[:,1]
    cm=confusion_matrix(yte,pte).tolist()
    frac_pos,mean_pred=calibration_curve(yte,proba,n_bins=8,strategy="quantile")
    if hasattr(est,"feature_importances_"): imp=est.feature_importances_
    elif hasattr(est,"coef_"): imp=np.abs(est.coef_[0])
    else:
        imp=permutation_importance(est,Xte_t,yte,n_repeats=5,random_state=42).importances_mean
    importances=sorted(zip(feat_names,imp),key=lambda x:x[1],reverse=True)[:15]
    Xall=pre.transform(Xdf); Xall=Xall.toarray() if hasattr(Xall,"toarray") else Xall
    lr=LogisticRegression(max_iter=2000).fit(Xtr_t,ytr); prop=lr.predict_proba(Xall)[:,1]
    return (res.to_dict("records"),roc,cm,importances,best,prop.tolist(),
            list(frac_pos),list(mean_pred))

@st.cache_data(show_spinner=True)
def run_regression(df,target,alpha):
    pre,num,cat=build_prep(df,"__none__",scaler="Standard",num_drop=[target]); Xdf=df[num+cat]
    y=df[target].values
    Xtr,Xte,ytr,yte=train_test_split(Xdf,y,test_size=0.25,random_state=42)
    pre.fit(Xtr); Xtr_t=pre.transform(Xtr); Xte_t=pre.transform(Xte)
    if hasattr(Xtr_t,"toarray"): Xtr_t,Xte_t=Xtr_t.toarray(),Xte_t.toarray()
    names=list(pre.get_feature_names_out())
    models={"Linear":LinearRegression(),"Ridge":Ridge(alpha=alpha),"Lasso":Lasso(alpha=alpha,max_iter=5000),
            "ElasticNet":ElasticNet(alpha=alpha,l1_ratio=0.5,max_iter=5000)}
    rows={};coefs={};preds={}
    p=Xtr_t.shape[1]
    for nm,md in models.items():
        md.fit(Xtr_t,ytr); pr=md.predict(Xte_t); preds[nm]=pr
        r2=r2_score(yte,pr); n=len(yte)
        adj=1-(1-r2)*(n-1)/(n-p-1) if n-p-1>0 else np.nan
        rmse=mean_squared_error(yte,pr)**0.5; mae=mean_absolute_error(yte,pr)
        mape=np.mean(np.abs((yte-pr)/np.where(yte==0,np.nan,yte)))*100
        rows[nm]={"R2":round(r2,3),"Adj_R2":round(adj,3),"MAE":round(mae,0),"RMSE":round(rmse,0),"MAPE_%":round(mape,1)}
        coefs[nm]=dict(zip(names,getattr(md,"coef_",np.zeros(p))))
    # lasso path on a compact standardized numeric subset
    from sklearn.preprocessing import StandardScaler as SS
    sub=[c for c in num][:10]; Xs=SS().fit_transform(df[sub].values); ys=(y-y.mean())/y.std()
    alphas,coefs_path,_=lasso_path(Xs,ys,n_alphas=30)
    metrics=pd.DataFrame(rows).T.reset_index().rename(columns={"index":"Model"})
    # coefficient importance from Ridge
    ridge_coef=pd.Series(coefs["Ridge"]).sort_values(key=np.abs,ascending=False).head(12)
    return dict(metrics=metrics,yte=list(yte),preds={k:list(v) for k,v in preds.items()},
                alphas=list(alphas),coefs_path=coefs_path.tolist(),path_feats=sub,
                ridge_coef=ridge_coef.to_dict())

@st.cache_data(show_spinner=False)
def recommender(df):
    services=FLAG_COLS; M=df[services].astype(float).values
    nmf=NMF(n_components=4,init="nndsvda",random_state=42,max_iter=400)
    W=nmf.fit_transform(M); H=nmf.components_; R=W@H
    lab={s:s.replace("Uses_","").replace("_"," ") for s in services}
    # item-item cosine
    Mn=M/ (np.linalg.norm(M,axis=0,keepdims=True)+1e-9)
    item_sim=Mn.T@Mn
    idx={c:i for i,c in enumerate(df["Customer_ID"].tolist())}
    return services,lab,M,R,item_sim,idx

def arima_forecast(s,order,horizon):
    from statsmodels.tsa.arima.model import ARIMA
    model=ARIMA(s,order=order).fit(); fc=model.get_forecast(steps=horizon)
    return model,fc.predicted_mean,fc.conf_int()

# =========================================================================================
# PAGES
# =========================================================================================
def _breakdown(df,dim,meas):
    if meas=="Merchants": a=df[dim].value_counts().reset_index(); a.columns=[dim,"Value"]
    elif meas=="Monthly spend": a=df.groupby(dim)["Monthly_Logistics_Spend_AED"].sum().sort_values(ascending=False).reset_index(); a.columns=[dim,"Value"]
    else: a=df.groupby(dim)["Monthly_Order_Volume"].sum().sort_values(ascending=False).reset_index(); a.columns=[dim,"Value"]
    return a

def page_overview(df,feas):
    page_header("🏠","Executive Overview","How healthy is the business right now, and where is value concentrated?")
    churn=(df["Churned"]=="Yes").mean()*100; ret=100-churn
    prom=(df["NPS_Rating"]>=9).mean()*100; det=(df["NPS_Rating"]<=6).mean()*100; nps=prom-det
    prem=(df["Premium_Tier_Adoption"]=="Yes").mean()*100
    active=(df["Recency_Days"]<=60).mean()*100
    kpi_cards([
        {"label":"Total Respondents","value":f"{len(df):,}","tone":"teal"},
        {"label":"Avg Satisfaction","value":f"{df['Satisfaction_Score'].mean():.1f}/10","tone":"violet"},
        {"label":"Avg Monthly Spend","value":aed(df['Monthly_Logistics_Spend_AED'].mean()),"tone":"amber"},
        {"label":"Purchase Frequency","value":f"{df['Order_Frequency_Per_Month'].mean():.1f}/mo","tone":"sky"},
        {"label":"Avg CLV","value":aed(df['Customer_Lifetime_Value_AED'].mean()),"tone":"indigo"},
        {"label":"Net Promoter Score","value":f"{nps:+.0f}","tone":"green" if nps>=0 else "red"},
        {"label":"Retention Rate","value":f"{ret:.1f}%","tone":"green","sub":f"churn {churn:.1f}%","up":ret>=80},
        {"label":"Recently Active","value":f"{active:.0f}%","tone":"teal","sub":"ordered ≤60d"},
    ])
    dq=st.session_state.get("_dq")
    if dq:
        with st.expander("🧼 Data Quality Report (auto-generated before analysis)"):
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Completeness",f"{dq['completeness']:.1f}%"); c2.metric("Missing cells",f"{dq['missing']:,}→0")
            c3.metric("Duplicates removed",f"{dq['dups']}"); c4.metric("Columns",f"{len(dq['dtypes'])}")
            st.dataframe(dq["dtypes"],use_container_width=True,hide_index=True,height=240)

    if feas["trend"][0]!="gate":
        section("Trend Analysis","New-merchant acquisition over time (transaction-level sales history isn't in the survey).")
        c1,c2=st.columns([1,3])
        with c1:
            met=st.selectbox("Metric",["New merchants","Cumulative merchants","Onboarded spend (AED)"],key="ov_m")
            style=st.radio("Style",["Area","Line","Bars"],horizontal=True,key="ov_s")
        g=df.dropna(subset=["Signup_Month"]).groupby("Signup_Month")
        s=g.size() if met=="New merchants" else (g.size().cumsum() if met=="Cumulative merchants" else g["Monthly_Logistics_Spend_AED"].sum())
        ts=s.reset_index(); ts.columns=["Month","Value"]
        fig=(px.area(ts,x="Month",y="Value",markers=True) if style=="Area" else
             px.line(ts,x="Month",y="Value",markers=True) if style=="Line" else px.bar(ts,x="Month",y="Value"))
        fig.update_traces(line_color=TEAL) if style=="Line" else fig.update_traces(marker_color=TEAL)
        fig.update_layout(height=320)
        with c2: st.plotly_chart(fig,use_container_width=True)
        growth=(ts["Value"].iloc[-1]/max(ts["Value"].iloc[-4],1)-1)*100 if len(ts)>4 and met!="Cumulative merchants" else np.nan
        if np.isfinite(growth): insight(f"Latest-month {met.lower()} is <b>{growth:+.0f}%</b> vs three months prior.")

    section("Distribution &amp; Concentration")
    c1,c2,c3=st.columns([1,2,2])
    with c1:
        dim=st.selectbox("Dimension",["Industry_Category","Emirate","Company_Size","Current_Provider","Contract_Type"],key="ov_dim")
        meas=st.selectbox("Measure",["Merchants","Monthly spend","Monthly orders"],key="ov_meas")
    a=_breakdown(df,dim,meas)
    bar=px.bar(a,x="Value",y=dim,orientation="h",color="Value",color_continuous_scale=["#9ee7df",TEAL,NAVY])
    bar.update_layout(height=340,yaxis={"categoryorder":"total ascending"},coloraxis_showscale=False,yaxis_title=None)
    with c2: st.plotly_chart(bar,use_container_width=True)
    donut=px.pie(a,names=dim,values="Value",hole=.58,color_discrete_sequence=SEQ)
    donut.update_traces(textposition="inside",textinfo="percent"); donut.update_layout(height=340)
    with c3: st.plotly_chart(donut,use_container_width=True)

    section("Pareto — Revenue Concentration (80/20)")
    pg=df.groupby("Customer_ID")["Monthly_Logistics_Spend_AED"].sum().sort_values(ascending=False).reset_index()
    pg["cum"]=pg["Monthly_Logistics_Spend_AED"].cumsum()/pg["Monthly_Logistics_Spend_AED"].sum()*100
    top20=int(len(pg)*0.2); share=pg["cum"].iloc[top20-1] if top20>0 else 0
    fig=go.Figure()
    fig.add_bar(x=list(range(1,len(pg)+1)),y=pg["Monthly_Logistics_Spend_AED"],marker_color=INDIGO,name="Spend")
    fig.add_trace(go.Scatter(x=list(range(1,len(pg)+1)),y=pg["cum"],yaxis="y2",line=dict(color=AMBER,width=3),name="Cumulative %"))
    fig.add_hline(y=80,yref="y2",line=dict(color=RED,dash="dash"))
    fig.update_layout(height=330,yaxis2=dict(overlaying="y",side="right",range=[0,105]),
                      xaxis_title="Merchants (ranked by spend)",legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig,use_container_width=True)
    insight(f"The top 20% of merchants generate <b>{share:.0f}%</b> of monthly logistics revenue.")
    action("Protect and grow the top-20% cohort with named account management; they concentrate most revenue and churn risk.")

    if feas["cohort"][0]!="gate":
        section("Cohort Retention")
        coh=cohort_matrix(df)
        if coh is not None and not coh.empty:
            fig=px.imshow(coh,color_continuous_scale=["#fee2e2","#fde68a",TEAL,NAVY],aspect="auto",
                          labels=dict(x="Months since signup",y="Cohort",color="Retention %"))
            fig.update_layout(height=380); fig.update_xaxes(side="top")
            st.plotly_chart(fig,use_container_width=True)

    section("Cross-Tabulation")
    dims=["Company_Size","Industry_Category","Emirate","Contract_Type","Tech_Integration_Level","Price_Sensitivity","Current_Provider"]
    x1,x2,x3=st.columns(3)
    rdim=x1.selectbox("Rows",dims,0,key="ov_ctr"); cdim=x2.selectbox("Columns",dims,3,key="ov_ctc")
    val=x3.selectbox("Value",["Count","Churn rate %","Avg monthly spend","Avg satisfaction"],key="ov_ctv")
    if rdim!=cdim:
        if val=="Count": ct=pd.crosstab(df[rdim],df[cdim])
        elif val=="Churn rate %": ct=pd.crosstab(df[rdim],df[cdim],values=(df["Churned"]=="Yes"),aggfunc="mean").round(3)*100
        elif val=="Avg monthly spend": ct=pd.crosstab(df[rdim],df[cdim],values=df["Monthly_Logistics_Spend_AED"],aggfunc="mean").round(0)
        else: ct=pd.crosstab(df[rdim],df[cdim],values=df["Satisfaction_Score"],aggfunc="mean").round(2)
        st.dataframe(style_grad(ct,["#e8f6f1",TEAL,NAVY],axis=None,precision=1),use_container_width=True)

    section("Auto-Generated Executive Summary")
    top_ind=df.groupby("Industry_Category")["Monthly_Logistics_Spend_AED"].sum().idxmax()
    top_em=df["Emirate"].value_counts().idxmax()
    worst_prov=df.groupby("Current_Provider")["Satisfaction_Score"].mean().idxmin()
    interp(f"The portfolio spans <b>{len(df):,}</b> D2C merchants, concentrated in <b>{top_em}</b>, with "
           f"<b>{top_ind}</b> the largest revenue category. Retention is <b>{ret:.0f}%</b> and NPS is <b>{nps:+.0f}</b>. "
           f"Revenue is highly concentrated (top-20% = {share:.0f}%). Merchants on <b>{worst_prov}</b> report the lowest satisfaction.")

def page_customer(df,feas):
    page_header("👥","Customer Intelligence","Who are our customers and how do they behave?")
    if feas["demographics"][0]=="adapt":
        st.markdown(f"<div class='callout cm'>ℹ️ {feas['demographics'][1]}</div>",unsafe_allow_html=True)
    section("Firmographic Breakdown")
    cols=st.columns(4)
    for (d,t),c in zip([("Company_Size","indigo"),("Industry_Category","teal"),("Emirate","amber"),("Tech_Integration_Level","violet")],cols):
        a=df[d].value_counts().reset_index(); a.columns=[d,"n"]
        fig=px.pie(a,names=d,values="n",hole=.6,color_discrete_sequence=SEQ); fig.update_traces(textinfo="percent")
        fig.update_layout(height=230,margin=dict(t=30,b=0,l=0,r=0),title=d.replace("_"," "),showlegend=False)
        c.plotly_chart(fig,use_container_width=True)

    section("Behaviour Analysis")
    b1,b2=st.columns(2)
    with b1:
        fig=px.histogram(df,x="Monthly_Order_Volume",nbins=40,color_discrete_sequence=[TEAL])
        fig.update_layout(height=280,title="Purchase behaviour — monthly order volume"); st.plotly_chart(fig,use_container_width=True)
    with b2:
        fig=px.histogram(df,x="Monthly_Logistics_Spend_AED",nbins=40,color_discrete_sequence=[INDIGO])
        fig.update_layout(height=280,title="Spending behaviour — monthly logistics spend"); st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        ch=df["Primary_Sales_Channel"].value_counts().reset_index(); ch.columns=["Channel","n"]
        fig=px.bar(ch,x="n",y="Channel",orientation="h",color="n",color_continuous_scale=["#cdeee8",TEAL,NAVY])
        fig.update_layout(height=300,title="Channel preference",coloraxis_showscale=False,yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        adopt=(df[FLAG_COLS].mean()*100).sort_values().reset_index(); adopt.columns=["Service","Adoption %"]
        adopt["Service"]=adopt["Service"].str.replace("Uses_","").str.replace("_"," ")
        fig=px.bar(adopt,x="Adoption %",y="Service",orientation="h",color="Adoption %",color_continuous_scale=["#fde68a",AMBER,RED])
        fig.update_layout(height=300,title="Service preference (adoption rate)",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

    section("Customer Journey Funnel")
    stages=["Signed up","Active (≤60d)","Frequent (≥ median)","Premium tier","RFM Champions"]
    rf=rfm(df); med=df["Order_Frequency_Per_Month"].median()
    vals=[len(df),(df["Recency_Days"]<=60).sum(),(df["Order_Frequency_Per_Month"]>=med).sum(),
          (df["Premium_Tier_Adoption"]=="Yes").sum(),(rf["RFM_Segment"]=="Champions").sum()]
    fig=px.funnel(pd.DataFrame({"Stage":stages,"Merchants":vals}),x="Merchants",y="Stage",color_discrete_sequence=[TEAL])
    fig.update_layout(height=320); st.plotly_chart(fig,use_container_width=True)
    conv=vals[-1]/vals[0]*100
    interp(f"Only <b>{conv:.1f}%</b> of merchants reach 'Champion' status — the funnel narrows most between 'Active' and 'Frequent', "
           "pointing to an activation/engagement gap rather than an acquisition problem.")

    section("Customer Personas (RFM-derived)")
    persona=(rf.groupby("RFM_Segment").agg(Merchants=("Customer_ID","count"),
        Avg_Spend=("Monthly_Logistics_Spend_AED","mean"),Avg_Recency=("Recency_Days","mean"),
        Avg_Freq=("Order_Frequency_Per_Month","mean"),Churn=("Churned",lambda s:(s=="Yes").mean()*100)).round(1)
        .sort_values("Merchants",ascending=False))
    st.dataframe(style_grad(persona,["#e8f6f1",TEAL,NAVY],axis=0,subset=["Avg_Spend"]),use_container_width=True)

    section("Interactive Segmentation Explorer")
    e1,e2,e3=st.columns(3)
    xx=e1.selectbox("X",["Monthly_Logistics_Spend_AED","Monthly_Order_Volume","Satisfaction_Score","Recency_Days"],key="ce_x")
    yy=e2.selectbox("Y",["Order_Frequency_Per_Month","Avg_Order_Value_AED","NPS_Rating","Number_of_SKUs"],1,key="ce_y")
    cc=e3.selectbox("Colour",["Company_Size","Churned","Price_Sensitivity","Industry_Category"],key="ce_c")
    fig=px.scatter(df,x=xx,y=yy,color=cc,opacity=.65,color_discrete_sequence=SEQ)
    fig.update_layout(height=420); st.plotly_chart(fig,use_container_width=True)

def page_diagnostic(df,feas):
    page_header("🔬","Diagnostic Analytics","Why are churn, spend and satisfaction moving the way they are?")
    section("Deviation &amp; Variance vs Targets")
    tg={"On_Time_Delivery_Rate_Pct":("On-time",95,"%",True),"Order_Accuracy_Pct":("Accuracy",98,"%",True),
        "Avg_Delivery_Time_Days":("Delivery time",2.0," d",False),"Satisfaction_Score":("Satisfaction",8.0,"/10",True)}
    cols=st.columns(len(tg))
    for (col,(lbl,t,u,hi)),cc in zip(tg.items(),cols):
        a=df[col].mean(); d=a-t; good=(d>=0) if hi else (d<=0)
        cc.metric(lbl,f"{a:.1f}{u}",f"{d:+.1f} vs {t}{u}",delta_color="normal" if good else "inverse")
    c1,c2=st.columns(2)
    with c1:
        risk=(df[df["Churned"]=="Yes"].groupby("Company_Size")["Monthly_Logistics_Spend_AED"].sum()
              .reindex(["Startup","SME","Mid-Market","Enterprise"]).fillna(0))
        wf=go.Figure(go.Waterfall(orientation="v",measure=["relative"]*len(risk)+["total"],
            x=list(risk.index)+["Total at risk"],y=list(risk.values)+[0],
            increasing=dict(marker=dict(color=RED)),totals=dict(marker=dict(color=NAVY))))
        wf.update_layout(height=320,title="Monthly spend at risk from churn, by segment",yaxis_title="AED")
        st.plotly_chart(wf,use_container_width=True)
    with c2:
        mean_sat=df["Satisfaction_Score"].mean()
        dev=(df.groupby("Current_Provider")["Satisfaction_Score"].mean()-mean_sat).sort_values()
        fig=go.Figure(go.Bar(x=dev.values,y=dev.index,orientation="h",
            marker_color=[RED if v<0 else GREEN for v in dev.values]))
        fig.add_vline(x=0,line=dict(color="#94a3b8")); fig.update_layout(height=320,title="Provider satisfaction vs mean")
        st.plotly_chart(fig,use_container_width=True)

    section("Contribution &amp; Drill-down")
    d1,d2=st.columns(2)
    with d1:
        tm=px.treemap(df,path=[px.Constant("All"),"Emirate","Industry_Category","Company_Size"],
            values="Monthly_Logistics_Spend_AED",color="Monthly_Logistics_Spend_AED",
            color_continuous_scale=["#d1faf3",TEAL,NAVY]); tm.update_layout(height=400,margin=dict(t=30))
        st.plotly_chart(tm,use_container_width=True)
    with d2:
        sb=df.groupby(["Company_Size","Contract_Type"]).size().reset_index(name="n")
        fig=px.bar(sb,x="Company_Size",y="n",color="Contract_Type",barmode="stack",
            category_orders={"Company_Size":["Startup","SME","Mid-Market","Enterprise"]},color_discrete_sequence=SEQ)
        fig.update_layout(height=400,title="Contract mix by size"); st.plotly_chart(fig,use_container_width=True)

    section("Driver Analysis — Correlation &amp; Feature Importance")
    num=["Monthly_Order_Volume","Avg_Order_Value_AED","Monthly_Logistics_Spend_AED","Return_Rate_Pct",
         "On_Time_Delivery_Rate_Pct","Order_Accuracy_Pct","Avg_Support_Response_Hours","Complaints_Last_Quarter",
         "Damaged_Shipment_Pct","Satisfaction_Score","NPS_Rating","Recency_Days"]
    e1,e2=st.columns([3,2])
    with e1:
        corr=df[num].corr()
        hm=px.imshow(corr,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
        hm.update_layout(height=430,margin=dict(l=10,t=10)); st.plotly_chart(hm,use_container_width=True)
    with e2:
        Xd=pd.get_dummies(df[["Company_Size","Contract_Type","Price_Sensitivity","On_Time_Delivery_Rate_Pct",
            "Complaints_Last_Quarter","Satisfaction_Score","Recency_Days","Damaged_Shipment_Pct"]],drop_first=True)
        yb=(df["Churned"]=="Yes").astype(int)
        dt=DecisionTreeClassifier(max_depth=4,random_state=42).fit(Xd,yb)
        imp=pd.Series(dt.feature_importances_,index=Xd.columns).sort_values(ascending=False).head(10)[::-1]
        fig=px.bar(x=imp.values,y=imp.index,orientation="h",color=imp.values,color_continuous_scale=["#cdeee8",TEAL,NAVY])
        fig.update_layout(height=430,title="Churn drivers (decision-tree importance)",coloraxis_showscale=False,yaxis_title=None)
        st.plotly_chart(fig,use_container_width=True)
    with st.expander("Decision-tree split rules explaining churn"):
        st.code(export_text(dt,feature_names=list(Xd.columns),max_depth=3))

    section("Anomaly, Exception &amp; Root Cause")
    a1,a2=st.columns([1,2])
    m=a1.selectbox("Metric (IQR scan)",["Monthly_Logistics_Spend_AED","Avg_Order_Value_AED","Complaints_Last_Quarter","Return_Rate_Pct"],key="dg_m")
    q1,q3=df[m].quantile([.25,.75]); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
    out=df[(df[m]<lo)|(df[m]>hi)]
    a2.markdown(f"<div class='hint'>{len(out)} outliers outside [{lo:,.0f}, {hi:,.0f}] on <b>{m}</b></div>",unsafe_allow_html=True)
    r1,r2=st.columns([2,3])
    with r1:
        box=px.box(df,y=m,points="outliers",color_discrete_sequence=[INDIGO]); box.update_layout(height=300)
        st.plotly_chart(box,use_container_width=True)
    with r2:
        rc=(df.groupby("Primary_Pain_Point").agg(Merchants=("Customer_ID","count"),
            Avg_Complaints=("Complaints_Last_Quarter","mean"),Churn_Rate=("Churned",lambda s:(s=="Yes").mean()*100))
            .round(2).sort_values("Churn_Rate",ascending=False))
        st.dataframe(style_grad(rc,["#fff7ec","#fdbb84","#e34a33"],axis=0,subset=["Churn_Rate"]),use_container_width=True)
    worst=rc.index[0]
    action(f"'{worst}' is the pain point most associated with churn — prioritise a service fix and proactive outreach for those accounts.")

def page_segmentation(df,feas):
    page_header("🧩","Customer Segmentation","What natural customer groups exist, and how should we treat each?")
    if feas["kmeans"][0]=="gate": gate(feas["kmeans"][1]); return
    section("K-Means — choosing K (Elbow · Silhouette · Davies-Bouldin)")
    s1,s2=st.columns([1,3])
    with s1:
        scaler=st.selectbox("Scaler",list(SCALERS.keys()),key="sg_sc")
        feats=st.multiselect("Features",CLUSTER_FEATURES,
            default=["Monthly_Order_Volume","Avg_Order_Value_AED","Number_of_SKUs","Monthly_Logistics_Spend_AED","Number_of_Sales_Channels"],key="sg_ft")
    if len(feats)<2: st.warning("Pick ≥2 features."); return
    ks,inertia,sil,dbi=kmeans_sweep(df,tuple(feats),scaler)
    bestk=ks[int(np.argmax(sil))]
    c1,c2,c3=st.columns(3)
    for c,(title,y,color,mark) in zip([c1,c2,c3],[("Elbow (inertia)",inertia,INDIGO,None),
            ("Silhouette (higher=better)",sil,TEAL,bestk),("Davies-Bouldin (lower=better)",dbi,AMBER,ks[int(np.argmin(dbi))])]):
        fig=go.Figure(go.Scatter(x=ks,y=y,mode="lines+markers",line=dict(color=color,width=3)))
        if mark: fig.add_vline(x=mark,line=dict(color=RED,dash="dash"))
        fig.update_layout(height=250,title=title,xaxis_title="k"); c.plotly_chart(fig,use_container_width=True)
    k=st.slider("Clusters (k)",2,8,int(bestk),key="sg_k")
    X=scale_X(df,feats,scaler); labels=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(X)
    pca=PCA(n_components=2,random_state=42).fit_transform(X)
    proj=pd.DataFrame({"PC1":pca[:,0],"PC2":pca[:,1],"Cluster":[f"C{l}" for l in labels]})
    prof=df.copy(); prof["Cluster"]=[f"C{l}" for l in labels]
    means=prof.groupby("Cluster")[feats].mean()
    # heuristic names — read from full frame so it never depends on the selected features
    gm=prof.groupby("Cluster").agg(spend=("Monthly_Logistics_Spend_AED","mean"),freq=("Order_Frequency_Per_Month","mean"))
    spend_rank=gm["spend"].rank(); names={}
    for cl in gm.index:
        sp=spend_rank[cl]
        if sp==spend_rank.max(): names[cl]="💎 Premium / Enterprise"
        elif sp==spend_rank.min(): names[cl]="🌱 Occasional / Small"
        elif gm.loc[cl,"freq"]>=gm["freq"].median(): names[cl]="🔁 Loyal / Frequent"
        else: names[cl]="💸 Price-sensitive"
    disp={cl:f"{names[cl]} · {cl}" for cl in names}  # unique labels (avoids duplicate-index issues)
    g1,g2=st.columns([3,2])
    with g1:
        proj["Segment"]=proj["Cluster"].map(disp)
        sc=px.scatter(proj,x="PC1",y="PC2",color="Segment",opacity=.7,color_discrete_sequence=SEQ)
        sc.update_layout(height=420,title="Clusters on 2 principal components"); st.plotly_chart(sc,use_container_width=True)
    with g2:
        norm=(means-means.min())/(means.max()-means.min()+1e-9)
        rad=go.Figure()
        for cl in norm.index:
            rad.add_trace(go.Scatterpolar(r=norm.loc[cl].values,theta=feats,fill="toself",name=disp[cl]))
        rad.update_layout(height=420,title="Cluster radar (normalised)",polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                          showlegend=True,legend=dict(font=dict(size=8)))
        st.plotly_chart(rad,use_container_width=True)
    prof2=means.copy(); prof2["Merchants"]=prof.groupby("Cluster").size(); prof2.index=[disp[i] for i in prof2.index]
    st.dataframe(style_grad(prof2,["#e8f6f1",TEAL,NAVY],axis=0),use_container_width=True)
    interp(f"Silhouette and Davies-Bouldin both favour ~{bestk} clusters — the segments are well separated on spend and volume, "
           "so segment-specific pricing and service tiers are statistically justified.")

    if feas["lca"][0]!="gate":
        section("Latent Class Analysis (service-usage indicators)","Model-based clustering on the 9 binary service flags; optimal classes by BIC.")
        tab,lk,model=lca_select(df)
        t1,t2=st.columns([2,3])
        with t1:
            bt=pd.DataFrame(tab,columns=["Classes","BIC","AIC"])
            fig=go.Figure(go.Scatter(x=bt["Classes"],y=bt["BIC"],mode="lines+markers",line=dict(color=INDIGO,width=3)))
            fig.add_vline(x=lk,line=dict(color=RED,dash="dash"),annotation_text=f"best={lk}")
            fig.update_layout(height=300,title="BIC by number of latent classes",xaxis_title="classes")
            st.plotly_chart(fig,use_container_width=True)
        with t2:
            P=pd.DataFrame(model["p"],columns=[c.replace("Uses_","").replace("_"," ") for c in FLAG_COLS],
                           index=[f"Class {i+1} ({p*100:.0f}%)" for i,p in enumerate(model["pi"])])
            hm=px.imshow(P,color_continuous_scale=["#fff","#a5d8ff",INDIGO],aspect="auto",
                         labels=dict(color="P(use)"),zmin=0,zmax=1)
            hm.update_layout(height=300,title="Service-usage probability per latent class"); st.plotly_chart(hm,use_container_width=True)
        from sklearn.metrics import adjusted_rand_score
        ari=adjusted_rand_score(labels,model["labels"])
        interp(f"LCA identifies <b>{lk}</b> service-adoption classes. Agreement with K-Means (Adjusted Rand = {ari:.2f}) is "
               f"{'high' if ari>0.5 else 'moderate' if ari>0.2 else 'low'} — LCA groups by <b>what services merchants buy</b> "
               "(better for bundling/cross-sell), while K-Means groups by <b>scale</b> (better for account tiering).")

def page_predictive(df,feas):
    page_header("🔮","Predictive Analytics","Which merchants will churn (or adopt premium), and what drives it?")
    if feas["classification"][0]=="gate": gate(feas["classification"][1]); return
    c1,c2,c3,c4,c5=st.columns(5)
    target=c1.selectbox("Target",["Churned","Premium_Tier_Adoption"],key="pr_t")
    autos=auto_scaler(df,NUMERIC_COLS)
    scaler=c2.selectbox("Scaler",list(SCALERS.keys()),index=list(SCALERS.keys()).index(autos),key="pr_sc")
    ts=c3.slider("Test size",0.15,0.4,0.25,0.05,key="pr_ts")
    tune=c4.toggle("Tune",value=False,key="pr_tune"); cv=c5.toggle("Cross-val",value=True,key="pr_cv")
    st.markdown(f"<span class='badge'>Auto-suggested scaler: {autos}</span>"
                f"<span class='badge {'best' if HAS_XGB else 'off'}'>XGBoost {'enabled' if HAS_XGB else 'not installed'}</span>"
                f"<span class='badge {'best' if HAS_SHAP else 'off'}'>SHAP {'enabled' if HAS_SHAP else 'not installed'}</span>",
                unsafe_allow_html=True)
    res,roc,cm,imp,best,prop,fp,mp=run_classification(df,target,scaler,tune,ts,cv)
    res_df=pd.DataFrame(res)
    st.markdown(f"<span class='badge best'>Best model: {best}</span><span class='badge'>ranked by ROC-AUC</span>",unsafe_allow_html=True)
    sty=style_grad(res_df,["#e8f5e9","#66bb6a","#1b5e20"],axis=0,subset=["Test Acc","F1","ROC-AUC"])
    sty=sty.apply(lambda r:["background-color:#e8fff4" if r["Model"]==best else "" for _ in r],axis=1)
    st.dataframe(sty,use_container_width=True)
    st.download_button("⬇️ Download model results (CSV)",res_df.to_csv(index=False).encode(),"model_results.csv","text/csv")
    g1,g2,g3=st.columns(3)
    with g1:
        fig=go.Figure()
        for i,(nm,(f_,t_)) in enumerate(roc.items()):
            au=res_df.loc[res_df.Model==nm,"ROC-AUC"].values[0]
            fig.add_trace(go.Scatter(x=f_,y=t_,mode="lines",name=f"{nm} ({au})",
                line=dict(width=3 if nm==best else 1.4,color=SEQ[i%len(SEQ)])))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",showlegend=False,line=dict(color="#94a3b8",dash="dash")))
        fig.update_layout(height=360,title="ROC curves",xaxis_title="FPR",yaxis_title="TPR",legend=dict(font=dict(size=8),y=.02,x=.35))
        st.plotly_chart(fig,use_container_width=True)
    with g2:
        cmf=px.imshow(cm,text_auto=True,color_continuous_scale="Teal",x=["No","Yes"],y=["No","Yes"],
            labels=dict(x="Predicted",y="Actual")); cmf.update_layout(height=360,title=f"Confusion — {best}",coloraxis_showscale=False)
        st.plotly_chart(cmf,use_container_width=True)
    with g3:
        cal=go.Figure(); cal.add_trace(go.Scatter(x=mp,y=fp,mode="lines+markers",line=dict(color=TEAL,width=3),name="Model"))
        cal.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="#94a3b8",dash="dash"),name="Perfect"))
        cal.update_layout(height=360,title="Calibration curve",xaxis_title="Mean predicted",yaxis_title="Observed")
        st.plotly_chart(cal,use_container_width=True)
    imp_df=pd.DataFrame(imp,columns=["Feature","Importance"]); imp_df["Feature"]=imp_df["Feature"].str.replace("num__","").str.replace("cat__","")
    fig=px.bar(imp_df.iloc[::-1],x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#cdeee8",TEAL,NAVY])
    fig.update_layout(height=380,title=f"Top drivers — {best}",coloraxis_showscale=False,yaxis_title=None)
    st.plotly_chart(fig,use_container_width=True)
    section("Propensity — who to act on now")
    pser=pd.Series(prop,index=df.index); tp=df.assign(Propensity=pser.round(3)).nlargest(15,"Propensity")
    st.dataframe(tp[["Customer_ID","Company_Name","Company_Size","Current_Provider","Satisfaction_Score","Propensity"]],use_container_width=True,height=300)
    action(f"<b>{best}</b> is the strongest model (AUC {res_df.iloc[0]['ROC-AUC']}). Feed its propensity scores to CRM and trigger "
           f"retention plays on the top-decile '{target}=Yes' accounts.")

def page_regression(df,feas):
    page_header("📈","Regression Analytics","Which factors drive Customer Lifetime Value, and by how much?")
    if feas["regression"][0]=="gate": gate(feas["regression"][1]); return
    alpha=st.slider("Regularisation strength (alpha)",0.01,10.0,1.0,0.01,key="rg_a")
    r=run_regression(df,"Customer_Lifetime_Value_AED",alpha)
    st.dataframe(style_grad(r["metrics"],["#e8f5e9","#66bb6a","#1b5e20"],axis=0,subset=["R2","Adj_R2"]),use_container_width=True)
    best=r["metrics"].sort_values("R2",ascending=False).iloc[0]["Model"]
    g1,g2=st.columns(2)
    with g1:
        yte=np.array(r["yte"]); pr=np.array(r["preds"][best])
        sc=px.scatter(x=yte,y=pr,opacity=.55,labels={"x":"Actual CLV","y":"Predicted CLV"},color_discrete_sequence=[TEAL])
        lim=max(yte.max(),pr.max()); sc.add_trace(go.Scatter(x=[0,lim],y=[0,lim],mode="lines",showlegend=False,line=dict(color="#94a3b8",dash="dash")))
        sc.update_layout(height=340,title=f"Predicted vs actual — {best}"); st.plotly_chart(sc,use_container_width=True)
    with g2:
        resid=yte-pr
        fig=px.scatter(x=pr,y=resid,opacity=.5,labels={"x":"Predicted","y":"Residual"},color_discrete_sequence=[INDIGO])
        fig.add_hline(y=0,line=dict(color="#94a3b8")); fig.update_layout(height=340,title="Residual analysis")
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        rc=pd.Series(r["ridge_coef"]).sort_values(); 
        fig=px.bar(x=rc.values,y=rc.index,orientation="h",color=rc.values,color_continuous_scale="RdBu",color_continuous_midpoint=0)
        fig.update_layout(height=380,title="Standardised coefficients (Ridge)",coloraxis_showscale=False,yaxis_title=None)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        alphas=np.array(r["alphas"]); path=np.array(r["coefs_path"])
        fig=go.Figure()
        for i,fn in enumerate(r["path_feats"]):
            fig.add_trace(go.Scatter(x=np.log10(alphas),y=path[i],mode="lines",name=fn.replace("_"," ")))
        fig.update_layout(height=380,title="Lasso regularisation paths",xaxis_title="log10(alpha)",yaxis_title="coef",legend=dict(font=dict(size=8)))
        st.plotly_chart(fig,use_container_width=True)
    top=pd.Series(r["ridge_coef"]).abs().idxmax()
    interp(f"<b>{best}</b> explains R²={r['metrics'].sort_values('R2',ascending=False).iloc[0]['R2']} of CLV variance. "
           f"The strongest standardised driver is <b>{top.replace('num__','').replace('cat__','')}</b>. "
           "Note MAPE is inflated by a few very-low-CLV accounts, so trust RMSE/MAE more here.")
    action("Grow CLV by moving mid-tier accounts up the strongest drivers (order volume & service depth) rather than chasing many small logos.")

def page_basket(df,feas):
    page_header("🛒","Service Basket Analysis","Which logistics services are bought together, and what should we bundle?")
    if feas["basket"][0]=="gate": gate(feas["basket"][1]); return
    st.markdown(f"<div class='callout cm'>ℹ️ {feas['basket'][1]}</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    ms=c1.slider("Min support",0.02,0.3,0.05,0.01,key="mb_s")
    mc=c2.slider("Min confidence",0.1,0.9,0.3,0.05,key="mb_c")
    ml=c3.slider("Min lift",1.0,3.0,1.0,0.1,key="mb_l")
    rules=assoc_rules(df,ms,mc); rules=rules[rules["Lift"]>=ml] if not rules.empty else rules
    if rules.empty: st.info("No rules at these thresholds — relax support/confidence."); return
    section("Rule Explorer")
    st.dataframe(style_grad(rules.head(25),["#e8f6f1",TEAL,NAVY],axis=0,subset=["Lift","Confidence"]),use_container_width=True,height=320)
    g1,g2=st.columns([3,2])
    with g1:
        top=rules.head(12); nodes=list(set(sum([r.split(" + ") for r in top["Antecedent"]],[]))|set(top["Consequent"]))
        ang={n:i/max(len(nodes),1)*2*np.pi for i,n in enumerate(nodes)}
        nx=[np.cos(a) for a in ang.values()]; ny=[np.sin(a) for a in ang.values()]
        fig=go.Figure()
        for _,r in top.iterrows():
            for a in r["Antecedent"].split(" + "):
                fig.add_trace(go.Scatter(x=[np.cos(ang[a]),np.cos(ang[r["Consequent"]])],
                    y=[np.sin(ang[a]),np.sin(ang[r["Consequent"]])],mode="lines",
                    line=dict(width=r["Lift"],color="rgba(99,102,241,.4)"),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=nx,y=ny,mode="markers+text",text=nodes,textposition="top center",
            marker=dict(size=16,color=TEAL),showlegend=False))
        fig.update_layout(height=420,title="Service affinity network (edge width = lift)",
            xaxis=dict(visible=False),yaxis=dict(visible=False)); st.plotly_chart(fig,use_container_width=True)
    with g2:
        M=df[FLAG_COLS].astype(int); co=M.T@M; lab=[c.replace("Uses_","").replace("_"," ") for c in FLAG_COLS]
        aff=pd.DataFrame(co.values,index=lab,columns=lab)
        hm=px.imshow(aff,color_continuous_scale=["#fff",TEAL,NAVY],aspect="auto"); hm.update_layout(height=420,title="Co-occurrence matrix")
        st.plotly_chart(hm,use_container_width=True)
    tr=rules.iloc[0]
    action(f"Bundle <b>{tr['Antecedent']}</b> with <b>{tr['Consequent']}</b> (lift {tr['Lift']}, confidence {tr['Confidence']}): "
           "merchants buying the first are far more likely to need the second — a natural cross-sell package.")

def page_forecast(df,feas):
    page_header("⏱️","Forecasting","What does the merchant-acquisition pipeline look like over the next months?")
    if feas["forecast"][0]=="gate": gate(feas["forecast"][1]); return
    f1,f2=st.columns([1,3])
    with f1:
        series=st.selectbox("Series",["New merchants / month","Onboarded spend / month"],key="fc_s")
        h=st.slider("Horizon (months)",3,12,6,key="fc_h")
        p=st.slider("p",0,3,1,key="fc_p"); d=st.slider("d",0,2,1,key="fc_d"); q=st.slider("q",0,3,1,key="fc_q")
    g=df.dropna(subset=["Signup_Month"]).groupby("Signup_Month")
    s=(g.size() if series.startswith("New") else g["Monthly_Logistics_Spend_AED"].sum()).asfreq("MS").fillna(0)
    with f2:
        try:
            if len(s)<10: st.info("Not enough monthly history for ARIMA."); return
            model,mean,ci=arima_forecast(s,(p,d,q),h)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines+markers",name="History",line=dict(color=NAVY)))
            fig.add_trace(go.Scatter(x=mean.index,y=mean.values,mode="lines+markers",name="Forecast",line=dict(color=TEAL,width=3)))
            fig.add_trace(go.Scatter(x=list(ci.index)+list(ci.index[::-1]),
                y=list(ci.iloc[:,1])+list(ci.iloc[:,0][::-1]),fill="toself",fillcolor="rgba(20,184,166,.15)",
                line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
            fig.update_layout(height=380,title=f"ARIMA({p},{d},{q}) — next {h} months"); st.plotly_chart(fig,use_container_width=True)
            # backtest
            if len(s)>16:
                tr=s.iloc[:-6]; te=s.iloc[-6:]
                _,fc6,_=arima_forecast(tr,(p,d,q),6)
                mae=mean_absolute_error(te.values,fc6.values[:len(te)]); rmse=mean_squared_error(te.values,fc6.values[:len(te)])**0.5
                c1,c2=st.columns(2); c1.metric("Backtest MAE (6-mo)",f"{mae:,.1f}"); c2.metric("Backtest RMSE",f"{rmse:,.1f}")
        except Exception as e:
            st.warning(f"ARIMA could not fit with these parameters — try different p,d,q. ({e})"); return
    trend="rising" if mean.values[-1]>=s.values[-6:].mean() else "cooling"
    interp(f"The acquisition pipeline is <b>{trend}</b>. Forecasts widen quickly (see CI) because only ~{len(s)} monthly points exist.")
    gate("SARIMA & Prophet are intentionally disabled: the series is short (~40 synthetic months) with no reliable seasonality, "
         "so seasonal models would over-fit. Re-enable once you have 2–3 years of real monthly data.")

def page_reco(df,feas):
    page_header("🎯","Recommendation Engine","What is the next-best service to offer each merchant?")
    if feas["reco"][0]=="gate": gate(feas["reco"][1]); return
    services,lab,M,R,item_sim,idx=recommender(df)
    c1,c2=st.columns([1,2])
    with c1:
        cid=st.selectbox("Merchant",df["Customer_ID"].tolist(),key="rc_c")
        method=st.radio("Method",["Matrix factorisation (NMF)","Item-item collaborative"],key="rc_m")
    i=idx[cid]; cur=[lab[s] for s,v in zip(services,M[i]) if v==1]
    if method.startswith("Matrix"):
        scores=[(lab[s],float(R[i,j])) for j,s in enumerate(services) if M[i,j]==0]
    else:
        scores=[]
        for j,s in enumerate(services):
            if M[i,j]==0:
                sim=sum(item_sim[j,k] for k in range(len(services)) if M[i,k]==1)
                scores.append((lab[s],float(sim)))
    scores=sorted(scores,key=lambda x:x[1],reverse=True)[:4]
    with c2:
        st.markdown(f"**Currently uses:** {', '.join(cur) if cur else 'no add-on services'}")
        if scores:
            rd=pd.DataFrame(scores,columns=["Recommended service","Affinity"])
            fig=px.bar(rd,x="Affinity",y="Recommended service",orientation="h",color="Affinity",color_continuous_scale=["#cdeee8",TEAL,NAVY])
            fig.update_layout(height=240,coloraxis_showscale=False,yaxis_title=None); st.plotly_chart(fig,use_container_width=True)
        else: st.info("This merchant already uses every service.")
    section("Service similarity")
    lab_l=[lab[s] for s in services]; sim=pd.DataFrame(item_sim,index=lab_l,columns=lab_l)
    hm=px.imshow(sim,color_continuous_scale=["#fff",TEAL,NAVY],aspect="auto"); hm.update_layout(height=380,title="Item–item similarity")
    st.plotly_chart(hm,use_container_width=True)
    st.caption("Cold-start note: brand-new merchants with no service history fall back to global popularity. "
               "Content-based filtering is limited (few item attributes); LightFM/implicit-ALS with side-features is the documented upgrade "
               "(kept off-Cloud because it needs a C build).")

def page_advisor(df,feas):
    page_header("🧠","AI Business Advisor","Given everything, what should management do next?")
    churn=(df["Churned"]=="Yes").mean()*100
    pg=df.groupby("Customer_ID")["Monthly_Logistics_Spend_AED"].sum().sort_values(ascending=False)
    top20=pg.head(int(len(pg)*0.2)).sum()/pg.sum()*100
    worst_prov=df.groupby("Current_Provider")["Satisfaction_Score"].mean().idxmin()
    worst_pain=df.groupby("Primary_Pain_Point").apply(lambda g:(g["Churned"]=="Yes").mean()).idxmax()
    hi_size=df.groupby("Company_Size").apply(lambda g:(g["Churned"]=="Yes").mean()*100).idxmax()
    rf=rfm(df); champs=(rf["RFM_Segment"]=="Champions").sum()
    atrisk=df[(df["Churned"]!="Yes")&(df["Satisfaction_Score"]<=df["Satisfaction_Score"].median())&(df["Recency_Days"]>60)]
    rules=assoc_rules(df,0.05,0.3); toprule=rules.iloc[0] if not rules.empty else None

    section("Executive Scorecard")
    kpi_cards([
        {"label":"Churn Rate","value":f"{churn:.1f}%","tone":"red","up":False},
        {"label":"Revenue in Top-20%","value":f"{top20:.0f}%","tone":"amber"},
        {"label":"Champion Accounts","value":f"{champs}","tone":"green"},
        {"label":"At-Risk Accounts","value":f"{len(atrisk):,}","tone":"violet","sub":"low sat + dormant"},
    ])
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### 🚀 Top opportunities")
        st.markdown(f"- **Cross-sell bundle:** {toprule['Antecedent']} → {toprule['Consequent']} (lift {toprule['Lift']})" if toprule is not None else "- Cross-sell across services")
        st.markdown(f"- **Upsell Champions** ({champs} accounts) to premium tier & annual contracts")
        st.markdown(f"- **Win share from {worst_prov}** — its merchants report the lowest satisfaction")
        st.markdown("#### ⚠️ Key risks")
        st.markdown(f"- **{hi_size}** segment has the highest churn rate")
        st.markdown(f"- **'{worst_pain}'** is the pain point most linked to churn")
        st.markdown(f"- Revenue concentration ({top20:.0f}% in top-20%) → dependency risk")
    with c2:
        st.markdown("#### 🎯 Priority actions (next 90 days)")
        st.markdown(f"1. Launch a save-play for the **{len(atrisk)} at-risk** accounts (proactive outreach + SLA credits)")
        st.markdown(f"2. Ship the **{toprule['Consequent'] if toprule is not None else 'top'} bundle** to relevant merchants")
        st.markdown(f"3. Fix the **'{worst_pain}'** journey; it drives the most churn")
        st.markdown("#### 🌍 Long-term")
        st.markdown("- Diversify revenue beyond the top-20% via mid-market activation")
        st.markdown("- Move merchants from pay-as-you-go to annual contracts to lift retention")
        st.markdown("- Deepen API/tech integration — it correlates with premium adoption")

    # downloadable executive report
    report=f"""# 3PL D2C — Executive Decision Report

## Business Health
- Merchants analysed: {len(df):,}
- Churn rate: {churn:.1f}%  |  Retention: {100-churn:.1f}%
- Avg satisfaction: {df['Satisfaction_Score'].mean():.1f}/10  |  Avg CLV: {aed(df['Customer_Lifetime_Value_AED'].mean())}
- Revenue concentration: top-20% of merchants = {top20:.0f}% of spend

## Segments
- Champion accounts (RFM): {champs}
- Highest-churn segment: {hi_size}
- At-risk accounts (low satisfaction + dormant): {len(atrisk)}

## Drivers & Opportunities
- Lowest-satisfaction provider (share-gain target): {worst_prov}
- Pain point most linked to churn: {worst_pain}
- Top service bundle: {toprule['Antecedent']+' -> '+toprule['Consequent'] if toprule is not None else 'n/a'} (lift {toprule['Lift'] if toprule is not None else 'n/a'})

## Priority Actions (90 days)
1. Save-play for {len(atrisk)} at-risk accounts.
2. Launch the top cross-sell bundle.
3. Remediate the '{worst_pain}' journey.

## Long-term
- Diversify revenue via mid-market activation.
- Shift merchants to annual contracts.
- Deepen API/tech integration.
"""
    d1,d2=st.columns(2)
    d1.download_button("⬇️ Download executive report (.md)",report.encode(),"executive_report.md","text/markdown")
    d2.download_button("⬇️ Download at-risk accounts (CSV)",
        atrisk[["Customer_ID","Company_Name","Company_Size","Current_Provider","Satisfaction_Score","Recency_Days"]].to_csv(index=False).encode(),
        "at_risk_accounts.csv","text/csv")

PAGES={
    "🏠 Executive Overview":page_overview,"👥 Customer Intelligence":page_customer,
    "🔬 Diagnostic Analytics":page_diagnostic,"🧩 Customer Segmentation":page_segmentation,
    "🔮 Predictive Analytics":page_predictive,"📈 Regression Analytics":page_regression,
    "🛒 Service Basket Analysis":page_basket,"⏱️ Forecasting":page_forecast,
    "🎯 Recommendation Engine":page_reco,"🧠 AI Business Advisor":page_advisor}

# =========================================================================================
# DATA LOADING + SIDEBAR + ROUTER
# =========================================================================================
@st.cache_data(show_spinner=True)
def load_raw(fb,name):
    if name.lower().endswith((".xlsx",".xls")):
        xls=pd.ExcelFile(io.BytesIO(fb)); sheet="Raw_Survey_Data" if "Raw_Survey_Data" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls,sheet_name=sheet)
    return pd.read_csv(io.BytesIO(fb))

@st.cache_data(show_spinner=True)
def get_clean(fb,name):
    raw=load_raw(fb,name); clean,rep=clean_data(raw); dq=data_quality(raw,clean); return raw,clean,rep,dq

with st.sidebar:
    st.markdown("### 📦 Decision Intelligence")
    dark=st.toggle("🌙 Dark mode",value=False)
apply_plotly_theme(dark); inject_css(dark)

with st.sidebar:
    st.caption("Upload survey data → auto-clean → analytics → strategy.")
    up=st.file_uploader("Dataset (CSV / XLSX)",type=["csv","xlsx","xls"])
    if up is not None: fb,fname=up.getvalue(),up.name
    else:
        with open("sample_data.csv","rb") as f: fb,fname=f.read(),"sample_data.csv"
        st.info("Using bundled sample (1,178 raw records).")

raw,df_clean,rep,dq=get_clean(fb,fname); st.session_state["_dq"]=dq
feas=feasibility(df_clean)

with st.sidebar:
    st.markdown("---"); st.markdown("#### 🧭 Navigate")
    choice=st.radio("Page",list(PAGES.keys()),label_visibility="collapsed")
    st.markdown("---"); st.markdown("#### 🔎 Filters")
    em=st.multiselect("Emirate",sorted(df_clean["Emirate"].unique()),default=sorted(df_clean["Emirate"].unique()))
    ind=st.multiselect("Industry",sorted(df_clean["Industry_Category"].unique()),default=sorted(df_clean["Industry_Category"].unique()))
    size=st.multiselect("Size",["Startup","SME","Mid-Market","Enterprise"],default=["Startup","SME","Mid-Market","Enterprise"])
    prov=st.multiselect("Provider",sorted(df_clean["Current_Provider"].unique()),default=sorted(df_clean["Current_Provider"].unique()))
    status=st.radio("Status",["All","Active only","Churned only"])

df=df_clean[df_clean["Emirate"].isin(em)&df_clean["Industry_Category"].isin(ind)
            &df_clean["Company_Size"].isin(size)&df_clean["Current_Provider"].isin(prov)].copy()
if status=="Active only": df=df[df["Churned"]!="Yes"]
elif status=="Churned only": df=df[df["Churned"]=="Yes"]

with st.sidebar:
    st.markdown("---")
    st.metric("Records in view",f"{len(df):,}",f"{len(df)-len(df_clean):+,}")
    st.download_button("⬇️ Cleaned dataset (CSV)",df_clean.to_csv(index=False).encode(),"cleaned_data.csv","text/csv")
    st.caption("Streamlit · scikit-learn · statsmodels · Plotly")

hero("Executive Decision Intelligence — 3PL D2C Logistics",
     "Upload → auto-clean → descriptive · diagnostic · predictive · prescriptive → strategy")

if df.empty:
    st.warning("No records match the current filters — widen your selection in the sidebar."); st.stop()

try:
    PAGES[choice](df,feas)
except Exception as e:
    st.error("⚠️ This page hit an issue and was skipped so the rest of the dashboard keeps working. "
             "Try adjusting filters or controls.")
    with st.expander("Technical detail (for debugging)"):
        st.exception(e)

