
import io
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="NIH Portfolio Explorer",
    page_icon="🧭",
    layout="wide",
)

APP_TITLE = "NIH Portfolio Explorer"
DEFAULT_DATA_PATH = Path("data/searchresult_export.xlsx")

TEXT_FIELDS = [
    "Project Title",
    "Public Health Relevance",
    "Project Terms",
    "Organization Name",
    "Organization City",
    "Organization State",
    "Organization Type",
    "Funding Mechanism",
    "Activity",
    "Administering IC",
]

THERA_PATTERNS = [
    ("ADC / antibody-drug conjugate", [r"\badc\b", r"antibody[- ]drug[- ]conjugate"]),
    ("CAR / CAR-T", [r"\bcar[- ]?t\b", r"chimeric antigen receptor"]),
    ("Gene therapy / editing", [r"\bgene therapy\b", r"\bcrispr\b", r"\bgene editing\b", r"\bviral vector\b"]),
    ("Cell therapy", [r"\bcell therapy\b", r"\bt[- ]cell\b", r"\bnk cell\b", r"\bcellular therapy\b"]),
    ("Antibody / biologic", [r"\bmonoclonal antibody\b", r"\bantibody\b", r"\bmab\b"]),
    ("Immunotherapy / vaccine", [r"\bimmunotherap", r"\bvaccine\b", r"\boncolytic\b"]),
    ("Small molecule", [r"\bsmall molecule\b", r"\binhibitor\b", r"\bagonist\b", r"\bcompound\b"]),
]

DIAG_PATTERNS = [
    ("Diagnostic / assay", [r"\bdiagnos", r"\bassay\b", r"\btest\b"]),
    ("Biomarker / companion diagnostic", [r"\bbiomarker\b", r"companion diagnostic"]),
    ("Imaging / screening", [r"\bimaging\b", r"\bscreen", r"\bscan\b"]),
]

DIGI_PATTERNS = [
    ("Digital health / software", [r"\bsoftware\b", r"\bdigital\b", r"\bapp\b", r"\bplatform\b", r"\binformatics\b"]),
    ("AI / machine learning", [r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bai\b", r"\bml\b", r"\balgorithm\b"]),
]

CANCER_PATTERNS = [
    ("Hematologic", [r"lymphoma", r"leukemia", r"myeloma", r"hematolog", r"lymphoid", r"b-cell", r"t-cell"]),
    ("Breast", [r"\bbreast\b"]),
    ("Lung", [r"\blung\b", r"nsclc", r"sclc"]),
    ("Prostate", [r"\bprostate\b"]),
    ("Colorectal", [r"colorectal", r"\bcolon\b", r"\brectal\b"]),
    ("Pancreatic", [r"pancreatic"]),
    ("Ovarian", [r"ovarian"]),
    ("Brain/CNS", [r"brain", r"glioblast", r"glioma", r"\bcns\b"]),
    ("Melanoma/Skin", [r"melanoma", r"skin cancer", r"\bskin\b"]),
    ("Head & Neck", [r"head and neck", r"head & neck", r"orophary", r"laryng"]),
    ("Bladder/Urologic", [r"bladder", r"\buro", r"kidney", r"renal"]),
    ("Liver", [r"\bliver\b", r"hepatocellular", r"\bhcc\b"]),
    ("Gynecologic", [r"cervical", r"uterine", r"endometrial", r"gynecolog"]),
    ("Sarcoma", [r"sarcoma"]),
    ("Pediatric", [r"pediatric", r"childhood"]),
]

STAGE_PATTERNS = [
    ("Clinical - Phase 3", [r"\bphase\s*iii\b", r"\bphase\s*3\b"]),
    ("Clinical - Phase 2", [r"\bphase\s*ii\b", r"\bphase\s*2\b"]),
    ("Clinical - Phase 1", [r"\bphase\s*i\b", r"\bphase\s*1\b", r"\bfirst[- ]in[- ]human\b", r"\bfih\b"]),
]

def normalize(x):
    return "" if pd.isna(x) else str(x)

def make_search_text(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in TEXT_FIELDS if c in df.columns]
    return df[cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

def count_family_hits(text: str, families):
    hits = []
    for label, pats in families:
        if any(re.search(p, text) for p in pats):
            hits.append(label)
    return hits

def portfolio_bucket(text: str) -> str:
    h_thera = count_family_hits(text, THERA_PATTERNS)
    h_diag = count_family_hits(text, DIAG_PATTERNS)
    h_digi = count_family_hits(text, DIGI_PATTERNS)

    scores = {
        "Therapeutics": len(h_thera),
        "Diagnostics": len(h_diag),
        "Digital Health": len(h_digi),
    }

    # Reduce false positives in the digital bucket when only generic platform words are present.
    if scores["Digital Health"] and not re.search(r"digital|software|artificial intelligence|machine learning|\bai\b|\bml\b|\bapp\b", text):
        scores["Digital Health"] = 0

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Other"

    return best

def modality_primary(text: str) -> str:
    ordered = [
        ("ADC / antibody-drug conjugate", [r"\badc\b", r"antibody[- ]drug[- ]conjugate"]),
        ("CAR / CAR-T", [r"\bcar[- ]?t\b", r"chimeric antigen receptor"]),
        ("Gene therapy / editing", [r"\bgene therapy\b", r"\bcrispr\b", r"\bgene editing\b", r"\bviral vector\b"]),
        ("Cell therapy", [r"\bcell therapy\b", r"\bt[- ]cell\b", r"\bnk cell\b", r"\bcellular therapy\b"]),
        ("Antibody / biologic", [r"\bmonoclonal antibody\b", r"\bantibody\b", r"\bmab\b"]),
        ("Immunotherapy / vaccine", [r"\bimmunotherap", r"\bvaccine\b", r"\boncolytic\b"]),
        ("Small molecule", [r"\bsmall molecule\b", r"\binhibitor\b", r"\bagonist\b", r"\bcompound\b"]),
        ("Diagnostics / assay", [r"\bdiagnos", r"\bassay\b", r"\bbiomarker\b", r"\bimaging\b", r"\bscreen"]),
        ("Digital health / software", [r"\bsoftware\b", r"\bdigital\b", r"\bapp\b", r"\binformatics\b", r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bai\b"]),
    ]
    for label, pats in ordered:
        if any(re.search(p, text) for p in pats):
            return label
    return "Other"

def modality_secondary(text: str) -> str:
    labels = []
    for label, pats in THERA_PATTERNS + DIAG_PATTERNS + DIGI_PATTERNS:
        if any(re.search(p, text) for p in pats):
            labels.append(label)
    return "; ".join(labels[:4])

def cancer_type(text: str) -> str:
    for label, pats in CANCER_PATTERNS:
        if any(re.search(p, text) for p in pats):
            return label
    if re.search(r"cancer|tumou?r|neoplasm", text):
        return "Pan-cancer / unspecified"
    return "Non-cancer / other"

def development_stage(text: str) -> str:
    for label, pats in STAGE_PATTERNS:
        if any(re.search(p, text) for p in pats):
            return label
    if re.search(r"\bclinical trial\b|\bpatient(s)?\b|\bhuman\b|\bsubject(s)?\b", text) and not re.search(r"\bpreclinical\b|\bmouse\b|\banimal\b|\bin vitro\b|\bxenograft\b", text):
        return "Clinical / late-stage"
    if re.search(r"\bpreclinical\b|\bmouse\b|\banimal\b|\bin vitro\b|\bxenograft\b|\bind[- ]?enabling\b", text):
        return "Preclinical / translational"
    if re.search(r"\bfeasibility\b|\bprototype\b|\bproof of concept\b|\bdevelopment\b|\boptimization\b", text):
        return "Discovery / development"
    return "Unknown"

def clinical_flag(stage: str) -> str:
    if stage.startswith("Clinical"):
        return "Clinical"
    if stage == "Unknown":
        return "Unknown"
    return "Non-clinical"

def institution_class(row) -> str:
    org_type = normalize(row.get("Organization Type", "")).lower()
    org_name = normalize(row.get("Organization Name", "")).lower()

    if any(k in org_name for k in ["university", "college", "school of medicine", "medical center", "hospital", "institute"]):
        return "Academic / university"
    if any(k in org_type for k in ["for-profit", "for profits", "company"]):
        return "Company"
    if any(k in org_type for k in ["non-profit", "non profits", "nonprofit"]):
        return "Nonprofit"
    if any(k in org_type for k in ["government", "federal"]):
        return "Government"
    return "Other"

def top_keyword_hits(text: str, max_hits: int = 6) -> str:
    terms = []
    families = THERA_PATTERNS + DIAG_PATTERNS + DIGI_PATTERNS + CANCER_PATTERNS
    for label, pats in families:
        if any(re.search(p, text) for p in pats):
            terms.append(label)
    return "; ".join(terms[:max_hits])

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.lower().endswith(("xls", "xlsx")) else pd.read_csv(uploaded_file)
    elif DEFAULT_DATA_PATH.exists():
        df = pd.read_excel(DEFAULT_DATA_PATH)
    else:
        raise FileNotFoundError("No data file found. Upload a .xlsx or .csv file.")

    df = df.copy()
    if "Total Cost" in df.columns:
        df["Total Cost"] = pd.to_numeric(df["Total Cost"], errors="coerce")
    if "Fiscal Year" in df.columns:
        df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce").astype("Int64")

    df["search_text"] = make_search_text(df)
    df["portfolio_bucket"] = df["search_text"].map(portfolio_bucket)
    df["modality_primary"] = df["search_text"].map(modality_primary)
    df["modality_secondary"] = df["search_text"].map(modality_secondary)
    df["cancer_type"] = df["search_text"].map(cancer_type)
    df["development_stage"] = df["search_text"].map(development_stage)
    df["clinical_flag"] = df["development_stage"].map(clinical_flag)
    df["institution_class"] = df.apply(institution_class, axis=1)
    df["keyword_hits"] = df["search_text"].map(top_keyword_hits)

    # useful aliases
    df["org_label"] = df.get("Organization Name", pd.Series([""] * len(df))).astype(str)
    df["state_label"] = df.get("Organization State", pd.Series([""] * len(df))).astype(str)
    df["title"] = df.get("Project Title", pd.Series([""] * len(df))).astype(str)
    df["abstract_like"] = df.get("Public Health Relevance", pd.Series([""] * len(df))).astype(str)
    return df

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()

    if filters["years"]:
        out = out[out["Fiscal Year"].isin(filters["years"])]
    for col, key in [
        ("portfolio_bucket", "buckets"),
        ("modality_primary", "modalities"),
        ("cancer_type", "cancers"),
        ("development_stage", "stages"),
        ("Organization State", "states"),
        ("institution_class", "institution_classes"),
        ("Organization Type", "org_types"),
        ("Funding Mechanism", "funding_mechanisms"),
        ("Activity", "activities"),
        ("Administering IC", "ics"),
    ]:
        vals = filters.get(key, [])
        if vals and col in out.columns:
            out = out[out[col].astype(str).isin(vals)]

    if filters["min_cost"] is not None and "Total Cost" in out.columns:
        out = out[out["Total Cost"].fillna(0) >= filters["min_cost"]]
    if filters["max_cost"] is not None and "Total Cost" in out.columns:
        out = out[out["Total Cost"].fillna(0) <= filters["max_cost"]]

    q = filters["query"].strip().lower()
    if q:
        # simple AND/OR-friendly keyword search over the combined text blob
        tokens = [t for t in re.split(r"\s+", q) if t]
        mask = pd.Series(True, index=out.index)
        for tok in tokens:
            if tok.startswith('"') and tok.endswith('"') and len(tok) > 2:
                phrase = tok.strip('"')
                mask &= out["search_text"].str.contains(re.escape(phrase), na=False)
            else:
                mask &= out["search_text"].str.contains(re.escape(tok), na=False)
        out = out[mask]

    return out

def add_similarity_scores(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        df = df.copy()
        df["relevance_score"] = 0.0
        return df

    corpus = df["search_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
    matrix = vectorizer.fit_transform(corpus + [query.lower()])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    out = df.copy()
    out["relevance_score"] = sims
    return out.sort_values(["relevance_score", "Total Cost"], ascending=[False, False])

def money(x):
    try:
        if pd.isna(x):
            return "—"
        return f"${x:,.0f}"
    except Exception:
        return "—"

def pct(part, whole):
    return 0 if whole == 0 else (part / whole) * 100

def chart_bar(df, x, y, title, color=None, top_n=15):
    tmp = df.groupby(x, dropna=False)[y].sum().sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(tmp, x=x, y=y, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def chart_donut(df, names, values, title):
    tmp = df.groupby(names, dropna=False)[values].sum().reset_index()
    fig = px.pie(tmp, names=names, values=values, hole=0.45, title=title)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def choropleth_states(df):
    tmp = df.groupby("Organization State", dropna=False)["Total Cost"].sum().reset_index()
    tmp = tmp[tmp["Organization State"].astype(str).str.len() == 2]
    fig = px.choropleth(
        tmp,
        locations="Organization State",
        locationmode="USA-states",
        color="Total Cost",
        scope="usa",
        color_continuous_scale="Blues",
        title="Funding by state",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def build_summary_cards(df: pd.DataFrame):
    total_projects = len(df)
    total_cost = df["Total Cost"].fillna(0).sum() if "Total Cost" in df.columns else 0
    clinical_share = pct((df["clinical_flag"] == "Clinical").sum(), total_projects)
    top_bucket = df["portfolio_bucket"].value_counts().idxmax() if total_projects else "—"
    top_state = df["Organization State"].value_counts().idxmax() if "Organization State" in df.columns and total_projects else "—"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Projects", f"{total_projects:,}")
    c2.metric("Total dollars", money(total_cost))
    c3.metric("Clinical share", f"{clinical_share:.1f}%")
    c4.metric("Top bucket", top_bucket)
    c5.metric("Top state", top_state)

def visible_columns():
    cols = [
        "Project Title",
        "Organization Name",
        "Organization State",
        "Organization Type",
        "Fiscal Year",
        "Total Cost",
        "portfolio_bucket",
        "modality_primary",
        "cancer_type",
        "development_stage",
        "clinical_flag",
        "institution_class",
        "Funding Mechanism",
        "Activity",
        "Administering IC",
        "keyword_hits",
        "Public Health Relevance",
    ]
    return [c for c in cols if c in st.session_state.get("df_cols", cols)]

st.title(APP_TITLE)
st.caption("Interactive portfolio analysis with keyword search, text-derived tags, and clinical-stage inference from title, project terms, and abstract-like text.")

uploaded = st.sidebar.file_uploader("Upload NIH export (.xlsx or .csv)", type=["xlsx", "xls", "csv"])
with st.sidebar.expander("About the derived tags", expanded=False):
    st.write("Portfolio bucket, modality, cancer type, and stage are heuristic text classifications derived from the project title, public health relevance, and project terms.")
    st.write("They are designed to help with portfolio exploration and can be tuned for your taxonomy.")

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.session_state["df_cols"] = list(df.columns)

years = sorted([int(x) for x in df["Fiscal Year"].dropna().unique().tolist()]) if "Fiscal Year" in df.columns else []
states = sorted(df["Organization State"].dropna().astype(str).unique().tolist()) if "Organization State" in df.columns else []
buckets = sorted(df["portfolio_bucket"].dropna().astype(str).unique().tolist())
modalities = sorted(df["modality_primary"].dropna().astype(str).unique().tolist())
cancers = sorted(df["cancer_type"].dropna().astype(str).unique().tolist())
stages = sorted(df["development_stage"].dropna().astype(str).unique().tolist())
inst_classes = sorted(df["institution_class"].dropna().astype(str).unique().tolist())
org_types = sorted(df["Organization Type"].dropna().astype(str).unique().tolist()) if "Organization Type" in df.columns else []
funding_mechanisms = sorted(df["Funding Mechanism"].dropna().astype(str).unique().tolist()) if "Funding Mechanism" in df.columns else []
activities = sorted(df["Activity"].dropna().astype(str).unique().tolist()) if "Activity" in df.columns else []
ics = sorted(df["Administering IC"].dropna().astype(str).unique().tolist()) if "Administering IC" in df.columns else []

st.sidebar.header("Filters")
query = st.sidebar.text_input("Keyword search", placeholder="e.g., CAR-T relapse phase 1 glioblastoma")
selected_years = st.sidebar.multiselect("Fiscal year", years, default=years)
selected_buckets = st.sidebar.multiselect("Portfolio bucket", buckets, default=buckets)
selected_modalities = st.sidebar.multiselect("Modality", modalities, default=modalities)
selected_cancers = st.sidebar.multiselect("Cancer type", cancers, default=cancers)
selected_stages = st.sidebar.multiselect("Development stage", stages, default=stages)
selected_states = st.sidebar.multiselect("State", states, default=states)
selected_inst_classes = st.sidebar.multiselect("Institution class", inst_classes, default=inst_classes)
selected_org_types = st.sidebar.multiselect("Organization type", org_types, default=org_types)
selected_funding_mechanisms = st.sidebar.multiselect("Funding mechanism", funding_mechanisms, default=funding_mechanisms)
selected_activities = st.sidebar.multiselect("Activity", activities, default=activities)
selected_ics = st.sidebar.multiselect("Administering IC", ics, default=ics)
if "Total Cost" in df.columns and df["Total Cost"].notna().any():
    min_cost = float(df["Total Cost"].fillna(0).min())
    max_cost = float(df["Total Cost"].fillna(0).max())
    cost_range = st.sidebar.slider("Total cost range", min_value=min_cost, max_value=max_cost, value=(min_cost, max_cost))
else:
    cost_range = (None, None)

filtered = apply_filters(
    df,
    {
        "query": query,
        "years": selected_years,
        "buckets": selected_buckets,
        "modalities": selected_modalities,
        "cancers": selected_cancers,
        "stages": selected_stages,
        "states": selected_states,
        "institution_classes": selected_inst_classes,
        "org_types": selected_org_types,
        "funding_mechanisms": selected_funding_mechanisms,
        "activities": selected_activities,
        "ics": selected_ics,
        "min_cost": cost_range[0],
        "max_cost": cost_range[1],
    },
)

if query.strip():
    filtered = add_similarity_scores(filtered, query)
else:
    filtered = filtered.copy()
    filtered["relevance_score"] = 0.0

build_summary_cards(filtered)

tab_overview, tab_search, tab_projects, tab_quality = st.tabs(["Overview", "Search", "Projects", "Quality checks"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        if "Total Cost" in filtered.columns:
            st.plotly_chart(chart_donut(filtered, "portfolio_bucket", "Total Cost", "Portfolio mix by category"), use_container_width=True)
        else:
            st.info("Total Cost column not available.")
    with right:
        if "Total Cost" in filtered.columns:
            st.plotly_chart(chart_bar(filtered, "modality_primary", "Total Cost", "Funding by modality"), use_container_width=True)
    left2, right2 = st.columns(2)
    with left2:
        if "Total Cost" in filtered.columns:
            st.plotly_chart(chart_bar(filtered, "cancer_type", "Total Cost", "Funding by cancer type"), use_container_width=True)
    with right2:
        if "Total Cost" in filtered.columns and "Fiscal Year" in filtered.columns:
            fy = filtered.groupby("Fiscal Year", dropna=False)["Total Cost"].sum().reset_index().sort_values("Fiscal Year")
            fig = px.line(fy, x="Fiscal Year", y="Total Cost", markers=True, title="Funding by fiscal year")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
    if "Organization State" in filtered.columns and "Total Cost" in filtered.columns:
        st.plotly_chart(choropleth_states(filtered), use_container_width=True)

with tab_search:
    st.subheader("Search and rank projects")
    st.write("Search uses the combined text from title, project terms, abstract-like text, organization, and funding fields.")
    top_n = st.slider("How many results to show", 5, 50, 15)
    ranked = filtered.sort_values(["relevance_score", "Total Cost"], ascending=[False, False]).head(top_n)
    display_cols = [c for c in [
        "Project Title", "Organization Name", "Organization State", "Fiscal Year", "Total Cost",
        "portfolio_bucket", "modality_primary", "cancer_type", "development_stage", "clinical_flag",
        "relevance_score", "keyword_hits"
    ] if c in ranked.columns]
    st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)
    if len(ranked):
        picked = st.selectbox("Inspect a project", ranked.index.tolist(), format_func=lambda i: ranked.loc[i, "Project Title"][:120])
        proj = ranked.loc[picked]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bucket", proj.get("portfolio_bucket", "—"))
        c2.metric("Modality", proj.get("modality_primary", "—"))
        c3.metric("Cancer type", proj.get("cancer_type", "—"))
        c4.metric("Stage", proj.get("development_stage", "—"))
        st.markdown("**Abstract-like text**")
        st.write(proj.get("Public Health Relevance", ""))
        st.markdown("**Project terms**")
        st.write(proj.get("Project Terms", ""))

with tab_projects:
    st.subheader("Project explorer")
    show_cols = st.multiselect(
        "Columns to display",
        [c for c in df.columns.tolist() if c not in {"search_text"}] + ["portfolio_bucket", "modality_primary", "cancer_type", "development_stage", "clinical_flag", "institution_class", "keyword_hits"],
        default=[c for c in [
            "Project Title", "Organization Name", "Organization State", "Fiscal Year", "Total Cost",
            "portfolio_bucket", "modality_primary", "cancer_type", "development_stage", "clinical_flag"
        ] if c in df.columns] + ["institution_class"],
    )
    show_cols = [c for c in show_cols if c in filtered.columns]
    explorer = filtered.copy()
    explorer = explorer.sort_values(["Total Cost", "relevance_score"], ascending=[False, False])
    st.dataframe(explorer[show_cols], use_container_width=True, hide_index=True)
    csv = explorer[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data", csv, "filtered_projects.csv", "text/csv")

with tab_quality:
    st.subheader("Data and classification checks")
    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.metric("Projects", f"{len(filtered):,}")
    qc2.metric("Unknown stage", f"{(filtered['development_stage'] == 'Unknown').sum():,}")
    qc3.metric("Non-cancer / other", f"{(filtered['cancer_type'] == 'Non-cancer / other').sum():,}")
    qc4.metric("Other bucket", f"{(filtered['portfolio_bucket'] == 'Other').sum():,}")
    st.write("Use the filters and keywords to zero in on the portfolio slices you care about. If you want, the taxonomy in the code can be tuned to match your internal review categories.")
    st.dataframe(
        filtered[[
            c for c in [
                "Project Title", "portfolio_bucket", "modality_primary", "cancer_type",
                "development_stage", "clinical_flag", "institution_class", "keyword_hits"
            ] if c in filtered.columns
        ]].head(30),
        use_container_width=True,
        hide_index=True,
    )
