"""
BioNOVA — Streamlit application entry point.

Run with:
    streamlit run app.py

Architecture: The UI contains no scientific algorithms.
All computation is delegated to retrieval, extraction, graph, hypothesis, utils.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import sys
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioNOVA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "BioNOVA — Biomedical Literature Mining and Hypothesis Discovery",
    },
)

# ── Logging ───────────────────────────────────────────────────────────────────
from styles import (  # noqa: E402
    get_dark_theme_css,
    get_empty_state_html,
    get_evidence_card_html,
    get_light_theme_css,
    get_page_header_html,
    get_polarity_badge_html,
    get_score_bar_html,
)
from utils import BioNOVAConfig, escape_html, setup_logging, truncate  # noqa: E402

_logger = setup_logging()


# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def _load_config() -> BioNOVAConfig:
    return BioNOVAConfig.load_env()


# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "page": "DISCOVER",
    "query": "",
    "articles": [],
    "entities": [],
    "relations": [],
    "bionova_graph": None,
    "gap_candidates": [],
    "hypotheses": [],
    "cutoff_year": None,
    "pipeline_ran": False,
    "pipeline_running": False,
    "analysis_metadata": {},
    "dark_mode": False,
    "search_use_year_filter": False,
    "search_min_year": 2000,
    "search_max_year": 2026,
    "search_max_results": 100,
    "use_demo": False,
    "setting_entrez_email": "",
    "setting_entrez_api_key": "",
    "setting_gemini_api_key": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_cfg_boot = _load_config()
if not st.session_state.get("setting_entrez_email"):
    st.session_state["setting_entrez_email"] = _cfg_boot.entrez_email
if not st.session_state.get("setting_entrez_api_key"):
    st.session_state["setting_entrez_api_key"] = _cfg_boot.entrez_api_key
if not st.session_state.get("setting_gemini_api_key"):
    st.session_state["setting_gemini_api_key"] = _cfg_boot.gemini_api_key

# ── CSS theme ─────────────────────────────────────────────────────────────────
_dark = st.session_state.get("dark_mode", False)
st.markdown(get_dark_theme_css() if _dark else get_light_theme_css(), unsafe_allow_html=True)

PAGES = ["DISCOVER", "EVIDENCE", "KNOWLEDGE GRAPH", "RESEARCH GAPS", "HYPOTHESES"]


def _get_runtime_config() -> BioNOVAConfig:
    cfg = _load_config()
    return dataclasses.replace(
        cfg,
        entrez_email=st.session_state.get("setting_entrez_email", cfg.entrez_email),
        entrez_api_key=st.session_state.get("setting_entrez_api_key", cfg.entrez_api_key),
        gemini_api_key=st.session_state.get("setting_gemini_api_key", cfg.gemini_api_key),
    )


def _format_years_covered(articles: List[Any]) -> str:
    years = sorted({a.year for a in articles if getattr(a, "year", None)})
    if not years:
        return "N/A"
    return f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])


def _corpus_info_text() -> Optional[str]:
    if not st.session_state.get("pipeline_ran"):
        return None
    meta = st.session_state.get("analysis_metadata", {})
    if not meta:
        return None
    parts = [f"{meta.get('n_articles', 0)} papers", f"{meta.get('n_relations', 0)} evidence claims"]
    years = meta.get("years_covered")
    if years:
        parts.append(str(years))
    return " · ".join(parts)


def _render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        get_page_header_html(title, subtitle, corpus_info=_corpus_info_text()),
        unsafe_allow_html=True,
    )


def _render_metric_card(label: str, value: Any, detail: str = "") -> None:
    detail_html = (
        f'<div style="margin-top:6px;font-size:12px;color:var(--text-secondary);">{escape_html(detail)}</div>'
        if detail
        else ""
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{escape_html(label)}</div>
            <div class="metric-card-value">{escape_html(str(value))}</div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_status_state() -> str:
    if st.session_state.get("pipeline_running"):
        return "running"
    if st.session_state.get("pipeline_ran"):
        return "complete"
    return "ready"


def _render_status_indicator() -> None:
    status = _get_status_state()
    label = {
        "ready": "● Pipeline ready",
        "running": "● Running",
        "complete": "● Complete",
    }[status]
    st.markdown(
        f'<div class="status-indicator {status}">{escape_html(label)}</div>',
        unsafe_allow_html=True,
    )


def _render_pipeline_progress(current_step: int, current_text: str = "") -> None:
    steps = [
        "Retrieve Literature",
        "Extract Entities",
        "Build Graph",
        "Find Gaps",
        "Ready for Review",
    ]
    cards: List[str] = []
    for index, label in enumerate(steps):
        cls = "complete" if index < current_step else "active" if index == current_step else "pending"
        detail = current_text if index == current_step and current_text else "Waiting"
        if cls == "complete":
            detail = "Complete"
        cards.append(
            f"""
            <div class="pipeline-step {cls}">
                <div class="pipeline-step-label">Step {index + 1}</div>
                <div class="pipeline-step-text">{escape_html(label)}</div>
                <div style="font-size:12px;color:#53657A;margin-top:4px;">{escape_html(detail)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="pipeline-progress">{"".join(cards)}</div>', unsafe_allow_html=True)


def _get_novelty_badge_html(status: str) -> str:
    label = str(status or "INSUFFICIENT_EVIDENCE").upper()
    config = {
        "UNEXPLORED": ("#2E8B57", "rgba(46,139,87,0.10)"),
        "INDIRECTLY_SUPPORTED": ("#1769AA", "rgba(23,105,170,0.10)"),
        "PARTIALLY_SUPPORTED": ("#D99000", "rgba(217,144,0,0.10)"),
        "KNOWN": ("#53657A", "rgba(83,101,122,0.10)"),
        "CONTRADICTED": ("#C73E3A", "rgba(199,62,58,0.10)"),
        "INSUFFICIENT_EVIDENCE": ("#6C63FF", "rgba(108,99,255,0.10)"),
    }
    color, bg = config.get(label, ("#53657A", "rgba(83,101,122,0.10)"))
    pretty = label.replace("_", " ")
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color};border-radius:999px;'
        f'padding:2px 10px;font-size:11px;font-weight:600;">{escape_html(pretty)}</span>'
    )


def _serialize_json(data: Any) -> str:
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=lambda value: dataclasses.asdict(value) if dataclasses.is_dataclass(value) else str(value),
    )


def _csv_from_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _article_lookup() -> Dict[str, Any]:
    return {str(article.pmid): article for article in st.session_state.get("articles", [])}


def _render_sidebar() -> str:
    cfg = _load_config()
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">BioNOVA</div>
                <div class="sidebar-brand-subtitle">Biomedical Novelty &amp; Opportunity Discovery</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_page = str(st.session_state.get("page", "DISCOVER")).upper()
        if current_page not in PAGES:
            current_page = "DISCOVER"
        selected_page = st.radio(
            "Navigation",
            PAGES,
            index=PAGES.index(current_page),
            label_visibility="collapsed",
        )
        st.session_state["page"] = selected_page

        _render_status_indicator()
        st.toggle("Dark mode", key="dark_mode")

        with st.expander("Settings", expanded=False):
            st.text_input("Entrez email", key="setting_entrez_email", placeholder=cfg.entrez_email or "name@institution.edu")
            st.text_input("Entrez API key", key="setting_entrez_api_key", type="password", placeholder="Optional")
            st.text_input("Gemini API key", key="setting_gemini_api_key", type="password", placeholder="Optional")

        with st.expander("About", expanded=False):
            st.markdown(
                "BioNOVA integrates literature retrieval, biomedical relation extraction, graph-based opportunity mapping, and hypothesis generation for scientific discovery workflows."
            )

        if st.session_state.get("pipeline_ran"):
            meta = st.session_state.get("analysis_metadata", {})
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="sidebar-corpus-summary">
                    <div style="font-weight:700;color:var(--text-primary);margin-bottom:6px;">Corpus summary</div>
                    <div><strong>Query:</strong> {escape_html(truncate(st.session_state.get('query', ''), 60))}</div>
                    <div><strong>Papers:</strong> {meta.get('n_articles', 0)}</div>
                    <div><strong>Entities:</strong> {meta.get('n_entities', 0)}</div>
                    <div><strong>Relations:</strong> {meta.get('n_relations', 0)}</div>
                    <div><strong>Years:</strong> {escape_html(str(meta.get('years_covered', 'N/A')))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return selected_page


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_discover_page() -> None:
    """Literature discovery and pipeline execution."""
    _render_page_header(
        "DISCOVER",
        "Explore biomedical literature, run the extraction pipeline, and establish the corpus for downstream evidence synthesis.",
    )

    cfg = _get_runtime_config()

    query = st.text_area(
        "Scientific research question",
        value=st.session_state.get("query", ""),
        placeholder="Explore the relationship between host-directed therapy, tuberculosis, and immune signaling...",
        height=140,
        key="discover_query_input",
    )

    with st.expander("Search Parameters", expanded=not st.session_state.get("pipeline_ran")):
        st.checkbox("Apply publication year filter", value=bool(st.session_state.get("search_use_year_filter", False)), key="search_use_year_filter")
        col1, col2, col3 = st.columns([1, 1, 1.2])
        with col1:
            st.number_input(
                "Min year",
                min_value=1900,
                max_value=2100,
                value=int(st.session_state.get("search_min_year", 2000)),
                step=1,
                key="search_min_year",
            )
        with col2:
            st.number_input(
                "Max year",
                min_value=1900,
                max_value=2100,
                value=int(st.session_state.get("search_max_year", 2026)),
                step=1,
                key="search_max_year",
            )
        with col3:
            st.slider(
                "Max results",
                min_value=10,
                max_value=300,
                value=int(st.session_state.get("search_max_results", 100)),
                step=10,
                key="search_max_results",
            )
        st.checkbox("Use demo data", value=bool(st.session_state.get("use_demo", False)), key="use_demo")

    run = st.button("Explore Literature", type="primary", use_container_width=True)

    _render_pipeline_progress(5 if st.session_state.get("pipeline_ran") else 0, "Ready")

    quick_cols = st.columns(4)
    quick_cards = [
        ("RESEARCH GAPS", "Research Gap Discovery", "Prioritize unexplored or weakly connected entity pairs for follow-up investigation."),
        ("EVIDENCE", "Temporal Trends", "Inspect study years, confidence levels, and evolving support across the literature."),
        ("KNOWLEDGE GRAPH", "Mechanistic Relationships", "Review graph structure, high-degree entities, and relation topology."),
        ("HYPOTHESES", "Contradictory Evidence", "Surface conflicting findings and route them into targeted hypotheses."),
    ]
    for column, (page_name, title, text) in zip(quick_cols, quick_cards):
        with column:
            st.markdown(
                f"""
                <div class="quick-card">
                    <div class="quick-card-kicker">Navigate to {escape_html(page_name)}</div>
                    <div class="quick-card-title">{escape_html(title)}</div>
                    <div class="quick-card-text">{escape_html(text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if run:
        q = query.strip()
        if st.session_state.get("search_use_year_filter") and st.session_state.get("search_min_year") and st.session_state.get("search_max_year"):
            if int(st.session_state["search_min_year"]) > int(st.session_state["search_max_year"]):
                st.warning("Min year cannot be greater than max year.")
                return
        if not q and not st.session_state.get("use_demo"):
            st.warning("Please enter a research question or enable demo data.")
            return
        _run_pipeline(
            query=q,
            max_results=int(st.session_state.get("search_max_results", 100)),
            cutoff_year=(
                int(st.session_state.get("search_max_year"))
                if st.session_state.get("search_use_year_filter") and st.session_state.get("search_max_year")
                else None
            ),
            use_demo=bool(st.session_state.get("use_demo")),
            cfg=cfg,
        )

    if st.session_state.get("pipeline_ran") and not st.session_state.get("pipeline_running"):
        meta = st.session_state.get("analysis_metadata", {})
        metrics = st.columns(5)
        metric_values = [
            ("Papers", meta.get("n_articles", 0), "Retrieved corpus"),
            ("Entities", meta.get("n_entities", 0), "Extracted mentions"),
            ("Relations", meta.get("n_relations", 0), "Structured claims"),
            ("Evidence Claims", meta.get("n_evidence_claims", 0), "Sentence-level evidence"),
            ("Years Covered", meta.get("years_covered", "N/A"), "Publication span"),
        ]
        for col, (label, value, detail) in zip(metrics, metric_values):
            with col:
                _render_metric_card(label, value, detail)

        st.markdown(
            f"""
            <div class="recent-analysis-card">
                <div class="metric-card-label">Recent analysis</div>
                <div style="font-size:16px;font-weight:600;color:var(--text-primary);margin-top:6px;">{escape_html(truncate(st.session_state.get('query', ''), 140))}</div>
                <div style="font-size:13px;color:var(--text-secondary);margin-top:8px;">
                    {meta.get('n_nodes', 0)} graph nodes · {meta.get('n_gaps', 0)} gap candidates · {meta.get('n_hypotheses', len(st.session_state.get('hypotheses', [])))} hypotheses
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        articles = st.session_state.get("articles", [])
        if articles:
            st.markdown("### Corpus preview")
            for article in articles[:5]:
                title = article.title or f"PMID {article.pmid}"
                with st.expander(f"[{article.pmid}] {truncate(title, 120)}", expanded=False):
                    st.markdown(f"**Year:** {article.year or 'N/A'}  ")
                    st.markdown(f"**Journal:** {escape_html(article.journal or 'N/A')}  ", unsafe_allow_html=False)
                    if article.doi:
                        st.markdown(f"**DOI:** [{article.doi}](https://doi.org/{article.doi})")
                    st.markdown(escape_html(article.abstract or "No abstract available."))


def _run_pipeline(
    query: str,
    max_results: int,
    cutoff_year: Optional[int],
    use_demo: bool,
    cfg: BioNOVAConfig,
) -> None:
    """Execute the BioNOVA pipeline with live progress display."""
    from retrieval import create_sample_articles, retrieve_and_clean
    from extraction import ExtractionPipeline
    from graph import build_graph

    st.session_state["pipeline_running"] = True
    st.session_state["query"] = query
    st.session_state["cutoff_year"] = cutoff_year

    progress_bar = st.progress(0.0, text="Initialising…")
    status = st.empty()
    pipeline_placeholder = st.empty()

    def _set_step(step_index: int, message: str, progress: float) -> None:
        pipeline_placeholder.markdown("", unsafe_allow_html=True)
        with pipeline_placeholder.container():
            _render_pipeline_progress(step_index, message)
        status.info(message)
        progress_bar.progress(progress, text=message)

    try:
        _set_step(0, "Retrieving literature from PubMed…", 0.05)

        if use_demo or not query:
            articles = create_sample_articles()
            _logger.info("Using demo articles: %d", len(articles))
        else:
            try:
                articles = retrieve_and_clean(
                    query=query,
                    config=cfg,
                    max_results=max_results,
                    progress_callback=lambda fetched, total: progress_bar.progress(
                        0.05 + 0.25 * (fetched / max(total, 1)),
                        text=f"Retrieving literature from PubMed… {fetched}/{total}",
                    ),
                )
            except Exception as exc:
                _logger.warning("PubMed retrieval failed, using demo: %s", exc)
                st.warning(f"PubMed retrieval failed ({exc}). Using demo data.")
                articles = create_sample_articles()

        if not articles:
            st.error("No articles retrieved. Check the query or enable demo data.")
            return

        min_year = st.session_state.get("search_min_year")
        if st.session_state.get("search_use_year_filter") and min_year:
            articles = [a for a in articles if a.year is None or a.year >= int(min_year)]
        if cutoff_year:
            articles = [a for a in articles if a.year is None or a.year <= cutoff_year]
            _logger.info("After year filters: %d articles", len(articles))

        _set_step(1, "Extracting biomedical entities and relationships…", 0.35)
        extractor = ExtractionPipeline(use_scispacy=False)
        entities, relations = extractor.process_articles(articles)

        _set_step(2, "Building knowledge graph…", 0.65)
        bg = build_graph(relations, cutoff_year=cutoff_year)
        stats = bg.get_statistics()

        _set_step(3, "Discovering research gaps…", 0.82)
        gap_candidates = bg.find_gap_candidates(cutoff_year=cutoff_year, top_k=50)

        years_covered = _format_years_covered(articles)
        st.session_state["articles"] = articles
        st.session_state["entities"] = entities
        st.session_state["relations"] = relations
        st.session_state["bionova_graph"] = bg
        st.session_state["gap_candidates"] = gap_candidates
        st.session_state["hypotheses"] = []
        st.session_state["analysis_metadata"] = {
            "query": query,
            "n_articles": len(articles),
            "n_entities": len(entities),
            "n_relations": len(relations),
            "n_evidence_claims": sum(1 for relation in relations if getattr(relation, "sentence", "")),
            "n_nodes": stats.get("n_nodes", 0),
            "n_edges": stats.get("n_edges", 0),
            "n_biological_edges": stats.get("n_biological_edges", 0),
            "n_gaps": len(gap_candidates),
            "n_hypotheses": 0,
            "cutoff_year": cutoff_year,
            "years_covered": years_covered,
        }
        st.session_state["pipeline_ran"] = True

        _set_step(5, "Pipeline complete.", 1.0)
        status.success(
            f"Pipeline complete — {len(articles)} papers, {len(relations)} relations, and {len(gap_candidates)} candidate gaps ready for review."
        )

    except Exception as exc:
        _logger.exception("Pipeline failed: %s", exc)
        st.error(f"Pipeline error: {exc}")
    finally:
        st.session_state["pipeline_running"] = False


def _render_evidence_page() -> None:
    """Display extracted evidence relations."""
    _render_page_header(
        "EVIDENCE",
        "Review extracted subject–predicate–object claims, inspect supporting sentences, and filter the evidence landscape before graph or hypothesis analysis.",
    )

    relations = st.session_state.get("relations", [])
    if not relations:
        st.markdown(
            get_empty_state_html(
                "🧾",
                "No evidence available yet",
                "Run DISCOVER to retrieve literature and extract relation evidence for review.",
                "Start from DISCOVER to populate this evidence workspace.",
            ),
            unsafe_allow_html=True,
        )
        return

    years = [int(r.publication_year) for r in relations if getattr(r, "publication_year", None)]
    year_min = min(years) if years else 1990
    year_max = max(years) if years else 2026
    predicate_options = sorted({str(r.predicate) for r in relations if getattr(r, "predicate", None)})
    polarity_options = sorted({str(r.polarity or "uncertain").lower() for r in relations})
    certainty_options = sorted({str(r.certainty or "uncertain").lower() for r in relations})

    filter_col, results_col = st.columns([1, 2.2], gap="large")
    with filter_col:
        st.markdown("### Filters")
        entity_filter = st.text_input("Entity contains", placeholder="e.g. TNF, tuberculosis")
        selected_predicates = st.multiselect("Relation type", predicate_options)
        selected_polarities = st.multiselect("Polarity", polarity_options, default=polarity_options)
        selected_certainties = st.multiselect("Certainty", certainty_options, default=certainty_options)
        selected_years = st.slider("Publication year", min_value=year_min, max_value=year_max, value=(year_min, year_max))
        min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    entity_filter_lc = entity_filter.strip().lower()
    filtered = []
    for relation in relations:
        haystack = " ".join([str(relation.subject), str(relation.object), str(relation.predicate)]).lower()
        relation_year = int(relation.publication_year) if relation.publication_year else None
        if entity_filter_lc and entity_filter_lc not in haystack:
            continue
        if selected_predicates and relation.predicate not in selected_predicates:
            continue
        if selected_polarities and str(relation.polarity or "uncertain").lower() not in selected_polarities:
            continue
        if selected_certainties and str(relation.certainty or "uncertain").lower() not in selected_certainties:
            continue
        if relation_year and not (selected_years[0] <= relation_year <= selected_years[1]):
            continue
        if float(relation.confidence or 0.0) < min_confidence:
            continue
        filtered.append(relation)
    filtered.sort(key=lambda item: float(getattr(item, "confidence", 0.0) or 0.0), reverse=True)

    with results_col:
        st.markdown(f"### Filtered evidence ({len(filtered)})")
        if not filtered:
            st.markdown(
                get_empty_state_html(
                    "🔎",
                    "No relations matched the current filters",
                    "Adjust entity, polarity, certainty, or confidence filters to broaden the evidence view.",
                ),
                unsafe_allow_html=True,
            )
            return

        article_by_pmid = _article_lookup()
        for idx, relation in enumerate(filtered):
            st.markdown(
                get_evidence_card_html(
                    relation.subject,
                    relation.predicate,
                    relation.object,
                    relation.sentence,
                    relation.pmid,
                    relation.publication_year,
                    relation.confidence,
                    relation.polarity,
                ),
                unsafe_allow_html=True,
            )
            with st.expander(f"Evidence details · PMID {relation.pmid or 'N/A'}", expanded=False):
                article = article_by_pmid.get(str(relation.pmid))
                left, right = st.columns([2, 1])
                with left:
                    st.markdown(f"**Sentence**  \n{escape_html(relation.sentence or 'No sentence available.')}")
                    if article:
                        st.markdown(f"**Paper**  \n{escape_html(article.title or 'Untitled record')}")
                        st.markdown(f"**Journal**  \n{escape_html(article.journal or 'N/A')}")
                with right:
                    st.markdown(f"**Subject type**  \n{escape_html(relation.subject_type or 'N/A')}")
                    st.markdown(f"**Object type**  \n{escape_html(relation.object_type or 'N/A')}")
                    st.markdown(f"**Certainty**  \n{escape_html(relation.certainty or 'N/A')}")
                    st.markdown(f"**Confidence**  \n{float(relation.confidence or 0.0):.2f}")
                    st.markdown(f"**Polarity**  \n{escape_html(str(relation.polarity or 'uncertain').upper())}")
            if idx < len(filtered) - 1:
                st.divider()


def _render_graph_page() -> None:
    """Display knowledge graph statistics and exports."""
    _render_page_header(
        "KNOWLEDGE GRAPH",
        "Examine graph structure, prioritize dominant entities, and export the network for downstream analysis in graph-native tools.",
    )

    bg = st.session_state.get("bionova_graph")
    if bg is None:
        st.markdown(
            get_empty_state_html(
                "🕸️",
                "Knowledge graph not yet built",
                "Run DISCOVER to construct the biomedical graph from extracted relations.",
            ),
            unsafe_allow_html=True,
        )
        return

    import pandas as pd
    import networkx as nx
    from networkx.readwrite import json_graph

    stats = bg.get_statistics()
    combined = nx.compose(bg.graph, bg.semantic_graph)
    export_graph = nx.DiGraph()
    for node, data in combined.nodes(data=True):
        export_graph.add_node(
            node,
            label=str(data.get("label", node)),
            entity_type=str(data.get("entity_type", "OTHER")),
        )
    for subject, obj, data in combined.edges(data=True):
        export_graph.add_edge(
            subject,
            obj,
            predicate=str(data.get("predicate", "")),
            polarity=str(data.get("polarity", "positive")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            weight=float(data.get("weight", 0.0) or 0.0),
            pmids=";".join(str(pmid) for pmid in data.get("pmids", [])),
            years=";".join(str(year) for year in data.get("years", [])),
        )

    metric_cols = st.columns(5)
    metric_data = [
        ("Nodes", stats.get("n_nodes", 0), "Entity concepts"),
        ("Biological Edges", stats.get("n_biological_edges", 0), "Mechanistic relations"),
        ("Semantic Edges", stats.get("n_semantic_edges", 0), "Contextual relations"),
        ("Avg Degree", f"{stats.get('avg_degree', 0.0):.2f}", "Connectivity"),
        ("Components", stats.get("connected_components", 0), "Undirected graph"),
    ]
    for col, (label, value, detail) in zip(metric_cols, metric_data):
        with col:
            _render_metric_card(label, value, detail)

    export_cols = st.columns([1, 1, 3])
    graphml_data = "\n".join(nx.generate_graphml(export_graph))
    graph_json = _serialize_json(json_graph.node_link_data(export_graph))
    with export_cols[0]:
        st.download_button("Export GraphML", data=graphml_data, file_name="bionova_graph.graphml", mime="application/graphml+xml")
    with export_cols[1]:
        st.download_button("Export JSON", data=graph_json, file_name="bionova_graph.json", mime="application/json")
    with export_cols[2]:
        st.info("Full interactive graph visualization requires graph rendering library support. Use the exports for Cytoscape, Gephi, or downstream network tooling.")

    st.markdown("### Top entities by degree")
    top_nodes = stats.get("top_nodes_by_degree", [])
    if top_nodes:
        rows = []
        for node_key, degree in top_nodes:
            node_data = combined.nodes.get(node_key, {})
            rows.append(
                {
                    "Entity": node_data.get("label") or node_key,
                    "Type": node_data.get("entity_type") or "OTHER",
                    "Degree": degree,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Edge list")
    relations = st.session_state.get("relations", [])
    edge_rows = [
        {
            "Subject": relation.subject,
            "Predicate": relation.predicate,
            "Object": relation.object,
            "Polarity": relation.polarity,
            "Certainty": relation.certainty,
            "Confidence": f"{float(relation.confidence or 0.0):.2f}",
            "PMID": relation.pmid,
            "Year": relation.publication_year or "",
        }
        for relation in relations[:250]
    ]
    if edge_rows:
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown(get_empty_state_html("🧬", "No edges available", "No relations were extracted for graph construction."), unsafe_allow_html=True)


def _render_gap_card(candidate: Any, index: int) -> None:
    novelty_label = candidate.novelty_status.name if hasattr(candidate.novelty_status, "name") else str(candidate.novelty_status)
    neighbors = candidate.common_neighbors[:5]
    neighbors_text = ", ".join(neighbors) if neighbors else "No shared intermediates recorded"
    st.markdown(
        f"""
        <div class="gap-card">
            <div class="gap-card-header">
                <div>
                    <div class="metric-card-label">Gap {index + 1}</div>
                    <div class="gap-card-title">{escape_html(candidate.node_a)} → ? → {escape_html(candidate.node_b)}</div>
                </div>
                <div>{_get_novelty_badge_html(novelty_label)}</div>
            </div>
            <div class="gap-card-text">Indirect evidence via {escape_html(neighbors_text)}.</div>
            <div class="badge-row">
                <span class="soft-badge">Method: {escape_html(candidate.method)}</span>
                <span class="soft-badge">Entity types: {escape_html(candidate.node_a_type)} / {escape_html(candidate.node_b_type)}</span>
            </div>
            {get_score_bar_html('Gap score', float(candidate.score or 0.0), show_level=True)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_research_gaps_page() -> None:
    """Display gap candidates and novelty verification."""
    _render_page_header(
        "RESEARCH GAPS",
        "Prioritize candidate connections that are absent, weakly supported, or contradictory in the current corpus to guide opportunity discovery.",
    )

    gap_candidates = st.session_state.get("gap_candidates", [])
    if not gap_candidates:
        st.markdown(
            get_empty_state_html(
                "🧭",
                "No research gaps detected yet",
                "Run DISCOVER to build the graph and identify candidate gaps for novelty assessment.",
            ),
            unsafe_allow_html=True,
        )
        return

    from collections import Counter

    novelty_counts = Counter(
        c.novelty_status.name if hasattr(c.novelty_status, "name") else str(c.novelty_status)
        for c in gap_candidates
    )
    total_gaps = len(gap_candidates)
    high_confidence = len([c for c in gap_candidates if float(c.score or 0.0) >= 0.7])
    unexplored_count = novelty_counts.get("UNEXPLORED", 0)
    indirect_count = novelty_counts.get("INDIRECTLY_SUPPORTED", 0)

    top_metrics = st.columns(4)
    summary_values = [
        ("Total gaps", total_gaps, "Candidate pairs"),
        ("High-confidence gaps", high_confidence, "Score ≥ 0.70"),
        ("Unexplored", unexplored_count, "No direct support"),
        ("Indirectly supported", indirect_count, "Multi-hop support"),
    ]
    for col, (label, value, detail) in zip(top_metrics, summary_values):
        with col:
            _render_metric_card(label, value, detail)

    breakdown_html = " ".join(
        f'<span class="soft-badge">{escape_html(status.replace("_", " "))}: {count}</span>'
        for status, count in sorted(novelty_counts.items())
    )
    st.markdown(f"<div class='badge-row' style='margin-bottom:16px;'>{breakdown_html}</div>", unsafe_allow_html=True)

    novelty_options = sorted(novelty_counts.keys())
    entity_types = sorted({c.node_a_type for c in gap_candidates} | {c.node_b_type for c in gap_candidates})
    controls = st.columns([1.3, 1, 1])
    with controls[0]:
        selected_novelty = st.multiselect("Novelty status", novelty_options, default=novelty_options)
    with controls[1]:
        min_score = st.slider("Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with controls[2]:
        selected_types = st.multiselect("Entity type filter", entity_types)

    sort_by = st.selectbox("Sort gaps by", ["Score", "Novelty", "Alphabetical"])

    filtered = []
    for candidate in gap_candidates:
        novelty_label = candidate.novelty_status.name if hasattr(candidate.novelty_status, "name") else str(candidate.novelty_status)
        if selected_novelty and novelty_label not in selected_novelty:
            continue
        if float(candidate.score or 0.0) < min_score:
            continue
        if selected_types and candidate.node_a_type not in selected_types and candidate.node_b_type not in selected_types:
            continue
        filtered.append(candidate)

    novelty_rank = {
        "UNEXPLORED": 0,
        "INDIRECTLY_SUPPORTED": 1,
        "PARTIALLY_SUPPORTED": 2,
        "INSUFFICIENT_EVIDENCE": 3,
        "KNOWN": 4,
        "CONTRADICTED": 5,
    }
    if sort_by == "Score":
        filtered.sort(key=lambda item: float(item.score or 0.0), reverse=True)
    elif sort_by == "Novelty":
        filtered.sort(
            key=lambda item: (
                novelty_rank.get(item.novelty_status.name if hasattr(item.novelty_status, "name") else str(item.novelty_status), 99),
                -float(item.score or 0.0),
            )
        )
    else:
        filtered.sort(key=lambda item: (item.node_a.lower(), item.node_b.lower()))

    export_rows = [
        {
            "entity_a": c.node_a,
            "entity_b": c.node_b,
            "entity_a_type": c.node_a_type,
            "entity_b_type": c.node_b_type,
            "score": c.score,
            "novelty_status": c.novelty_status.name if hasattr(c.novelty_status, "name") else str(c.novelty_status),
            "method": c.method,
            "common_neighbors": "; ".join(c.common_neighbors),
            "max_year": c.max_year,
        }
        for c in filtered
    ]
    st.download_button(
        "Export gaps as CSV",
        data=_csv_from_rows(export_rows) if export_rows else "",
        file_name="bionova_research_gaps.csv",
        mime="text/csv",
        disabled=not bool(export_rows),
    )

    if not filtered:
        st.markdown(
            get_empty_state_html(
                "📉",
                "No gaps matched the current filters",
                "Broaden novelty, score, or entity type filters to recover candidates.",
            ),
            unsafe_allow_html=True,
        )
        return

    for index, candidate in enumerate(filtered):
        _render_gap_card(candidate, index)
        action_cols = st.columns([1.2, 4])
        with action_cols[0]:
            if st.button("Generate Hypothesis", key=f"gap_hypothesis_{index}"):
                st.session_state["page"] = "HYPOTHESES"
                st.rerun()
        with action_cols[1]:
            pmid_count = len(getattr(candidate, "evidence_pmids", []) or [])
            st.caption(f"Supporting neighbor evidence PMIDs: {pmid_count}")


def _generate_hypotheses(gap_candidates, relations, query, cfg, top_k, use_llm):
    """Run hypothesis generation pipeline."""
    from hypothesis import HypothesisPipeline

    progress = st.progress(0.0, text="Generating hypotheses…")
    try:
        pipeline = HypothesisPipeline(
            config_or_api_key=cfg.gemini_api_key if use_llm else None,
            use_llm=use_llm and bool(cfg.gemini_api_key),
        )
        hypotheses = pipeline.generate_hypotheses(
            candidates=gap_candidates,
            relations=relations,
            query=query,
            top_k=top_k,
        )
        progress.progress(1.0, text="Hypothesis generation complete.")
        st.session_state["hypotheses"] = hypotheses
        st.session_state["analysis_metadata"]["n_hypotheses"] = len(hypotheses)
        st.success(f"Generated {len(hypotheses)} hypotheses.")
        st.rerun()
    except Exception as exc:
        progress.empty()
        _logger.exception("Hypothesis generation failed: %s", exc)
        st.error(f"Hypothesis generation failed: {exc}")


def _render_hypothesis_card(hyp, index: int):
    """Render a single hypothesis card with full evidence display."""
    st.markdown(
        f"""
        <div class="hypothesis-card">
            <div class="hypothesis-card-header">
                <div>
                    <div class="metric-card-label">Hypothesis {index + 1}</div>
                    <div class="hypothesis-card-title">{escape_html(hyp.hypothesis)}</div>
                </div>
                <div>{_get_novelty_badge_html(hyp.novelty_status)}</div>
            </div>
            <div class="hypothesis-card-text">{escape_html(hyp.mechanistic_rationale or 'Mechanistic rationale not available.')}</div>
            <div style="margin-top:10px;">
                {get_score_bar_html('Final score', float(hyp.final_score or 0.0))}
                {get_score_bar_html('Evidence', float(hyp.evidence_score or 0.0))}
                {get_score_bar_html('Novelty', float(hyp.novelty_score or 0.0))}
                {get_score_bar_html('Plausibility', float(hyp.plausibility_score or 0.0))}
                {get_score_bar_html('Feasibility', float(hyp.feasibility_score or 0.0))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_main, tab_evidence, tab_method = st.tabs(["Hypothesis Details", "Evidence", "Methods"])

    with tab_main:
        st.markdown(f"**Entities**  \n{escape_html(', '.join(hyp.entities))}")
        st.markdown(f"**Testable prediction**  \n{escape_html(hyp.testable_prediction or 'N/A')}")
        st.markdown(f"**Suggested experiment**  \n{escape_html(hyp.suggested_experiment or 'N/A')}")
        st.markdown(f"**Falsifiability**  \n{escape_html(hyp.falsifiability or 'N/A')}")
        st.markdown(f"**Alternative explanations**  \n{escape_html(hyp.alternative_explanations or 'N/A')}")
        st.markdown(f"**Devil's advocate**  \n{escape_html(hyp.devils_advocate or 'N/A')}")

    with tab_evidence:
        support_col, contradict_col = st.columns(2)
        with support_col:
            st.markdown(f"**Supporting evidence ({len(hyp.supporting_evidence)})**")
            if hyp.supporting_evidence:
                for evidence in hyp.supporting_evidence[:10]:
                    pmid = str(evidence.get("pmid", "") or "")
                    year = evidence.get("year")
                    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
                    pill = (
                        f'<a class="evidence-pill" href="{link}" target="_blank">PMID {escape_html(pmid or "N/A")}</a>'
                    )
                    year_text = f" · {year}" if year else ""
                    st.markdown(f"{pill} {escape_html(evidence.get('sentence', ''))}{escape_html(year_text)}", unsafe_allow_html=True)
            else:
                st.caption("No supporting evidence extracted.")
        with contradict_col:
            st.markdown(f"**Contradicting evidence ({len(hyp.contradicting_evidence)})**")
            if hyp.contradicting_evidence:
                for evidence in hyp.contradicting_evidence[:10]:
                    pmid = str(evidence.get("pmid", "") or "")
                    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
                    pill = (
                        f'<a class="evidence-pill" style="background:rgba(199,62,58,0.12);color:#C73E3A;border-color:rgba(199,62,58,0.24);" '
                        f'href="{link}" target="_blank">PMID {escape_html(pmid or "N/A")}</a>'
                    )
                    st.markdown(f"{pill} {escape_html(evidence.get('sentence', ''))}", unsafe_allow_html=True)
            else:
                st.caption("No contradicting evidence identified.")

    with tab_method:
        st.markdown(f"**Generation method**  \n{escape_html(hyp.method or 'N/A')}")
        st.markdown(f"**Limitations**  \n{escape_html(hyp.limitations or 'N/A')}")
        st.markdown(f"**Generated at**  \n{escape_html(hyp.generated_at or 'N/A')}")
        st.markdown(f"**Hypothesis ID**  \n`{escape_html(hyp.hypothesis_id)}`")

    st.download_button(
        "Export individual hypothesis",
        data=_serialize_json(dataclasses.asdict(hyp)),
        file_name=f"bionova_hypothesis_{index + 1}.json",
        mime="application/json",
        key=f"export_hypothesis_{index}",
    )
    st.divider()


def _render_hypotheses_page() -> None:
    """Generate and display ranked hypotheses."""
    _render_page_header(
        "HYPOTHESES",
        "Generate ranked, evidence-grounded scientific hypotheses from prioritized graph gaps and inspect the rationale behind each proposal.",
    )

    gap_candidates = st.session_state.get("gap_candidates", [])
    relations = st.session_state.get("relations", [])
    query = st.session_state.get("query", "")
    cfg = _get_runtime_config()

    if not gap_candidates:
        st.markdown(
            get_empty_state_html(
                "💡",
                "No candidate gaps available",
                "Run DISCOVER and review RESEARCH GAPS before generating hypotheses.",
            ),
            unsafe_allow_html=True,
        )
        return

    with st.expander("Generation settings", expanded=True):
        col_n, col_llm, col_btn = st.columns([1.3, 1.3, 1])
        with col_n:
            top_k = st.slider(
                "Top gap candidates to process",
                min_value=1,
                max_value=min(len(gap_candidates), 20),
                value=min(5, len(gap_candidates)),
                key="hyp_top_k",
            )
        with col_llm:
            use_llm = st.checkbox(
                "Use Gemini LLM",
                value=bool(cfg.gemini_api_key),
                key="hyp_use_llm",
                help="Uses the configured Gemini API key when available.",
            )
            if use_llm and not cfg.gemini_api_key:
                st.warning("Add a Gemini API key in Settings to enable LLM-assisted hypothesis drafting.")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("Generate hypotheses", type="primary", use_container_width=True)

    if generate_btn:
        _generate_hypotheses(
            gap_candidates=gap_candidates,
            relations=relations,
            query=query,
            cfg=cfg,
            top_k=int(top_k),
            use_llm=bool(use_llm),
        )

    hypotheses = st.session_state.get("hypotheses", [])
    if not hypotheses:
        st.markdown(
            get_empty_state_html(
                "🧪",
                "No hypotheses generated yet",
                "Use the generation controls above to create ranked scientific hypotheses from the current research gaps.",
            ),
            unsafe_allow_html=True,
        )
        return

    summary_cols = st.columns(4)
    summary_metrics = [
        ("Hypothesis count", len(hypotheses), "Current ranking set"),
        ("Avg final score", f"{mean(float(h.final_score or 0.0) for h in hypotheses):.2f}", "Overall strength"),
        ("Avg evidence", f"{mean(float(h.evidence_score or 0.0) for h in hypotheses):.2f}", "Support quality"),
        ("Avg novelty", f"{mean(float(h.novelty_score or 0.0) for h in hypotheses):.2f}", "Discovery signal"),
    ]
    for col, (label, value, detail) in zip(summary_cols, summary_metrics):
        with col:
            _render_metric_card(label, value, detail)

    sort_by = st.selectbox("Sort by", ["Final Score", "Evidence", "Novelty", "Plausibility"])
    sort_key_map = {
        "Final Score": "final_score",
        "Evidence": "evidence_score",
        "Novelty": "novelty_score",
        "Plausibility": "plausibility_score",
    }
    sort_key = sort_key_map[sort_by]
    sorted_hypotheses = sorted(hypotheses, key=lambda hyp: getattr(hyp, sort_key, 0.0), reverse=True)

    st.download_button(
        "Bulk export all hypotheses",
        data=_serialize_json([dataclasses.asdict(hyp) for hyp in hypotheses]),
        file_name="bionova_hypotheses.json",
        mime="application/json",
    )

    for index, hypothesis in enumerate(sorted_hypotheses):
        _render_hypothesis_card(hypothesis, index)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE
# ═══════════════════════════════════════════════════════════════════════════════
page_key = _render_sidebar()

if page_key == "DISCOVER":
    _render_discover_page()
elif page_key == "EVIDENCE":
    _render_evidence_page()
elif page_key == "KNOWLEDGE GRAPH":
    _render_graph_page()
elif page_key == "RESEARCH GAPS":
    _render_research_gaps_page()
elif page_key == "HYPOTHESES":
    _render_hypotheses_page()
