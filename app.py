"""
BioNOVA — Streamlit application entry point.

Run with:
    streamlit run app.py

Architecture: The UI contains no scientific algorithms.
All computation is delegated to retrieval, extraction, graph, hypothesis, utils.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
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
from utils import setup_logging, BioNOVAConfig  # noqa: E402

_logger = setup_logging()

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def _load_config() -> BioNOVAConfig:
    return BioNOVAConfig.load_env()


# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "page": "Search",
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
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CSS theme ─────────────────────────────────────────────────────────────────
_CSS = """
<style>
body, .stApp { background-color: #0A0F1E; color: #F9FAFB; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stButton>button { border-radius: 8px; font-weight: 600; }
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #3B82F6, #6366F1);
    border: none; color: white;
}
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background: #1C2539; color: #F9FAFB; border-color: #2D3A52;
    border-radius: 8px;
}
.stSelectbox>div>div { background: #1C2539; color: #F9FAFB; }
.metric-card {
    background: #111827; border: 1px solid #1F2937;
    border-radius: 12px; padding: 18px 22px; margin-bottom: 8px;
}
.hyp-card {
    background: #111827; border: 1px solid #1F2937;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 12px;
}
.evidence-pill {
    display: inline-block; background: #1E3A5F; color: #93C5FD;
    border-radius: 999px; padding: 2px 10px; font-size: 0.75rem;
    margin: 2px;
}
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #F9FAFB;
    margin-bottom: 0.25rem;
}
.section-sub {
    font-size: 0.875rem; color: #9CA3AF; margin-bottom: 1.25rem;
}
/* Sidebar */
section[data-testid="stSidebar"] { background-color: #111827; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "🔍 Search",
    "📄 Literature",
    "🧬 Knowledge Graph",
    "🔎 Discovery",
    "💡 Hypotheses",
    "📤 Export",
]

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.5rem 0;">
          <div style="font-size:1.6rem;font-weight:800;
               background:linear-gradient(135deg,#3B82F6,#9B72CF);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">BioNOVA</div>
          <div style="font-size:0.7rem;color:#6B7280;letter-spacing:0.08em;
               text-transform:uppercase;margin-top:2px;">
            Biomedical Literature Mining
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="nav_radio",
    )
    page_key = selected.split(" ", 1)[1].strip()

    if st.session_state.get("pipeline_ran"):
        st.divider()
        meta = st.session_state.get("analysis_metadata", {})
        st.markdown(
            f"""
            <div style="font-size:0.75rem;color:#9CA3AF;line-height:1.8;">
              <b style="color:#F9FAFB;">Last run</b><br>
              Query: <i>{str(st.session_state.get('query',''))[:40]}</i><br>
              Articles: {meta.get('n_articles', 0)}<br>
              Entities: {meta.get('n_entities', 0)}<br>
              Relations: {meta.get('n_relations', 0)}<br>
              Graph nodes: {meta.get('n_nodes', 0)}<br>
              Gaps found: {meta.get('n_gaps', 0)}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Page router (defined after all page functions below) ─────────────────────
# This is intentionally a forward-reference marker; the actual router call
# is at the bottom of this file after all _render_* functions are defined.


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_search_page():
    """Literature search and pipeline execution."""
    st.markdown(
        '<div class="section-header">Literature Search</div>'
        '<div class="section-sub">Enter a biomedical query to retrieve, '
        'extract, and analyse PubMed literature.</div>',
        unsafe_allow_html=True,
    )

    cfg = _load_config()

    # ── Query input ───────────────────────────────────────────────────────────
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        query = st.text_input(
            "Research query",
            value=st.session_state.get("query", ""),
            placeholder="e.g. BRCA1 breast cancer DNA repair",
            label_visibility="visible",
            key="search_query_input",
        )
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            max_results = st.slider(
                "Max articles",
                min_value=10,
                max_value=300,
                value=100,
                step=10,
                key="search_max_results",
            )
        with col_b:
            cutoff_year = st.number_input(
                "Historical cutoff year (optional)",
                min_value=1990,
                max_value=2030,
                value=0,
                step=1,
                help="If set, only literature up to this year is used for discovery.",
                key="search_cutoff_year",
            )
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            use_demo = st.checkbox("Use demo data", value=False, key="use_demo")

        run = st.button(
            "▶  Run BioNOVA Pipeline",
            type="primary",
            use_container_width=True,
            key="search_run_btn",
        )

    if run:
        q = query.strip()
        if not q and not use_demo:
            st.warning("Please enter a query or enable demo data.")
            return
        _run_pipeline(
            query=q,
            max_results=int(max_results),
            cutoff_year=int(cutoff_year) if cutoff_year else None,
            use_demo=bool(use_demo),
            cfg=cfg,
        )

    if st.session_state.get("pipeline_ran") and not st.session_state.get("pipeline_running"):
        meta = st.session_state.get("analysis_metadata", {})
        _, mid2, _ = st.columns([1, 3, 1])
        with mid2:
            st.success(
                f"✅ Pipeline complete — {meta.get('n_articles', 0)} articles · "
                f"{meta.get('n_entities', 0)} entities · "
                f"{meta.get('n_relations', 0)} relations · "
                f"{meta.get('n_nodes', 0)} graph nodes · "
                f"{meta.get('n_gaps', 0)} gap candidates"
            )
            st.info("Use the sidebar to navigate to any view.")


def _run_pipeline(
    query: str,
    max_results: int,
    cutoff_year: Optional[int],
    use_demo: bool,
    cfg: BioNOVAConfig,
) -> None:
    """Execute the BioNOVA pipeline with live progress display."""
    import time

    from retrieval import (
        PubMedRetriever, DataCleaner, create_sample_articles,
        MockRetriever, retrieve_and_clean,
    )
    from extraction import ExtractionPipeline
    from graph import BioNOVAGraph, build_graph

    st.session_state["pipeline_running"] = True
    st.session_state["query"] = query
    st.session_state["cutoff_year"] = cutoff_year

    progress_bar = st.progress(0.0, text="Initialising…")
    status = st.empty()

    try:
        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        status.info("📡 Retrieving literature from PubMed…")
        progress_bar.progress(0.05, text="Retrieving articles…")

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
                        text=f"Fetching {fetched}/{total} articles…",
                    ),
                )
            except Exception as exc:
                _logger.warning("PubMed retrieval failed, using demo: %s", exc)
                st.warning(f"PubMed retrieval failed ({exc}). Using demo data.")
                articles = create_sample_articles()

        if not articles:
            st.error("No articles retrieved. Check query or enable demo data.")
            return

        progress_bar.progress(0.30, text=f"Retrieved {len(articles)} articles.")

        # ── Step 2: Apply cutoff ──────────────────────────────────────────────
        if cutoff_year:
            articles = [a for a in articles if a.year is None or a.year <= cutoff_year]
            _logger.info("After cutoff %d: %d articles", cutoff_year, len(articles))

        # ── Step 3: Extract entities & relations ──────────────────────────────
        status.info("🔬 Extracting biomedical entities and relationships…")
        progress_bar.progress(0.35, text="Extracting entities…")

        extractor = ExtractionPipeline(use_scispacy=False)
        entities, relations = extractor.process_articles(articles)

        progress_bar.progress(0.60, text=f"{len(entities)} entities, {len(relations)} relations.")

        # ── Step 4: Build knowledge graph ─────────────────────────────────────
        status.info("🕸  Building knowledge graph…")
        progress_bar.progress(0.65, text="Building graph…")

        bg = build_graph(relations, cutoff_year=cutoff_year)
        stats = bg.get_statistics()

        progress_bar.progress(0.80, text="Graph built.")

        # ── Step 5: Find gap candidates ───────────────────────────────────────
        status.info("🔎 Discovering gap candidates…")
        gap_candidates = bg.find_gap_candidates(cutoff_year=cutoff_year, top_k=50)

        progress_bar.progress(0.95, text=f"{len(gap_candidates)} gap candidates found.")

        # ── Persist to session ────────────────────────────────────────────────
        st.session_state["articles"] = articles
        st.session_state["entities"] = entities
        st.session_state["relations"] = relations
        st.session_state["bionova_graph"] = bg
        st.session_state["gap_candidates"] = gap_candidates
        st.session_state["hypotheses"] = []  # reset; user can re-generate
        st.session_state["analysis_metadata"] = {
            "query": query,
            "n_articles": len(articles),
            "n_entities": len(entities),
            "n_relations": len(relations),
            "n_nodes": stats.get("n_nodes", 0),
            "n_edges": stats.get("n_edges", 0),
            "n_biological_edges": stats.get("n_biological_edges", 0),
            "n_gaps": len(gap_candidates),
            "cutoff_year": cutoff_year,
        }
        st.session_state["pipeline_ran"] = True

        progress_bar.progress(1.0, text="Pipeline complete.")
        status.empty()

    except Exception as exc:
        _logger.exception("Pipeline failed: %s", exc)
        st.error(f"Pipeline error: {exc}")
    finally:
        st.session_state["pipeline_running"] = False


def _render_literature_page():
    """Display retrieved articles."""
    st.markdown(
        '<div class="section-header">Literature</div>',
        unsafe_allow_html=True,
    )
    articles = st.session_state.get("articles", [])
    if not articles:
        st.info("Run the pipeline on the **Search** page first.")
        return

    st.markdown(
        f'<div class="section-sub">{len(articles)} articles retrieved · '
        f'Query: <i>{st.session_state.get("query","")}</i></div>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("Filter articles", placeholder="Search title/abstract…", key="lit_filter")

    displayed = articles
    if search_term:
        st_lower = search_term.lower()
        displayed = [
            a for a in articles
            if st_lower in (a.title or "").lower() or st_lower in (a.abstract or "").lower()
        ]

    st.markdown(f"Showing {len(displayed)} articles")

    for i, art in enumerate(displayed[:100]):
        with st.expander(f"[{art.pmid}] {art.title[:90]}…" if art.title and len(art.title) > 90 else f"[{art.pmid}] {art.title}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Abstract:** {art.abstract[:600]}{'…' if len(art.abstract or '') > 600 else ''}")
            with col2:
                st.markdown(f"**Year:** {art.year or 'N/A'}")
                st.markdown(f"**Journal:** {art.journal or 'N/A'}")
                if art.doi:
                    st.markdown(f"**DOI:** [{art.doi}](https://doi.org/{art.doi})")
                if art.mesh_terms:
                    st.markdown("**MeSH:** " + ", ".join(art.mesh_terms[:5]))

    if len(displayed) > 100:
        st.info(f"Showing first 100 of {len(displayed)} articles.")


def _render_graph_page():
    """Display knowledge graph statistics and edge list."""
    st.markdown(
        '<div class="section-header">Knowledge Graph</div>',
        unsafe_allow_html=True,
    )
    bg = st.session_state.get("bionova_graph")
    if bg is None:
        st.info("Run the pipeline on the **Search** page first.")
        return

    stats = bg.get_statistics()

    # ── Summary metrics ───────────────────────────────────────────────────────
    cols = st.columns(4)
    for col, (label, val) in zip(cols, [
        ("Nodes", stats.get("n_nodes", 0)),
        ("Biological Edges", stats.get("n_biological_edges", 0)),
        ("Semantic Edges", stats.get("n_semantic_edges", 0)),
        ("Avg Degree", f"{stats.get('avg_degree', 0):.2f}"),
    ]):
        col.metric(label, val)

    # ── Top nodes ─────────────────────────────────────────────────────────────
    st.subheader("Top Entities by Degree")
    top_nodes = stats.get("top_nodes_by_degree", [])
    if top_nodes:
        import pandas as pd
        df = pd.DataFrame(top_nodes, columns=["Entity", "Degree"])
        st.dataframe(df, use_container_width=True)

    # ── Edge list ─────────────────────────────────────────────────────────────
    st.subheader("Biological Edges (sample)")
    rels = st.session_state.get("relations", [])
    bio_rels = [r for r in rels if r.polarity != "negative"][:200]

    if bio_rels:
        import pandas as pd
        rows = [
            {
                "Subject": r.subject,
                "Predicate": r.predicate,
                "Object": r.object,
                "Polarity": r.polarity,
                "Certainty": r.certainty,
                "Confidence": f"{r.confidence:.2f}",
                "PMID": r.pmid,
                "Year": r.publication_year or "",
            }
            for r in bio_rels
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No biological relations extracted.")


def _render_discovery_page():
    """Display gap candidates and novelty verification."""
    st.markdown(
        '<div class="section-header">Gap Discovery & Novelty Verification</div>'
        '<div class="section-sub">Candidate relationships not yet directly '
        'supported in the literature.</div>',
        unsafe_allow_html=True,
    )

    from graph import NoveltyStatus

    bg = st.session_state.get("bionova_graph")
    gap_candidates = st.session_state.get("gap_candidates", [])

    if bg is None:
        st.info("Run the pipeline on the **Search** page first.")
        return

    cutoff_year = st.session_state.get("cutoff_year")

    if not gap_candidates:
        st.info("No gap candidates found. Try increasing Max Articles or using a broader query.")
        return

    # ── Novelty status breakdown ──────────────────────────────────────────────
    from collections import Counter
    status_counts = Counter(c.novelty_status.name for c in gap_candidates)

    cols = st.columns(len(status_counts))
    status_colors = {
        "UNEXPLORED": "#10B981",
        "INDIRECTLY_SUPPORTED": "#3B82F6",
        "PARTIALLY_SUPPORTED": "#F59E0B",
        "KNOWN": "#6B7280",
        "CONTRADICTED": "#EF4444",
        "INSUFFICIENT_EVIDENCE": "#9CA3AF",
    }
    for col, (status, count) in zip(cols, status_counts.items()):
        color = status_colors.get(status, "#9CA3AF")
        col.markdown(
            f'<div class="metric-card" style="border-color:{color}33;">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{color};">{count}</div>'
            f'<div style="font-size:0.75rem;color:#9CA3AF;">{status.replace("_", " ")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Candidate table ───────────────────────────────────────────────────────
    st.subheader(f"Top {min(len(gap_candidates), 50)} Gap Candidates")

    filter_status = st.multiselect(
        "Filter by novelty status",
        list(status_colors.keys()),
        default=["UNEXPLORED", "INDIRECTLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"],
        key="disc_filter_status",
    )

    filtered = [
        c for c in gap_candidates
        if c.novelty_status.name in filter_status
    ]

    if not filtered:
        st.info("No candidates match the selected filters.")
        return

    import pandas as pd
    rows = []
    for c in filtered[:50]:
        rows.append({
            "Entity A": c.node_a,
            "Type A": c.node_a_type,
            "Entity B": c.node_b,
            "Type B": c.node_b_type,
            "Score": f"{c.score:.4f}",
            "Method": c.method,
            "Shared Neighbors": ", ".join(c.common_neighbors[:5]),
            "Novelty": c.novelty_status.name,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_hypotheses_page():
    """Generate and display ranked hypotheses."""
    st.markdown(
        '<div class="section-header">Hypotheses</div>'
        '<div class="section-sub">Evidence-grounded hypotheses generated from '
        'gap candidates. Every hypothesis includes supporting/contradicting '
        'evidence, a testable prediction, and a falsifiability statement.</div>',
        unsafe_allow_html=True,
    )

    gap_candidates = st.session_state.get("gap_candidates", [])
    relations = st.session_state.get("relations", [])
    query = st.session_state.get("query", "")
    cfg = _load_config()

    if not gap_candidates:
        st.info("Run the pipeline on the **Search** page first.")
        return

    # ── Generation controls ───────────────────────────────────────────────────
    with st.expander("⚙️ Generation Settings", expanded=True):
        col_n, col_llm, col_btn = st.columns([2, 2, 1])
        with col_n:
            top_k = st.slider(
                "Number of gap candidates to process",
                min_value=1,
                max_value=min(len(gap_candidates), 20),
                value=min(5, len(gap_candidates)),
                key="hyp_top_k",
            )
        with col_llm:
            use_llm = st.checkbox(
                "Use Gemini LLM (requires API key)",
                value=bool(cfg.gemini_api_key),
                key="hyp_use_llm",
            )
            if use_llm and not cfg.gemini_api_key:
                st.warning("Set GEMINI_API_KEY in .env to use LLM generation.")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button(
                "✨ Generate",
                type="primary",
                use_container_width=True,
                key="hyp_generate_btn",
            )

    if generate_btn:
        _generate_hypotheses(
            gap_candidates=gap_candidates,
            relations=relations,
            query=query,
            cfg=cfg,
            top_k=int(top_k),
            use_llm=bool(use_llm),
        )

    # ── Display hypotheses ────────────────────────────────────────────────────
    hypotheses = st.session_state.get("hypotheses", [])
    if not hypotheses:
        if st.session_state.get("pipeline_ran"):
            st.info("Click **Generate** above to create hypotheses.")
        return

    st.divider()
    st.subheader(f"{len(hypotheses)} Ranked Hypotheses")

    sort_by = st.selectbox(
        "Sort by",
        ["Final Score ↓", "Evidence Score ↓", "Novelty Score ↓"],
        key="hyp_sort",
    )

    sort_key_map = {
        "Final Score ↓": "final_score",
        "Evidence Score ↓": "evidence_score",
        "Novelty Score ↓": "novelty_score",
    }
    sk = sort_key_map[sort_by]
    sorted_hyps = sorted(hypotheses, key=lambda h: getattr(h, sk, 0.0), reverse=True)

    for i, hyp in enumerate(sorted_hyps):
        _render_hypothesis_card(hyp, index=i)


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
        progress.progress(1.0)
        st.session_state["hypotheses"] = hypotheses
        st.success(f"Generated {len(hypotheses)} hypotheses.")
        st.rerun()
    except Exception as exc:
        progress.empty()
        _logger.exception("Hypothesis generation failed: %s", exc)
        st.error(f"Hypothesis generation failed: {exc}")


def _render_hypothesis_card(hyp, index: int):
    """Render a single hypothesis card with full evidence display."""
    score_color = (
        "#10B981" if hyp.final_score >= 0.7
        else "#F59E0B" if hyp.final_score >= 0.4
        else "#EF4444"
    )
    novelty_color = {
        "UNEXPLORED": "#10B981",
        "INDIRECTLY_SUPPORTED": "#3B82F6",
        "PARTIALLY_SUPPORTED": "#F59E0B",
        "KNOWN": "#6B7280",
        "CONTRADICTED": "#EF4444",
        "INSUFFICIENT_EVIDENCE": "#9CA3AF",
    }.get(hyp.novelty_status, "#9CA3AF")

    with st.expander(
        f"H{index+1}: {hyp.hypothesis[:100]}{'…' if len(hyp.hypothesis) > 100 else ''}",
        expanded=(index == 0),
    ):
        # Score bar
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Score", f"{hyp.final_score:.2f}")
        c2.metric("Evidence", f"{hyp.evidence_score:.2f}")
        c3.metric("Novelty", f"{hyp.novelty_score:.2f}")
        c4.metric("Plausibility", f"{hyp.plausibility_score:.2f}")
        c5.metric("Feasibility", f"{hyp.feasibility_score:.2f}")

        st.markdown(f"**Novelty Status:** "
                    f'<span style="color:{novelty_color};">{hyp.novelty_status}</span>',
                    unsafe_allow_html=True)

        tab_main, tab_evidence, tab_method = st.tabs(
            ["Hypothesis", "Evidence", "Methods & Limitations"]
        )

        with tab_main:
            st.markdown(f"**Hypothesis:** {hyp.hypothesis}")
            st.markdown(f"**Entities:** {', '.join(hyp.entities)}")
            st.markdown(f"**Mechanistic Rationale:** {hyp.mechanistic_rationale}")
            st.markdown(f"**Testable Prediction:** {hyp.testable_prediction}")
            st.markdown(f"**Suggested Experiment:** {hyp.suggested_experiment}")
            st.markdown(f"**Falsifiability:** {hyp.falsifiability}")
            st.markdown(f"**Alternative Explanations:** {hyp.alternative_explanations}")
            st.markdown(f"**Devil's Advocate:** {hyp.devils_advocate}")

        with tab_evidence:
            col_sup, col_con = st.columns(2)
            with col_sup:
                st.markdown(f"**Supporting Evidence ({len(hyp.supporting_evidence)})**")
                for ev in hyp.supporting_evidence[:10]:
                    st.markdown(
                        f'<div class="evidence-pill">[{ev.get("pmid","")}]</div> '
                        f'{ev.get("sentence","")[:200]}',
                        unsafe_allow_html=True,
                    )
            with col_con:
                st.markdown(f"**Contradicting Evidence ({len(hyp.contradicting_evidence)})**")
                if hyp.contradicting_evidence:
                    for ev in hyp.contradicting_evidence[:5]:
                        st.markdown(
                            f'<div class="evidence-pill" style="background:#3B1A1A;'
                            f'color:#FCA5A5;">[{ev.get("pmid","")}]</div> '
                            f'{ev.get("sentence","")[:200]}',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("*No contradicting evidence found.*")

        with tab_method:
            st.markdown(f"**Limitations:** {hyp.limitations}")
            st.markdown(f"**Generation Method:** {hyp.method}")
            st.markdown(f"**Generated At:** {hyp.generated_at}")
            st.markdown(f"**Hypothesis ID:** `{hyp.hypothesis_id}`")


def _render_export_page():
    """Export results as JSON."""
    st.markdown(
        '<div class="section-header">Export</div>'
        '<div class="section-sub">Download analysis results.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("pipeline_ran"):
        st.info("Run the pipeline first.")
        return

    col1, col2, col3 = st.columns(3)

    # ── Articles ──────────────────────────────────────────────────────────────
    with col1:
        articles = st.session_state.get("articles", [])
        if articles:
            data = [
                {
                    "pmid": a.pmid, "title": a.title, "abstract": a.abstract,
                    "year": a.year, "journal": a.journal, "authors": a.authors,
                }
                for a in articles
            ]
            st.download_button(
                "📄 Download Articles (JSON)",
                data=json.dumps(data, indent=2),
                file_name="bionova_articles.json",
                mime="application/json",
            )

    # ── Relations ─────────────────────────────────────────────────────────────
    with col2:
        relations = st.session_state.get("relations", [])
        if relations:
            import dataclasses
            data = [dataclasses.asdict(r) for r in relations[:1000]]
            st.download_button(
                "🧬 Download Relations (JSON)",
                data=json.dumps(data, indent=2, default=str),
                file_name="bionova_relations.json",
                mime="application/json",
            )

    # ── Hypotheses ────────────────────────────────────────────────────────────
    with col3:
        hypotheses = st.session_state.get("hypotheses", [])
        if hypotheses:
            import dataclasses
            data = [dataclasses.asdict(h) for h in hypotheses]
            st.download_button(
                "💡 Download Hypotheses (JSON)",
                data=json.dumps(data, indent=2, default=str),
                file_name="bionova_hypotheses.json",
                mime="application/json",
            )
        else:
            st.info("No hypotheses generated yet.")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = st.session_state.get("analysis_metadata", {})
    if meta:
        st.subheader("Analysis Metadata")
        st.json(meta)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE — placed after all function definitions so all callables exist
# ═══════════════════════════════════════════════════════════════════════════════
if page_key == "Search":
    _render_search_page()
elif page_key == "Literature":
    _render_literature_page()
elif page_key == "Knowledge Graph":
    _render_graph_page()
elif page_key == "Discovery":
    _render_discovery_page()
elif page_key == "Hypotheses":
    _render_hypotheses_page()
elif page_key == "Export":
    _render_export_page()
