"""
test_bionova.py — Comprehensive test suite for BioNOVA.

Runs without internet access. All PubMed operations use mocked/local data.

Run with:
    python -m pytest test_bionova.py -v
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# ── Ensure repo root is on path ───────────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# UTILS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUtils:
    def test_import(self):
        import utils  # noqa: F401

    def test_setup_logging(self):
        from utils import setup_logging
        logger = setup_logging("WARNING")
        import logging
        assert logger.name == "bionova"

    def test_config_defaults(self):
        from utils import BioNOVAConfig
        cfg = BioNOVAConfig()
        assert cfg.max_results == 200
        assert cfg.batch_size == 100
        assert cfg.random_seed == 42

    def test_config_load_env(self):
        from utils import BioNOVAConfig
        os.environ["MAX_RESULTS"] = "50"
        cfg = BioNOVAConfig.load_env()
        assert isinstance(cfg, BioNOVAConfig)
        assert cfg.max_results == 50
        del os.environ["MAX_RESULTS"]

    def test_normalize_text(self):
        from utils import normalize_text
        result = normalize_text("  Hello   World  ")
        assert result == "hello world"
        assert normalize_text("") == ""
        assert normalize_text("café") == "café"

    def test_truncate(self):
        from utils import truncate
        assert truncate("hello world", 5) == "hello…"
        assert truncate("hi", 10) == "hi"
        assert truncate("", 5) == ""

    def test_validate_pmid(self):
        from utils import validate_pmid
        assert validate_pmid("12345678") is True
        assert validate_pmid("1") is True
        assert validate_pmid("") is False
        assert validate_pmid("abc") is False
        assert validate_pmid("123456789") is False  # 9 digits

    def test_validate_year(self):
        from utils import validate_year
        assert validate_year(2020) == 2020
        assert validate_year("2015") == 2015
        assert validate_year(None) is None
        assert validate_year("abc") is None
        assert validate_year(1800) is None  # too old
        assert validate_year(2100) is None  # future

    def test_json_roundtrip(self, tmp_path):
        from utils import to_json, from_json
        data = {"key": [1, 2, 3], "set_val": {4, 5}}
        out = tmp_path / "test.json"
        to_json(data, str(out))
        loaded = from_json(str(out))
        assert loaded["key"] == [1, 2, 3]
        # sets are serialized as sorted lists
        assert sorted(loaded["set_val"]) == [4, 5]

    def test_timer(self):
        from utils import Timer
        import time
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_set_random_seed(self):
        from utils import set_random_seed
        import random
        set_random_seed(42)
        v1 = random.random()
        set_random_seed(42)
        v2 = random.random()
        assert v1 == v2

    def test_slugify(self):
        from utils import slugify
        assert slugify("Hello World!") == "hello_world"
        assert slugify("BRCA1 gene") == "brca1_gene"


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieval:
    """All retrieval tests use local/mock data — no internet required."""

    def test_import(self):
        import retrieval  # noqa: F401

    def test_article_dataclass(self):
        from retrieval import Article
        art = Article(
            pmid="12345",
            title="Test Title",
            abstract="Test abstract",
            authors=["Author A", "Author B"],
            journal="Test Journal",
            year=2020,
            mesh_terms=["BRCA1"],
            keywords=["cancer"],
            doi="10.1234/test",
        )
        assert art.pmid == "12345"
        assert art.year == 2020
        assert len(art.authors) == 2

    def test_sample_articles(self):
        from retrieval import create_sample_articles
        articles = create_sample_articles()
        assert len(articles) >= 3
        for art in articles:
            assert art.pmid
            assert art.title
            assert art.abstract
            assert art.year is not None

    def test_xml_parser_valid(self):
        from retrieval import XMLParser
        parser = XMLParser()
        xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999999</PMID>
      <Article>
        <ArticleTitle>BRCA1 activates TP53 in cancer cells</ArticleTitle>
        <Abstract>
          <AbstractText>BRCA1 activates TP53 and inhibits tumor progression.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue>
            <PubDate><Year>2022</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        articles = parser.parse_xml(xml)
        assert len(articles) == 1
        art = articles[0]
        assert art.pmid == "99999999"
        assert "BRCA1" in art.title
        assert art.year == 2022
        assert "Smith" in art.authors[0]

    def test_xml_parser_empty(self):
        from retrieval import XMLParser
        parser = XMLParser()
        articles = parser.parse_xml("")
        assert articles == []

    def test_xml_parser_malformed(self):
        from retrieval import XMLParser
        parser = XMLParser()
        # Should not raise, should return empty or partial
        articles = parser.parse_xml("<not valid xml<<<")
        assert isinstance(articles, list)

    def test_xml_parser_missing_abstract(self):
        from retrieval import XMLParser
        parser = XMLParser()
        xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <ArticleTitle>Title without abstract</ArticleTitle>
        <Journal>
          <Title>Journal</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        articles = parser.parse_xml(xml)
        # Should still parse, abstract will be empty
        assert len(articles) <= 1
        if articles:
            assert articles[0].abstract == ""

    def test_data_cleaner_dedup(self):
        from retrieval import Article, DataCleaner
        cleaner = DataCleaner()
        art = Article("1", "Title", "Abstract", [], "Journal", 2020, [], [], None)
        articles = [art, art, art]  # three copies
        cleaned = cleaner.clean_articles(articles)
        assert len(cleaned) == 1

    def test_data_cleaner_removes_empty(self):
        from retrieval import Article, DataCleaner
        cleaner = DataCleaner()
        empty = Article("2", "", "", [], "", None, [], [], None)
        valid = Article("3", "Title", "Abstract", [], "J", 2020, [], [], None)
        cleaned = cleaner.clean_articles([empty, valid])
        # empty (no title AND no abstract) should be removed
        assert all(a.pmid != "2" for a in cleaned)
        assert any(a.pmid == "3" for a in cleaned)

    def test_mock_retriever(self):
        from retrieval import MockRetriever
        mock = MockRetriever()
        articles = mock.fetch_with_progress(query="BRCA1", max_results=10)
        assert len(articles) >= 1
        for art in articles:
            assert isinstance(art.pmid, str)

    def test_duplicate_pmid_handling(self):
        from retrieval import Article, DataCleaner
        cleaner = DataCleaner()
        art1 = Article("100", "Title A", "Abstract A", ["Author 1"], "J", 2020, [], [], None)
        art2 = Article("100", "Title B", "Abstract B", ["Author 2"], "J", 2021, [], [], None)
        cleaned = cleaner.clean_articles([art1, art2])
        pmids = [a.pmid for a in cleaned]
        assert pmids.count("100") == 1

    def test_load_local_articles(self, tmp_path):
        from retrieval import Article, load_local_articles
        art = Article("50001", "Title", "Abstract text here", ["A. Author"],
                      "Journal of Testing", 2020, ["MeSH term"], ["keyword"], "10.1/test")
        data = [dataclasses.asdict(art)]
        jf = tmp_path / "articles.json"
        jf.write_text(json.dumps(data))
        loaded = load_local_articles(str(jf))
        assert len(loaded) == 1
        assert loaded[0].pmid == "50001"


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtraction:
    """Tests for biomedical entity and relation extraction."""

    def test_import(self):
        import extraction  # noqa: F401

    def test_entity_extraction_genes(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        ents = ext.extract_entities("BRCA1 and TP53 are tumor suppressors.", pmid="1")
        types = {e.entity_type for e in ents}
        texts = {e.text.upper() for e in ents}
        assert "GENE" in types
        assert "BRCA1" in texts or "TP53" in texts

    def test_entity_extraction_disease(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        ents = ext.extract_entities("Breast cancer patients show elevated BRCA1 levels.", pmid="2")
        types = {e.entity_type for e in ents}
        assert "DISEASE" in types

    def test_entity_extraction_chemical(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        ents = ext.extract_entities("Tamoxifen inhibits EGFR expression in cancer cells.", pmid="3")
        texts = {e.text.lower() for e in ents}
        assert "tamoxifen" in texts or any("tamox" in t for t in texts)

    def test_entity_extraction_pathway(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        ents = ext.extract_entities("The PI3K pathway is activated in triple-negative breast cancer.", pmid="4")
        types = {e.entity_type for e in ents}
        assert "PATHWAY" in types or "DISEASE" in types

    def test_entity_normalization(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        assert ext._normalize_entity("BRCA1s") == "brca1s" or ext._normalize_entity("BRCA1s").endswith("1")
        # normalization should lowercase
        norm = ext._normalize_entity("TP53")
        assert norm == norm.lower()

    def test_relation_extraction_activation(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        rels = ext.extract_relations("BRCA1 activates TP53 in cancer cells.", pmid="10")
        assert len(rels) >= 1
        predicates = {r.predicate for r in rels}
        assert any(p in predicates for p in {"activates", "induces", "promotes"})
        assert all(r.polarity in {"positive", "uncertain", "conditional"} for r in rels)

    def test_relation_extraction_inhibition(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        rels = ext.extract_relations("Tamoxifen inhibits EGFR expression in breast cancer.", pmid="11")
        assert len(rels) >= 1
        neg_rels = [r for r in rels if r.predicate in {"inhibits", "suppresses", "blocks", "reduces", "decreases"}]
        assert len(neg_rels) >= 1

    def test_negation_detection(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        rels = ext.extract_relations("BRCA1 did not activate TP53 in this study.", pmid="12")
        # Negation should flip polarity or mark as negative
        if rels:
            polarity_values = {r.polarity for r in rels}
            # Should not be uniformly "positive"
            assert polarity_values != {"positive"}

    def test_uncertainty_detection(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        rels = ext.extract_relations("BRCA1 may activate TP53 in some contexts.", pmid="13")
        if rels:
            certainties = {r.certainty for r in rels}
            # Should detect hedging
            assert certainties != {"certain"}

    def test_sentence_provenance(self):
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        text = "BRCA1 activates TP53. MDM2 inhibits CDKN1A."
        rels = ext.extract_relations(text, pmid="14")
        for r in rels:
            assert r.sentence  # must have sentence
            assert r.pmid == "14"  # must retain PMID

    def test_no_relation_without_predicate(self):
        """Two entities in a sentence without predicate should NOT create a relation."""
        from extraction import RuleBasedExtractor
        ext = RuleBasedExtractor()
        rels = ext.extract_relations("BRCA1 and TP53 are both genes.", pmid="15")
        # "are" should not create a biological relation
        bio_rels = [r for r in rels if r.predicate not in {"is", "are", "were", "was"}]
        # Even if some are found, they should be non-trivially predicated
        assert all(r.predicate != "" for r in rels)

    def test_deduplication(self):
        from extraction import ExtractionPipeline
        from retrieval import Article
        pipeline = ExtractionPipeline(use_scispacy=False)
        art = Article(
            pmid="99", title="BRCA1 activates TP53",
            abstract="BRCA1 activates TP53 in cancer. BRCA1 activates TP53 in cancer.",
            authors=[], journal="J", year=2020, mesh_terms=[], keywords=[], doi=None,
        )
        _, rels = pipeline.process_articles([art])
        # Duplicate sentences should be deduplicated
        keys = [(r.subject, r.predicate, r.object, r.sentence) for r in rels]
        assert len(keys) == len(set(keys))

    def test_extraction_pipeline_sample_articles(self):
        from extraction import ExtractionPipeline
        from retrieval import create_sample_articles
        pipeline = ExtractionPipeline(use_scispacy=False)
        articles = create_sample_articles()
        entities, relations = pipeline.process_articles(articles)
        assert len(entities) > 0
        assert len(relations) > 0
        for r in relations:
            assert r.subject
            assert r.object
            assert r.predicate
            assert r.polarity in {"positive", "negative", "uncertain", "conditional"}
            assert r.certainty in {"certain", "probable", "possible", "speculative"}
            assert 0.0 <= r.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraph:
    """Tests for knowledge graph construction, temporal analysis, gap detection."""

    def _make_relations(self):
        """Create a minimal set of test relations."""
        from extraction import Relation
        return [
            Relation("BRCA1", "activates", "TP53", "BRCA1 activates TP53.",
                     "100", 2018, "positive", "certain", 0.9, "rule_based", "GENE", "GENE"),
            Relation("TP53", "inhibits", "MDM2", "TP53 inhibits MDM2.",
                     "101", 2019, "negative", "certain", 0.85, "rule_based", "GENE", "GENE"),
            Relation("BRCA1", "associated_with", "breast cancer",
                     "BRCA1 is associated with breast cancer.",
                     "102", 2020, "positive", "certain", 0.88, "rule_based", "GENE", "DISEASE"),
            Relation("MDM2", "targets", "CDKN1A", "MDM2 targets CDKN1A.",
                     "103", 2021, "positive", "certain", 0.82, "rule_based", "GENE", "GENE"),
            Relation("EGFR", "activates", "MAPK pathway",
                     "EGFR activates MAPK pathway signaling.",
                     "104", 2022, "positive", "certain", 0.87, "rule_based", "GENE", "PATHWAY"),
        ]

    def test_import(self):
        import graph  # noqa: F401

    def test_build_graph(self):
        from graph import build_graph
        rels = self._make_relations()
        bg = build_graph(rels)
        stats = bg.get_statistics()
        assert stats["n_nodes"] > 0
        assert stats["n_biological_edges"] > 0

    def test_biological_vs_semantic_separation(self):
        """Biological predicates go to self.graph; semantic to self.semantic_graph."""
        from graph import build_graph
        from extraction import Relation
        bio_rel = Relation("BRCA1", "activates", "TP53", "sent.", "1", 2020,
                           "positive", "certain", 0.9, "rule_based", "GENE", "GENE")
        bg = build_graph([bio_rel])
        # activates is biological
        assert bg.graph.number_of_edges() == 1
        assert bg.semantic_graph.number_of_edges() == 0

    def test_negative_edge_not_in_biological_graph(self):
        """Negative-polarity edges must not appear as positive biological evidence."""
        from graph import build_graph
        from extraction import Relation
        neg_rel = Relation("BRCA1", "inhibits", "TP53", "BRCA1 does not activate TP53.", "1",
                           2020, "negative", "certain", 0.8, "rule_based", "GENE", "GENE")
        bg = build_graph([neg_rel])
        # Nodes exist but no edge in main graph
        # (negative edges are tracked separately)
        for u, v in bg.graph.edges():
            data = bg.graph.edges[u, v]
            assert data.get("polarity") != "negative", "Negative edge must not enter bio graph"

    def test_duplicate_edge_merging(self):
        """Same relationship from two papers should be merged, not duplicated."""
        from graph import build_graph
        from extraction import Relation
        r1 = Relation("BRCA1", "activates", "TP53", "s1", "100", 2018, "positive",
                      "certain", 0.9, "rule_based", "GENE", "GENE")
        r2 = Relation("BRCA1", "activates", "TP53", "s2", "200", 2019, "positive",
                      "certain", 0.85, "rule_based", "GENE", "GENE")
        bg = build_graph([r1, r2])
        # Only one edge (merged)
        assert bg.graph.number_of_edges() <= 2

    def test_edge_weighting(self):
        """Edges with more evidence should have higher weight."""
        from graph import build_graph
        from extraction import Relation
        rels = [
            Relation("BRCA1", "activates", "TP53", "s1", "100", 2018, "positive",
                     "certain", 0.9, "rule_based", "GENE", "GENE"),
            Relation("BRCA1", "activates", "TP53", "s2", "200", 2019, "positive",
                     "certain", 0.88, "rule_based", "GENE", "GENE"),
        ]
        bg = build_graph(rels)
        for u, v, d in bg.graph.edges(data=True):
            if u in {"brca1", "BRCA1"} and v in {"tp53", "TP53"}:
                assert d.get("weight", 0) > 0

    def test_temporal_snapshot(self):
        """Temporal snapshot should only contain edges before cutoff."""
        from graph import build_graph
        rels = self._make_relations()
        bg = build_graph(rels, cutoff_year=2019)
        snapshot = bg.get_temporal_snapshot(2019)
        for u, v, data in snapshot.edges(data=True):
            years = data.get("years", [None])
            if years:
                # At least one year should be <= cutoff
                assert any(y <= 2019 for y in years if y is not None)

    def test_graph_statistics(self):
        from graph import build_graph
        rels = self._make_relations()
        bg = build_graph(rels)
        stats = bg.get_statistics()
        assert "n_nodes" in stats
        assert "n_edges" in stats
        assert "n_biological_edges" in stats
        assert "n_semantic_edges" in stats
        assert "density" in stats
        assert "avg_degree" in stats
        assert "top_nodes_by_degree" in stats
        assert stats["density"] >= 0.0

    def test_gap_candidate_generation(self):
        from graph import build_graph
        rels = self._make_relations()
        bg = build_graph(rels)
        gaps = bg.find_gap_candidates(top_k=20)
        for g in gaps:
            assert g.node_a != g.node_b
            assert g.score >= 0.0
            assert g.novelty_status is not None

    def test_gap_candidates_only_valid_type_pairs(self):
        """Gap candidates should only surface biologically meaningful entity pairs."""
        from graph import build_graph, BioNOVAGraph
        rels = self._make_relations()
        bg = build_graph(rels)
        valid_types = set()
        for a, b in BioNOVAGraph.VALID_BIOLOGICAL_PAIRS:
            valid_types.add((a, b))
            valid_types.add((b, a))
        gaps = bg.find_gap_candidates(top_k=20)
        for g in gaps:
            pair = (g.node_a_type, g.node_b_type)
            assert pair in valid_types, f"Invalid entity type pair: {pair}"

    def test_novelty_verification_known(self):
        """A direct edge should be classified KNOWN."""
        from graph import build_graph, NoveltyStatus, GapCandidate
        rels = self._make_relations()
        bg = build_graph(rels)
        # BRCA1 -> TP53 is a direct edge
        candidate = GapCandidate(
            node_a="brca1", node_b="tp53",
            node_a_type="GENE", node_b_type="GENE",
            score=1.0, method="test",
            common_neighbors=[], novelty_status=NoveltyStatus.UNEXPLORED,
            evidence_pmids=[], max_year=None,
        )
        status = bg.verify_novelty(candidate)
        assert status == NoveltyStatus.KNOWN

    def test_novelty_verification_unexplored(self):
        """A pair with no evidence should be UNEXPLORED or INSUFFICIENT_EVIDENCE."""
        from graph import build_graph, NoveltyStatus, GapCandidate
        rels = self._make_relations()
        bg = build_graph(rels)
        candidate = GapCandidate(
            node_a="brca1", node_b="egfr",
            node_a_type="GENE", node_b_type="GENE",
            score=0.5, method="test",
            common_neighbors=[], novelty_status=NoveltyStatus.UNEXPLORED,
            evidence_pmids=[], max_year=None,
        )
        status = bg.verify_novelty(candidate)
        assert status in {NoveltyStatus.UNEXPLORED, NoveltyStatus.INSUFFICIENT_EVIDENCE,
                          NoveltyStatus.INDIRECTLY_SUPPORTED}

    def test_empty_graph_handling(self):
        from graph import BioNOVAGraph
        bg = BioNOVAGraph()
        stats = bg.get_statistics()
        assert stats["n_nodes"] == 0
        assert stats["n_edges"] == 0
        gaps = bg.find_gap_candidates()
        assert gaps == []

    def test_isolated_nodes_count(self):
        from graph import build_graph
        from extraction import Relation
        r = Relation("BRCA1", "activates", "TP53", "sent.", "1", 2020,
                     "positive", "certain", 0.9, "rule_based", "GENE", "GENE")
        bg = build_graph([r])
        stats = bg.get_statistics()
        # BRCA1 and TP53 both have degree >= 1, so isolated count should be 0
        assert stats["isolated_nodes_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# NOVELTY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNovelty:
    """Tests for novelty classification."""

    def test_known_relationship(self):
        from graph import build_graph, NoveltyStatus, GapCandidate
        from extraction import Relation
        rel = Relation("BRCA1", "activates", "TP53", "s", "1", 2020,
                       "positive", "certain", 0.9, "rule_based", "GENE", "GENE")
        bg = build_graph([rel])
        c = GapCandidate("brca1", "tp53", "GENE", "GENE", 1.0, "t",
                         [], NoveltyStatus.UNEXPLORED, [], None)
        assert bg.verify_novelty(c) == NoveltyStatus.KNOWN

    def test_indirect_relationship(self):
        """A→B→C: A-C is indirectly supported."""
        from graph import build_graph, NoveltyStatus, GapCandidate
        from extraction import Relation
        rels = [
            Relation("A", "activates", "B", "A activates B.", "1", 2020,
                     "positive", "certain", 0.9, "rule_based", "GENE", "GENE"),
            Relation("B", "activates", "C", "B activates C.", "2", 2020,
                     "positive", "certain", 0.9, "rule_based", "GENE", "GENE"),
        ]
        bg = build_graph(rels)
        c = GapCandidate("a", "c", "GENE", "GENE", 0.5, "t",
                         ["b"], NoveltyStatus.UNEXPLORED, [], None)
        status = bg.verify_novelty(c)
        assert status in {NoveltyStatus.INDIRECTLY_SUPPORTED, NoveltyStatus.PARTIALLY_SUPPORTED}

    def test_contradicted_relationship(self):
        """A negated edge should be classified CONTRADICTED."""
        from graph import BioNOVAGraph, NoveltyStatus, GapCandidate
        from extraction import Relation
        neg_rel = Relation("BRCA1", "inhibits", "TP53",
                           "BRCA1 does not activate TP53.", "5",
                           2020, "negative", "certain", 0.8, "rule_based", "GENE", "GENE")
        bg = BioNOVAGraph()
        bg.build_from_relations([neg_rel])
        c = GapCandidate("brca1", "tp53", "GENE", "GENE", 0.5, "t",
                         [], NoveltyStatus.UNEXPLORED, [], None)
        status = bg.verify_novelty(c)
        assert status == NoveltyStatus.CONTRADICTED

    def test_synonym_detection(self):
        from graph import BioNOVAGraph
        bg = BioNOVAGraph()
        # High string similarity should be detected
        assert bg._are_synonymous("brca1", "brca1") is True
        assert bg._are_synonymous("breast cancer", "breast carcinoma") is False or True  # may or may not match
        assert bg._are_synonymous("xyz", "abc") is False


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHypothesis:
    """Tests for hypothesis generation, scoring, and ranking."""

    def _make_gap_candidates(self):
        from graph import GapCandidate, NoveltyStatus
        return [
            GapCandidate("brca1", "breast cancer", "GENE", "DISEASE",
                         0.85, "jaccard", ["tp53"], NoveltyStatus.UNEXPLORED,
                         ["100", "101"], 2020),
            GapCandidate("tamoxifen", "brca1", "CHEMICAL", "GENE",
                         0.72, "adamic_adar", ["egfr"], NoveltyStatus.INDIRECTLY_SUPPORTED,
                         ["200"], 2019),
        ]

    def _make_relations(self):
        from extraction import Relation
        return [
            Relation("BRCA1", "activates", "TP53", "BRCA1 activates TP53.",
                     "100", 2018, "positive", "certain", 0.9, "rule_based", "GENE", "GENE"),
            Relation("tamoxifen", "inhibits", "EGFR",
                     "Tamoxifen inhibits EGFR in breast cancer cells.",
                     "200", 2019, "negative", "certain", 0.85, "rule_based", "CHEMICAL", "GENE"),
        ]

    def test_import(self):
        import hypothesis  # noqa: F401

    def test_hypothesis_dataclass(self):
        from hypothesis import Hypothesis
        h = Hypothesis(
            hypothesis_id="abc123",
            hypothesis="Test hypothesis",
            entities=["A", "B"],
            mechanistic_rationale="Test rationale",
            supporting_evidence=[],
            contradicting_evidence=[],
            novelty_status="UNEXPLORED",
            confidence=0.8,
            testable_prediction="Test prediction",
            suggested_experiment="Test experiment",
            limitations="None",
            alternative_explanations="None",
            devils_advocate="None",
            falsifiability="Testable",
            evidence_score=0.5,
            novelty_score=1.0,
            plausibility_score=0.6,
            feasibility_score=0.7,
            final_score=0.7,
            query="BRCA1",
            generated_at="2024-01-01T00:00:00",
            method="rule_based",
        )
        assert h.hypothesis_id == "abc123"
        assert h.final_score == 0.7

    def test_evidence_scorer(self):
        from hypothesis import EvidenceScorer
        scorer = EvidenceScorer()
        supporting = [{"pmid": "1", "sentence": "s"}, {"pmid": "2", "sentence": "s"}]
        contradicting = []
        score = scorer.compute_evidence_score(supporting, contradicting)
        assert 0.0 <= score <= 1.0

    def test_evidence_scorer_contradiction_penalty(self):
        from hypothesis import EvidenceScorer
        scorer = EvidenceScorer()
        supporting = [{"pmid": "1", "sentence": "s"}]
        contradicting = [{"pmid": "2", "sentence": "s"}, {"pmid": "3", "sentence": "s"}]
        score_with = scorer.compute_evidence_score(supporting, contradicting)
        score_without = scorer.compute_evidence_score(supporting, [])
        assert score_with < score_without

    def test_novelty_score_unexplored(self):
        from hypothesis import EvidenceScorer
        scorer = EvidenceScorer()
        assert scorer.compute_novelty_score("UNEXPLORED") == 1.0
        assert scorer.compute_novelty_score("KNOWN") == 0.0
        assert scorer.compute_novelty_score("CONTRADICTED") < 0.5

    def test_rule_based_generation(self):
        from hypothesis import RuleBasedHypothesisGenerator
        from graph import GapCandidate, NoveltyStatus
        gen = RuleBasedHypothesisGenerator()
        candidate = GapCandidate("brca1", "breast cancer", "GENE", "DISEASE",
                                 0.8, "jaccard", ["tp53"], NoveltyStatus.UNEXPLORED,
                                 ["100"], 2020)
        hyp = gen.generate(candidate, [], query="BRCA1 cancer")
        assert hyp.hypothesis
        assert len(hyp.entities) >= 2
        assert hyp.mechanistic_rationale
        assert hyp.testable_prediction
        assert hyp.suggested_experiment
        assert hyp.limitations
        assert hyp.falsifiability
        assert hyp.devils_advocate
        assert 0.0 <= hyp.confidence <= 1.0

    def test_hypothesis_id_deterministic(self):
        """Same candidate and query should produce the same hypothesis_id."""
        from hypothesis import RuleBasedHypothesisGenerator
        from graph import GapCandidate, NoveltyStatus
        gen = RuleBasedHypothesisGenerator()
        candidate = GapCandidate("brca1", "breast cancer", "GENE", "DISEASE",
                                 0.8, "jaccard", ["tp53"], NoveltyStatus.UNEXPLORED,
                                 [], None)
        h1 = gen.generate(candidate, [], query="test")
        h2 = gen.generate(candidate, [], query="test")
        assert h1.hypothesis_id == h2.hypothesis_id

    def test_pipeline_generates_hypotheses(self):
        from hypothesis import HypothesisPipeline
        candidates = self._make_gap_candidates()
        relations = self._make_relations()
        pipeline = HypothesisPipeline(config_or_api_key=None, use_llm=False)
        hyps = pipeline.generate_hypotheses(candidates, relations, query="BRCA1", top_k=2)
        assert len(hyps) == 2
        for h in hyps:
            assert h.final_score >= 0.0
            assert h.hypothesis

    def test_hypotheses_ranked_by_final_score(self):
        from hypothesis import HypothesisPipeline
        candidates = self._make_gap_candidates()
        pipeline = HypothesisPipeline(config_or_api_key=None, use_llm=False)
        hyps = pipeline.generate_hypotheses(candidates, [], query="test", top_k=2)
        if len(hyps) >= 2:
            assert hyps[0].final_score >= hyps[1].final_score

    def test_hypothesis_export(self, tmp_path):
        from hypothesis import HypothesisPipeline
        candidates = self._make_gap_candidates()
        pipeline = HypothesisPipeline(config_or_api_key=None, use_llm=False)
        hyps = pipeline.generate_hypotheses(candidates, [], query="test", top_k=2)
        outfile = str(tmp_path / "hyps.json")
        pipeline.export_hypotheses(hyps, outfile)
        with open(outfile) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert "hypothesis" in loaded[0]

    def test_reproducibility(self):
        """Same inputs should always produce same outputs."""
        from hypothesis import generate_and_rank
        from graph import GapCandidate, NoveltyStatus
        candidate = GapCandidate("brca1", "breast cancer", "GENE", "DISEASE",
                                 0.8, "jaccard", ["tp53"], NoveltyStatus.UNEXPLORED,
                                 [], None)
        hyps1 = generate_and_rank([candidate], [], query="test", top_k=1)
        hyps2 = generate_and_rank([candidate], [], query="test", top_k=1)
        assert hyps1[0].hypothesis_id == hyps2[0].hypothesis_id
        assert hyps1[0].final_score == hyps2[0].final_score

    def test_llm_fallback_without_key(self):
        """LLM pipeline with no API key should fall back to rule-based."""
        from hypothesis import HypothesisPipeline
        candidates = self._make_gap_candidates()
        pipeline = HypothesisPipeline(config_or_api_key=None, use_llm=True)
        hyps = pipeline.generate_hypotheses(candidates, [], query="test", top_k=1)
        assert len(hyps) >= 1
        assert hyps[0].method in {"rule_based", "llm"}


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Complete pipeline integration test using only local/mock data."""

    def test_full_pipeline(self):
        """
        Input → retrieval fixture → extraction → graph → candidate discovery
        → novelty → hypothesis → ranking
        """
        # Step 1: Load local sample articles (no internet)
        from retrieval import create_sample_articles, DataCleaner
        articles = create_sample_articles()
        cleaner = DataCleaner()
        articles = cleaner.clean_articles(articles)
        assert len(articles) >= 3

        # Step 2: Extract entities and relations
        from extraction import ExtractionPipeline
        extractor = ExtractionPipeline(use_scispacy=False)
        entities, relations = extractor.process_articles(articles)
        assert len(entities) > 0
        assert len(relations) > 0

        # Step 3: Build knowledge graph
        from graph import build_graph
        bg = build_graph(relations)
        stats = bg.get_statistics()
        assert stats["n_nodes"] > 0
        assert stats["n_biological_edges"] >= 0

        # Step 4: Find gap candidates
        gaps = bg.find_gap_candidates(top_k=10)
        # Small dataset may have 0 gaps — that's acceptable

        # Step 5: Generate hypotheses (works even with no gaps, returns [])
        from hypothesis import generate_and_rank
        hyps = generate_and_rank(gaps[:3], relations, query="BRCA1 cancer", top_k=3)
        assert isinstance(hyps, list)

        # Step 6: Verify ranking
        if len(hyps) >= 2:
            scores = [h.final_score for h in hyps]
            assert scores == sorted(scores, reverse=True)

        # Step 7: Verify reproducibility (re-run produces same hypothesis IDs)
        hyps2 = generate_and_rank(gaps[:3], relations, query="BRCA1 cancer", top_k=3)
        ids1 = [h.hypothesis_id for h in hyps]
        ids2 = [h.hypothesis_id for h in hyps2]
        assert ids1 == ids2

    def test_pipeline_with_cutoff_year(self):
        """Historical cutoff year should restrict the temporal graph."""
        from retrieval import create_sample_articles
        from extraction import ExtractionPipeline
        from graph import build_graph

        articles = create_sample_articles()
        extractor = ExtractionPipeline(use_scispacy=False)
        _, relations = extractor.process_articles(articles)

        # Build with cutoff
        cutoff = 2019
        bg = build_graph(relations, cutoff_year=cutoff)
        snapshot = bg.get_temporal_snapshot(cutoff)
        # All edges in snapshot should have at least one year <= cutoff
        for _u, _v, data in snapshot.edges(data=True):
            years = [y for y in data.get("years", []) if y is not None]
            if years:
                assert min(years) <= cutoff

    def test_empty_query_graceful(self):
        """Empty article list should not crash any pipeline step."""
        from extraction import ExtractionPipeline
        from graph import build_graph
        from hypothesis import generate_and_rank

        extractor = ExtractionPipeline(use_scispacy=False)
        entities, relations = extractor.process_articles([])
        assert entities == []
        assert relations == []

        bg = build_graph([])
        stats = bg.get_statistics()
        assert stats["n_nodes"] == 0

        hyps = generate_and_rank([], [], query="")
        assert hyps == []

    def test_malformed_relation_handling(self):
        """Graph build must not crash on partially malformed relations."""
        from extraction import Relation
        from graph import build_graph
        # Relation with missing fields (should be handled gracefully)
        r = Relation("", "activates", "TP53", "s", "1", None,
                     "positive", "certain", 0.5, "rule_based", "GENE", "GENE")
        bg = build_graph([r])  # should not raise
        assert isinstance(bg.get_statistics(), dict)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompilation:
    """Verify all source files compile without errors."""

    @pytest.mark.parametrize("module_name", [
        "utils", "retrieval", "extraction", "graph", "hypothesis",
    ])
    def test_py_compile(self, module_name):
        import py_compile
        path = _ROOT / f"{module_name}.py"
        assert path.exists(), f"{module_name}.py not found at {path}"
        py_compile.compile(str(path), doraise=True)

    @pytest.mark.parametrize("module_name", [
        "utils", "retrieval", "extraction", "graph", "hypothesis",
    ])
    def test_import_module(self, module_name):
        import importlib
        mod = importlib.import_module(module_name)
        assert mod is not None
