"""Lightweight graph construction and gap discovery utilities."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import networkx as nx

try:  # pragma: no cover - optional dependency
    import numpy as np  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    np = None

if TYPE_CHECKING:
    from extraction import Relation

logger = logging.getLogger(__name__)


class EdgeType(Enum):
    BIOLOGICAL = "biological"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"


class NoveltyStatus(Enum):
    KNOWN = "known"
    PARTIALLY_SUPPORTED = "partially_supported"
    INDIRECTLY_SUPPORTED = "indirectly_supported"
    UNEXPLORED = "unexplored"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class KGEdge:
    subject: str
    predicate: str
    object: str
    edge_type: EdgeType
    pmids: List[str]
    years: List[int]
    polarity: str
    confidence: float
    weight: float
    is_negated: bool


@dataclass
class GapCandidate:
    node_a: str
    node_b: str
    node_a_type: str
    node_b_type: str
    score: float
    method: str
    common_neighbors: List[str]
    novelty_status: NoveltyStatus
    evidence_pmids: List[str]
    max_year: Optional[int]


class BioNOVAGraph:
    BIOLOGICAL_PREDICATES = {
        "activates", "stimulates", "induces", "promotes", "upregulates",
        "increases", "enhances", "inhibits", "suppresses", "blocks",
        "downregulates", "decreases", "reduces", "attenuates", "targets",
        "binds", "interacts_with", "associated_with", "correlates_with",
        "requires",
    }
    SEMANTIC_PREDICATES = {"co_occurs_with", "semantic_similarity", "related_to"}
    VALID_BIOLOGICAL_PAIRS = {
        ("GENE", "DISEASE"),
        ("GENE", "GENE"),
        ("CHEMICAL", "GENE"),
        ("CHEMICAL", "DISEASE"),
        ("PROTEIN", "PATHWAY"),
        ("GENE", "PATHWAY"),
        ("PROTEIN", "PROTEIN"),
    }

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.semantic_graph: nx.DiGraph = nx.DiGraph()
        self.temporal_graph: nx.DiGraph = nx.DiGraph()
        self._negative_edges: Dict[Tuple[str, str], Dict[str, object]] = {}

    def add_relations(self, relations: List["Relation"]) -> None:
        for relation in relations or []:
            subject = self._node_key(relation.subject)
            obj = self._node_key(relation.object)
            if not subject or not obj or subject == obj:
                continue

            subject_type = getattr(relation, "subject_type", "OTHER") or "OTHER"
            object_type = getattr(relation, "object_type", "OTHER") or "OTHER"
            predicate = self._normalize_predicate(getattr(relation, "predicate", ""))
            year = getattr(relation, "publication_year", None)
            pmid = str(getattr(relation, "pmid", "") or "")
            confidence = float(getattr(relation, "confidence", 0.0) or 0.0)
            polarity = getattr(relation, "polarity", "positive") or "positive"

            self._ensure_node(self.graph, subject, relation.subject, subject_type)
            self._ensure_node(self.graph, obj, relation.object, object_type)
            self._ensure_node(self.semantic_graph, subject, relation.subject, subject_type)
            self._ensure_node(self.semantic_graph, obj, relation.object, object_type)

            if polarity == "negative":
                self._record_negative_edge(subject, obj, pmid, year, confidence, predicate)
                continue

            edge_type = (
                EdgeType.BIOLOGICAL
                if self._is_biological_predicate(predicate)
                else EdgeType.SEMANTIC
            )
            target_graph = self.graph if edge_type == EdgeType.BIOLOGICAL else self.semantic_graph
            self._add_or_merge_edge(
                target_graph=target_graph,
                subject=subject,
                obj=obj,
                label_subject=relation.subject,
                label_object=relation.object,
                subject_type=subject_type,
                object_type=object_type,
                predicate=predicate,
                pmid=pmid,
                year=year,
                confidence=confidence,
                edge_type=edge_type,
                is_negated=False,
            )
        self.compute_edge_weights()

    def _is_biological_predicate(self, predicate: str) -> bool:
        return self._normalize_predicate(predicate) in self.BIOLOGICAL_PREDICATES

    def add_temporal_edges(
        self, relations: List["Relation"], cutoff_year: Optional[int]
    ) -> None:
        self.temporal_graph = nx.DiGraph()
        for relation in relations or []:
            if getattr(relation, "polarity", "positive") == "negative":
                continue
            year = getattr(relation, "publication_year", None)
            if cutoff_year is not None and (year is None or year > cutoff_year):
                continue
            predicate = self._normalize_predicate(getattr(relation, "predicate", ""))
            if not self._is_biological_predicate(predicate):
                continue
            subject = self._node_key(relation.subject)
            obj = self._node_key(relation.object)
            if not subject or not obj or subject == obj:
                continue
            self._add_or_merge_edge(
                target_graph=self.temporal_graph,
                subject=subject,
                obj=obj,
                label_subject=relation.subject,
                label_object=relation.object,
                subject_type=getattr(relation, "subject_type", "OTHER") or "OTHER",
                object_type=getattr(relation, "object_type", "OTHER") or "OTHER",
                predicate=predicate,
                pmid=str(getattr(relation, "pmid", "") or ""),
                year=year,
                confidence=float(getattr(relation, "confidence", 0.0) or 0.0),
                edge_type=EdgeType.TEMPORAL,
                is_negated=False,
            )
        self.compute_edge_weights()

    def build_from_relations(
        self, relations: List["Relation"], cutoff_year: Optional[int] = None
    ) -> None:
        self.graph.clear()
        self.semantic_graph.clear()
        self.temporal_graph.clear()
        self._negative_edges.clear()
        self.add_relations(relations)
        self.add_temporal_edges(relations, cutoff_year)

    def get_statistics(self) -> dict:
        combined = nx.compose(self.graph, self.semantic_graph)
        n_nodes = combined.number_of_nodes()
        n_bio = self.graph.number_of_edges()
        n_sem = self.semantic_graph.number_of_edges()
        degree_graph = combined.to_undirected()
        degrees = dict(degree_graph.degree())
        avg_degree = sum(degrees.values()) / n_nodes if n_nodes else 0.0
        top_nodes = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:10]
        isolated = [node for node, degree in degrees.items() if degree == 0]
        components = (
            nx.number_connected_components(degree_graph) if n_nodes else 0
        )
        density = nx.density(degree_graph) if n_nodes > 1 else 0.0
        return {
            "n_nodes": n_nodes,
            "n_edges": n_bio + n_sem,
            "n_biological_edges": n_bio,
            "n_semantic_edges": n_sem,
            "density": density,
            "avg_degree": avg_degree,
            "top_nodes_by_degree": top_nodes,
            "isolated_nodes_count": len(isolated),
            "connected_components": components,
            "negative_edge_count": len(self._negative_edges),
            "summary_json": json.dumps(
                {
                    "n_nodes": n_nodes,
                    "n_biological_edges": n_bio,
                    "n_semantic_edges": n_sem,
                },
                sort_keys=True,
            ),
        }

    def find_gap_candidates(
        self,
        cutoff_year: Optional[int] = None,
        top_k: int = 20,
        entity_type_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[GapCandidate]:
        source = self.get_temporal_snapshot(cutoff_year) if cutoff_year is not None else self.graph
        undirected = source.to_undirected()
        if undirected.number_of_nodes() < 2:
            return []

        allowed_pairs = self._expand_pairs(entity_type_pairs or list(self.VALID_BIOLOGICAL_PAIRS))
        candidates: List[GapCandidate] = []
        for node_a, node_b in combinations(sorted(undirected.nodes()), 2):
            if undirected.has_edge(node_a, node_b):
                continue
            type_a = source.nodes[node_a].get("entity_type", "OTHER")
            type_b = source.nodes[node_b].get("entity_type", "OTHER")
            if (type_a, type_b) not in allowed_pairs:
                continue

            neighbors_a = set(undirected.neighbors(node_a))
            neighbors_b = set(undirected.neighbors(node_b))
            common = sorted(neighbors_a & neighbors_b)
            if not common:
                continue

            union_size = len(neighbors_a | neighbors_b)
            jaccard = len(common) / union_size if union_size else 0.0
            adamic_adar = 0.0
            resource_allocation = 0.0
            for neighbor in common:
                degree = max(undirected.degree(neighbor), 1)
                if degree > 1:
                    adamic_adar += 1.0 / math.log(degree)
                resource_allocation += 1.0 / degree
            common_neighbor_score = float(len(common))
            composite = (
                0.35 * jaccard
                + 0.25 * adamic_adar
                + 0.2 * resource_allocation
                + 0.2 * common_neighbor_score
            )
            evidence_pmids, max_year = self._collect_neighbor_evidence(source, common)
            method_scores = {
                "jaccard": jaccard,
                "adamic_adar": adamic_adar,
                "resource_allocation": resource_allocation,
                "common_neighbors": common_neighbor_score,
            }
            method = max(method_scores.items(), key=lambda item: item[1])[0]
            candidate = GapCandidate(
                node_a=node_a,
                node_b=node_b,
                node_a_type=type_a,
                node_b_type=type_b,
                score=round(composite, 6),
                method=method,
                common_neighbors=common,
                novelty_status=NoveltyStatus.UNEXPLORED,
                evidence_pmids=evidence_pmids,
                max_year=max_year,
            )
            candidate.novelty_status = self.verify_novelty(candidate, cutoff_year=cutoff_year)
            candidates.append(candidate)

        candidates.sort(key=lambda item: (-item.score, item.node_a, item.node_b))
        return candidates[: max(top_k, 0)]

    def verify_novelty(
        self, candidate: GapCandidate, cutoff_year: Optional[int] = None
    ) -> NoveltyStatus:
        graph = self.get_temporal_snapshot(cutoff_year) if cutoff_year is not None else self.graph
        a = candidate.node_a
        b = candidate.node_b
        if graph.has_edge(a, b) or graph.has_edge(b, a):
            return NoveltyStatus.KNOWN
        if self._has_negative_edge(a, b, cutoff_year):
            return NoveltyStatus.CONTRADICTED
        if self._path_exists_with_length(graph, a, b, 2):
            return NoveltyStatus.PARTIALLY_SUPPORTED
        if self._path_exists_with_length(graph, a, b, 3):
            return NoveltyStatus.INDIRECTLY_SUPPORTED
        if self._are_synonymous(a, b):
            return NoveltyStatus.PARTIALLY_SUPPORTED
        if len(candidate.evidence_pmids) < 2 and len(candidate.common_neighbors) < 2:
            return NoveltyStatus.INSUFFICIENT_EVIDENCE
        return NoveltyStatus.UNEXPLORED

    def _are_synonymous(self, a: str, b: str) -> bool:
        cleaned_a = re.sub(r"[^a-z0-9]+", "", a.lower())
        cleaned_b = re.sub(r"[^a-z0-9]+", "", b.lower())
        if not cleaned_a or not cleaned_b:
            return False
        return SequenceMatcher(None, cleaned_a, cleaned_b).ratio() >= 0.8

    def get_temporal_snapshot(self, cutoff_year: int) -> nx.DiGraph:
        snapshot = nx.DiGraph()
        for node, data in self.graph.nodes(data=True):
            snapshot.add_node(node, **dict(data))
        for subject, obj, data in self.graph.edges(data=True):
            years = [int(year) for year in data.get("years", []) if isinstance(year, int)]
            if not years or max(years) > cutoff_year:
                continue
            snapshot.add_edge(subject, obj, **dict(data))
        return snapshot

    def remove_duplicate_edges(self) -> int:
        removed = 0
        for target_graph in (self.graph, self.semantic_graph, self.temporal_graph):
            for _subject, _object, data in target_graph.edges(data=True):
                pmids = data.get("pmids", [])
                years = data.get("years", [])
                predicates = data.get("predicates", [])
                dedup_pmids = list(dict.fromkeys(pmids))
                dedup_years = sorted(set(years))
                dedup_predicates = list(dict.fromkeys(predicates))
                removed += max(0, len(pmids) - len(dedup_pmids))
                removed += max(0, len(years) - len(dedup_years))
                removed += max(0, len(predicates) - len(dedup_predicates))
                data["pmids"] = dedup_pmids
                data["years"] = dedup_years
                data["predicates"] = dedup_predicates
                if dedup_predicates:
                    counts = Counter(predicates or dedup_predicates)
                    data["predicate"] = counts.most_common(1)[0][0]
        self.compute_edge_weights()
        return removed

    def compute_edge_weights(self) -> None:
        for target_graph in (self.graph, self.semantic_graph, self.temporal_graph):
            for _subject, _object, data in target_graph.edges(data=True):
                evidence_count = len(set(data.get("pmids", [])))
                avg_confidence = float(data.get("confidence", 0.0) or 0.0)
                data["weight"] = round(math.log(1 + evidence_count) * avg_confidence, 6)

    @staticmethod
    def _node_key(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        return re.sub(r"\s+", "_", (predicate or "").strip().lower())

    @staticmethod
    def _ensure_node(graph: nx.DiGraph, node: str, label: str, entity_type: str) -> None:
        if graph.has_node(node):
            existing_type = graph.nodes[node].get("entity_type")
            if existing_type in (None, "OTHER") and entity_type:
                graph.nodes[node]["entity_type"] = entity_type
            if label and not graph.nodes[node].get("label"):
                graph.nodes[node]["label"] = label
            return
        graph.add_node(node, label=label, entity_type=entity_type or "OTHER")

    def _add_or_merge_edge(
        self,
        target_graph: nx.DiGraph,
        subject: str,
        obj: str,
        label_subject: str,
        label_object: str,
        subject_type: str,
        object_type: str,
        predicate: str,
        pmid: str,
        year: Optional[int],
        confidence: float,
        edge_type: EdgeType,
        is_negated: bool,
    ) -> None:
        self._ensure_node(target_graph, subject, label_subject, subject_type)
        self._ensure_node(target_graph, obj, label_object, object_type)
        year_list = [int(year)] if isinstance(year, int) else []
        if target_graph.has_edge(subject, obj):
            data = target_graph[subject][obj]
            data["pmids"] = list(dict.fromkeys(data.get("pmids", []) + ([pmid] if pmid else [])))
            data["years"] = sorted(set(data.get("years", []) + year_list))
            data["predicates"] = list(
                dict.fromkeys(data.get("predicates", []) + ([predicate] if predicate else []))
            )
            data["predicate_counts"][predicate] += 1
            data["predicate"] = data["predicate_counts"].most_common(1)[0][0]
            data["confidence_values"].append(confidence)
            data["confidence"] = round(
                sum(data["confidence_values"]) / len(data["confidence_values"]), 6
            )
            data["edge_type"] = edge_type
            data["polarity"] = "negative" if is_negated else data.get("polarity", "positive")
            data["is_negated"] = data.get("is_negated", False) or is_negated
            return

        target_graph.add_edge(
            subject,
            obj,
            predicate=predicate,
            predicates=[predicate] if predicate else [],
            predicate_counts=Counter([predicate]) if predicate else Counter(),
            pmids=[pmid] if pmid else [],
            years=year_list,
            polarity="negative" if is_negated else "positive",
            confidence=round(confidence, 6),
            confidence_values=[confidence],
            weight=0.0,
            edge_type=edge_type,
            is_negated=is_negated,
        )

    def _record_negative_edge(
        self,
        subject: str,
        obj: str,
        pmid: str,
        year: Optional[int],
        confidence: float,
        predicate: str,
    ) -> None:
        key = (subject, obj)
        current = self._negative_edges.get(key)
        if current is None:
            self._negative_edges[key] = {
                "pmids": [pmid] if pmid else [],
                "years": [int(year)] if isinstance(year, int) else [],
                "confidence_values": [confidence],
                "predicate": predicate,
            }
            return
        if pmid:
            current["pmids"] = list(dict.fromkeys(list(current["pmids"]) + [pmid]))
        if isinstance(year, int):
            current["years"] = sorted(set(list(current["years"]) + [year]))
        current["confidence_values"] = list(current["confidence_values"]) + [confidence]

    def _has_negative_edge(
        self, node_a: str, node_b: str, cutoff_year: Optional[int]
    ) -> bool:
        for key in ((node_a, node_b), (node_b, node_a)):
            record = self._negative_edges.get(key)
            if not record:
                continue
            years = [year for year in record.get("years", []) if isinstance(year, int)]
            if cutoff_year is None:
                return True
            if years and min(years) <= cutoff_year:
                return True
        return False

    @staticmethod
    def _path_exists_with_length(
        graph: nx.DiGraph, source: str, target: str, edge_length: int
    ) -> bool:
        if not graph.has_node(source) or not graph.has_node(target):
            return False
        undirected = graph.to_undirected()
        if not undirected.has_node(source) or not undirected.has_node(target):
            return False
        try:
            distance = nx.shortest_path_length(undirected, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
        return distance == edge_length

    @staticmethod
    def _expand_pairs(pairs: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        expanded: Set[Tuple[str, str]] = set()
        for left, right in pairs:
            expanded.add((left, right))
            expanded.add((right, left))
        return expanded

    @staticmethod
    def _collect_neighbor_evidence(
        graph: nx.DiGraph, neighbors: List[str]
    ) -> Tuple[List[str], Optional[int]]:
        pmids: List[str] = []
        years: List[int] = []
        for neighbor in neighbors:
            for _src, _dst, data in graph.in_edges(neighbor, data=True):
                pmids.extend(data.get("pmids", []))
                years.extend([year for year in data.get("years", []) if isinstance(year, int)])
            for _src, _dst, data in graph.out_edges(neighbor, data=True):
                pmids.extend(data.get("pmids", []))
                years.extend([year for year in data.get("years", []) if isinstance(year, int)])
        dedup_pmids = list(dict.fromkeys(pmids))
        return dedup_pmids, (max(years) if years else None)


def build_graph(
    relations: List["Relation"], cutoff_year: Optional[int] = None
) -> BioNOVAGraph:
    graph = BioNOVAGraph()
    graph.build_from_relations(relations, cutoff_year=cutoff_year)
    return graph
