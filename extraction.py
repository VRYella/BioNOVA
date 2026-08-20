"""Lightweight biomedical entity and relation extraction."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    try:
        from retrieval import Article  # type: ignore
    except ImportError:
        Article = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    sentence: str
    pmid: str
    publication_year: Optional[int]
    polarity: str
    certainty: str
    confidence: float
    source: str
    subject_type: str
    object_type: str


@dataclass(frozen=True)
class Entity:
    text: str
    normalized: str
    entity_type: str
    pmid: str
    sentence: str
    confidence: float


class RuleBasedExtractor:
    """Deterministic regex-based extractor that avoids heavy NLP dependencies."""

    GENE_TYPE = "GENE"
    DISEASE_TYPE = "DISEASE"
    CHEMICAL_TYPE = "CHEMICAL"
    PROTEIN_TYPE = "PROTEIN"
    PATHWAY_TYPE = "PATHWAY"
    PROCESS_TYPE = "BIOLOGICAL_PROCESS"
    OTHER_TYPE = "OTHER"

    GENE_PATTERN = re.compile(
        r"\b([A-Z][A-Z0-9]{1,9}[0-9]|[A-Z]{2,}[0-9]+[A-Z]*)\b"
    )
    DISEASE_PATTERNS = [
        re.compile(
            r"\b(cancer|tumor|tumour|carcinoma|sarcoma|lymphoma|leukemia|leukaemia|melanoma|glioma|neuroblastoma)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\w+\s+cancer\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+disease\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+syndrome\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+disorder\b", re.IGNORECASE),
    ]
    CHEMICAL_PATTERNS = [
        re.compile(r"\b[a-z]+mab\b"),
        re.compile(r"\b[a-z]+nib\b"),
        re.compile(r"\b[a-z]+zumab\b"),
        re.compile(r"\b[a-z]+tinib\b"),
        re.compile(
            r"\b(tamoxifen|cisplatin|doxorubicin|methotrexate)\b", re.IGNORECASE
        ),
    ]
    PATHWAY_PATTERNS = [
        re.compile(r"\b(PI3K|MAPK|mTOR|NF-κB|Wnt|Notch|Hedgehog|JAK|STAT|p53|Ras|ERK)\b"),
        re.compile(r"\b\w+\s+pathway\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+signaling\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+cascade\b", re.IGNORECASE),
    ]
    PROCESS_PATTERN = re.compile(
        r"\b(apoptosis|autophagy|necrosis|ferroptosis|pyroptosis|proliferation|angiogenesis|metastasis|inflammation|differentiation)\b",
        re.IGNORECASE,
    )
    PROTEIN_PATTERN = re.compile(
        r"\b[\w-]+\s+(protein|kinase|receptor|ligand|enzyme|transporter)\b",
        re.IGNORECASE,
    )

    ACTIVE_PATTERN = re.compile(
        r"\b(activates?|activated|stimulates?|stimulated|induces?|induced|"
        r"promotes?|promoted|upregulates?|upregulated|increases?|increased|"
        r"enhances?|enhanced|activating|inducing|promoting)\b",
        re.IGNORECASE,
    )
    PASSIVE_ACTIVE_PATTERN = re.compile(
        r"\b(is|are|was|were)\s+(activated|stimulated|induced|promoted|upregulated)\s+by\b",
        re.IGNORECASE,
    )
    INHIBIT_PATTERN = re.compile(
        r"\b(inhibits?|inhibited|suppresses?|suppressed|blocks?|blocked|"
        r"downregulates?|downregulated|decreases?|decreased|reduces?|reduced|"
        r"attenuates?|attenuated|inhibiting|suppressing)\b",
        re.IGNORECASE,
    )
    PASSIVE_INHIBIT_PATTERN = re.compile(
        r"\b(is|are|was|were)\s+(inhibited|suppressed|blocked|downregulated|reduced)\s+by\b",
        re.IGNORECASE,
    )
    # "associated with", "linked to", "related to", "linked BRCA1 to TP53"
    ASSOCIATION_PATTERN = re.compile(
        r"\b(associated\s+with|linked\s+to|linked\s+\w+\s+to|related\s+to|"
        r"associated\s+\w+\s+with|connection\s+between|relationship\s+between)\b",
        re.IGNORECASE,
    )
    CORRELATION_PATTERN = re.compile(
        r"\b(correlates?\s+with|correlation\s+with|correlating\s+with|"
        r"expression\s+correlates|mutation\s+correlates|activity\s+correlates)\b",
        re.IGNORECASE,
    )
    BINDING_PATTERN = re.compile(
        r"\b(binds?\s+to|binds?\s+\w+|interacts?\s+with|interaction\s+with|"
        r"binding\s+to|interacting\s+with)\b",
        re.IGNORECASE,
    )
    TARGET_PATTERN = re.compile(r"\btargets?\b", re.IGNORECASE)
    # "X required Y", "X required to do Y" — common in biomedical literature
    REQUIRE_PATTERN = re.compile(
        r"\b(requires?|required|requiring|depends?\s+on|dependent\s+on)\b",
        re.IGNORECASE,
    )

    UPPERCASE_STOPWORDS = {
        "AND",
        "ARE",
        "AS",
        "AT",
        "BY",
        "DNA",
        "FOR",
        "FROM",
        "HAS",
        "HAVE",
        "IL",
        "IN",
        "IS",
        "IT",
        "NOT",
        "OF",
        "ON",
        "OR",
        "RNA",
        "THE",
        "TO",
        "USA",
        "WAS",
        "WHO",
        "WITH",
    }
    KNOWN_GENES = {
        "AKT1",
        "ALK",
        "APC",
        "BRAF",
        "BRCA1",
        "BRCA2",
        "CDK4",
        "EGFR",
        "ERBB2",
        "HER2",
        "JAK2",
        "KRAS",
        "MAPK1",
        "MTOR",
        "MYC",
        "NFKB1",
        "NRAS",
        "PIK3CA",
        "PTEN",
        "STAT3",
        "TP53",
        "VEGFA",
    }
    NEGATION_TERMS = (
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "lack",
        "absent",
        "failed to",
        "did not",
        "does not",
        "cannot",
        "unable to",
    )
    PROBABLE_TERMS = ("likely", "suggest", "indicate", "appear", "seem")
    POSSIBLE_TERMS = ("may", "might", "could", "possibly", "potentially")
    SPECULATIVE_TERMS = ("hypothesize", "speculate", "speculative", "perhaps")
    CONDITIONAL_TERMS = ("if", "whether", "dependent on", "depending on", "when")

    def extract_entities(
        self, text: str, pmid: str = "", year: Optional[int] = None
    ) -> List[Entity]:
        del year
        if not text:
            return []
        entities: List[Entity] = []
        seen = set()
        for sentence in self._segment_sentences(text):
            for entity, _span in self._extract_entity_spans(sentence, pmid):
                key = (entity.normalized, entity.entity_type, entity.pmid, entity.sentence)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(entity)
        return entities

    def extract_relations(
        self, text: str, pmid: str = "", year: Optional[int] = None
    ) -> List[Relation]:
        if not text:
            return []

        relations: List[Relation] = []
        seen = set()
        for sentence in self._segment_sentences(text):
            sentence_entities = self._extract_entity_spans(sentence, pmid)
            if len(sentence_entities) < 2:
                continue
            certainty = self._detect_certainty(sentence)
            is_conditional = self._is_conditional(sentence)
            relation_specs = [
                (self.PASSIVE_ACTIVE_PATTERN, "activates", "positive", "passive"),
                (self.PASSIVE_INHIBIT_PATTERN, "inhibits", "negative", "passive"),
                (self.ACTIVE_PATTERN, None, "positive", "active"),
                (self.INHIBIT_PATTERN, None, "negative", "active"),
                (self.ASSOCIATION_PATTERN, "associated_with", "positive", "active"),
                (self.CORRELATION_PATTERN, "correlates_with", "positive", "active"),
                (self.BINDING_PATTERN, None, "positive", "active"),
                (self.TARGET_PATTERN, "targets", "positive", "active"),
                (self.REQUIRE_PATTERN, "requires", "positive", "active"),
            ]
            for pattern, canonical_predicate, base_polarity, direction in relation_specs:
                for match in pattern.finditer(sentence):
                    relation = self._build_relation_from_match(
                        sentence=sentence,
                        entities=sentence_entities,
                        match=match,
                        pmid=pmid,
                        year=year,
                        canonical_predicate=canonical_predicate,
                        base_polarity=base_polarity,
                        certainty=certainty,
                        is_conditional=is_conditional,
                        direction=direction,
                    )
                    if relation is None:
                        continue
                    key = (
                        relation.subject.lower(),
                        relation.predicate,
                        relation.object.lower(),
                        relation.sentence,
                        relation.pmid,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(relation)
        return relations

    def _segment_sentences(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9])", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _detect_negation(self, sentence: str, span_start: int) -> bool:
        window_start = max(0, span_start - 40)
        context = sentence[window_start:span_start].lower()
        return any(term in context for term in self.NEGATION_TERMS)

    def _detect_certainty(self, sentence: str) -> str:
        lower = sentence.lower()
        if any(term in lower for term in self.SPECULATIVE_TERMS):
            return "speculative"
        if any(term in lower for term in self.POSSIBLE_TERMS):
            return "possible"
        if any(term in lower for term in self.PROBABLE_TERMS):
            return "probable"
        return "certain"

    def _classify_entity_type(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return self.OTHER_TYPE
        if any(pattern.search(cleaned) for pattern in self.PATHWAY_PATTERNS):
            return self.PATHWAY_TYPE
        if any(pattern.search(cleaned) for pattern in self.DISEASE_PATTERNS):
            return self.DISEASE_TYPE
        if any(pattern.search(cleaned) for pattern in self.CHEMICAL_PATTERNS):
            return self.CHEMICAL_TYPE
        if self.PROCESS_PATTERN.search(cleaned):
            return self.PROCESS_TYPE
        if self.PROTEIN_PATTERN.search(cleaned):
            return self.PROTEIN_TYPE
        upper = cleaned.upper()
        if upper in self.KNOWN_GENES or (
            self.GENE_PATTERN.fullmatch(cleaned)
            and upper not in self.UPPERCASE_STOPWORDS
        ):
            return self.GENE_TYPE
        return self.OTHER_TYPE

    def _normalize_entity(self, text: str) -> str:
        cleaned = unicodedata.normalize("NFKC", text or "")
        cleaned = cleaned.strip().lower()
        cleaned = re.sub(r"[^\w\s\-κ]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned.endswith("s") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        return cleaned

    def _extract_entity_spans(
        self, sentence: str, pmid: str
    ) -> List[Tuple[Entity, Tuple[int, int]]]:
        spans: List[Tuple[int, int, str, float]] = []
        patterns: Iterable[Tuple[re.Pattern[str], str, float]] = [
            (self.GENE_PATTERN, self.GENE_TYPE, 0.82),
            *[(pattern, self.DISEASE_TYPE, 0.86) for pattern in self.DISEASE_PATTERNS],
            *[(pattern, self.CHEMICAL_TYPE, 0.85) for pattern in self.CHEMICAL_PATTERNS],
            *[(pattern, self.PATHWAY_TYPE, 0.84) for pattern in self.PATHWAY_PATTERNS],
            (self.PROCESS_PATTERN, self.PROCESS_TYPE, 0.78),
            (self.PROTEIN_PATTERN, self.PROTEIN_TYPE, 0.8),
        ]
        for pattern, fallback_type, confidence in patterns:
            for match in pattern.finditer(sentence):
                text = match.group(0).strip(" ,.;:()[]")
                if not text:
                    continue
                if fallback_type == self.GENE_TYPE and text.upper() in self.UPPERCASE_STOPWORDS:
                    continue
                spans.append((match.start(), match.end(), text, confidence))

        spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        selected: List[Tuple[int, int, str, float]] = []
        for start, end, text, confidence in spans:
            if any(not (end <= s or start >= e) for s, e, _, _ in selected):
                continue
            selected.append((start, end, text, confidence))

        entities: List[Tuple[Entity, Tuple[int, int]]] = []
        seen = set()
        for start, end, text, confidence in selected:
            entity_type = self._classify_entity_type(text)
            if entity_type == self.OTHER_TYPE and len(text) < 4:
                continue
            normalized = self._normalize_entity(text)
            key = (normalized, entity_type, start, end)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                (
                    Entity(
                        text=text,
                        normalized=normalized,
                        entity_type=entity_type,
                        pmid=pmid,
                        sentence=sentence,
                        confidence=round(confidence, 3),
                    ),
                    (start, end),
                )
            )
        return entities

    def _build_relation_from_match(
        self,
        sentence: str,
        entities: List[Tuple[Entity, Tuple[int, int]]],
        match: re.Match[str],
        pmid: str,
        year: Optional[int],
        canonical_predicate: Optional[str],
        base_polarity: str,
        certainty: str,
        is_conditional: bool,
        direction: str,
    ) -> Optional[Relation]:
        if direction == "passive":
            obj_entity = self._nearest_entity_before(entities, match.start())
            subj_entity = self._nearest_entity_after(entities, match.end())
        else:
            subj_entity = self._nearest_entity_before(entities, match.start())
            obj_entity = self._nearest_entity_after(entities, match.end())

        if subj_entity is None or obj_entity is None:
            return None
        if subj_entity.normalized == obj_entity.normalized:
            return None

        predicate = canonical_predicate or self._normalize_predicate(match.group(1))
        polarity = self._resolve_polarity(
            sentence=sentence,
            span_start=match.start(),
            base_polarity=base_polarity,
            certainty=certainty,
            is_conditional=is_conditional,
        )
        confidence = self._score_relation_confidence(predicate, certainty, polarity)
        return Relation(
            subject=subj_entity.text,
            predicate=predicate,
            object=obj_entity.text,
            sentence=sentence,
            pmid=pmid,
            publication_year=year,
            polarity=polarity,
            certainty=certainty,
            confidence=confidence,
            source="rule_based",
            subject_type=subj_entity.entity_type,
            object_type=obj_entity.entity_type,
        )

    @staticmethod
    def _nearest_entity_before(
        entities: List[Tuple[Entity, Tuple[int, int]]], position: int
    ) -> Optional[Entity]:
        candidates = [entity for entity, span in entities if span[1] <= position]
        return candidates[-1] if candidates else None

    @staticmethod
    def _nearest_entity_after(
        entities: List[Tuple[Entity, Tuple[int, int]]], position: int
    ) -> Optional[Entity]:
        for entity, span in entities:
            if span[0] >= position:
                return entity
        return None

    @staticmethod
    def _normalize_predicate(token: str) -> str:
        lower = token.lower()
        mapping = {
            # activate family
            "activate": "activates", "activates": "activates",
            "activated": "activates", "activating": "activates",
            # stimulate family
            "stimulate": "stimulates", "stimulates": "stimulates",
            "stimulated": "stimulates", "stimulating": "stimulates",
            # induce family
            "induce": "induces", "induces": "induces",
            "induced": "induces", "inducing": "induces",
            # promote family
            "promote": "promotes", "promotes": "promotes",
            "promoted": "promotes", "promoting": "promotes",
            # upregulate family
            "upregulate": "upregulates", "upregulates": "upregulates",
            "upregulated": "upregulates", "upregulating": "upregulates",
            # increase family
            "increase": "increases", "increases": "increases",
            "increased": "increases", "increasing": "increases",
            # enhance family
            "enhance": "enhances", "enhances": "enhances",
            "enhanced": "enhances", "enhancing": "enhances",
            # inhibit family
            "inhibit": "inhibits", "inhibits": "inhibits",
            "inhibited": "inhibits", "inhibiting": "inhibits",
            # suppress family
            "suppress": "suppresses", "suppresses": "suppresses",
            "suppressed": "suppresses", "suppressing": "suppresses",
            # block family
            "block": "blocks", "blocks": "blocks",
            "blocked": "blocks", "blocking": "blocks",
            # downregulate family
            "downregulate": "downregulates", "downregulates": "downregulates",
            "downregulated": "downregulates", "downregulating": "downregulates",
            # decrease family
            "decrease": "decreases", "decreases": "decreases",
            "decreased": "decreases", "decreasing": "decreases",
            # reduce family
            "reduce": "reduces", "reduces": "reduces",
            "reduced": "reduces", "reducing": "reduces",
            # attenuate family
            "attenuate": "attenuates", "attenuates": "attenuates",
            "attenuated": "attenuates", "attenuating": "attenuates",
            # bind family
            "bind": "binds", "binds": "binds",
            "bound": "binds", "binding": "binds",
            # interact family
            "interact": "interacts_with", "interacts": "interacts_with",
            "interacted": "interacts_with", "interacting": "interacts_with",
            # target family
            "target": "targets", "targets": "targets",
            "targeted": "targets", "targeting": "targets",
            # require family
            "require": "requires", "requires": "requires",
            "required": "requires", "requiring": "requires",
            # association family
            "associated_with": "associated_with", "linked_to": "associated_with",
            "related_to": "associated_with",
        }
        return mapping.get(lower, lower)

    def _resolve_polarity(
        self,
        sentence: str,
        span_start: int,
        base_polarity: str,
        certainty: str,
        is_conditional: bool,
    ) -> str:
        if is_conditional:
            return "conditional"
        if certainty != "certain":
            return "uncertain"
        if not self._detect_negation(sentence, span_start):
            return base_polarity
        return "positive" if base_polarity == "negative" else "negative"

    def _score_relation_confidence(
        self, predicate: str, certainty: str, polarity: str
    ) -> float:
        base = 0.75
        if predicate in {"activates", "inhibits", "targets"}:
            base = 0.86
        elif predicate in {"binds", "interacts_with"}:
            base = 0.8
        elif predicate in {"associated_with", "correlates_with"}:
            base = 0.72
        penalty = {
            "certain": 0.0,
            "probable": 0.08,
            "possible": 0.14,
            "speculative": 0.2,
        }.get(certainty, 0.1)
        if polarity == "conditional":
            penalty += 0.08
        if polarity == "uncertain":
            penalty += 0.05
        return max(0.0, min(1.0, round(base - penalty + 0.02 * math.sqrt(base), 3)))

    def _is_conditional(self, sentence: str) -> bool:
        lower = sentence.lower()
        return any(term in lower for term in self.CONDITIONAL_TERMS)


class _SciSpacyExtractor:
    """Optional SciSpaCy-backed adapter that falls back to rule-based output."""

    def __init__(self) -> None:
        import spacy  # type: ignore
        import scispacy  # noqa: F401  # type: ignore

        self._fallback = RuleBasedExtractor()
        self._nlp = spacy.load("en_core_sci_sm")

    def extract_entities(
        self, text: str, pmid: str = "", year: Optional[int] = None
    ) -> List[Entity]:
        del year
        if not text:
            return []
        try:
            doc = self._nlp(text)
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("SciSpaCy entity extraction failed, falling back: %s", exc)
            return self._fallback.extract_entities(text, pmid=pmid)

        sentence_lookup = []
        offset = 0
        for sentence in self._fallback._segment_sentences(text):
            start = text.find(sentence, offset)
            if start < 0:
                start = offset
            sentence_lookup.append((start, start + len(sentence), sentence))
            offset = start + len(sentence)

        entities: List[Entity] = []
        seen = set()
        for ent in getattr(doc, "ents", []):
            label = self._fallback._classify_entity_type(ent.text)
            normalized = self._fallback._normalize_entity(ent.text)
            sentence_text = text
            for start, end, sentence in sentence_lookup:
                if start <= ent.start_char < end:
                    sentence_text = sentence
                    break
            key = (normalized, label, sentence_text)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    text=ent.text.strip(),
                    normalized=normalized,
                    entity_type=label,
                    pmid=pmid,
                    sentence=sentence_text,
                    confidence=0.88,
                )
            )
        return entities or self._fallback.extract_entities(text, pmid=pmid)

    def extract_relations(
        self, text: str, pmid: str = "", year: Optional[int] = None
    ) -> List[Relation]:
        return self._fallback.extract_relations(text, pmid=pmid, year=year)


class ExtractionPipeline:
    """Article-level extraction pipeline with optional SciSpaCy support."""

    def __init__(self, use_scispacy: bool = False):
        self.extractor: Any = RuleBasedExtractor()
        if use_scispacy:
            try:
                self.extractor = _SciSpacyExtractor()
            except Exception as exc:
                logger.warning("SciSpaCy unavailable; using rule-based extractor: %s", exc)
                self.extractor = RuleBasedExtractor()

    def process_articles(self, articles: List[Any]) -> Tuple[List[Entity], List[Relation]]:
        entities: List[Entity] = []
        relations: List[Relation] = []
        for article in articles or []:
            text = self._article_text(article)
            if not text:
                continue
            pmid = self._article_value(article, "pmid", "")
            year = self._article_year(article)
            entities.extend(self.extractor.extract_entities(text, pmid=pmid, year=year))
            relations.extend(self.extractor.extract_relations(text, pmid=pmid, year=year))
        return entities, self._deduplicate_relations(relations)

    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        grouped: Dict[Tuple[str, str, str, str, str], Relation] = {}
        for relation in relations:
            key = (
                relation.subject.lower(),
                relation.predicate,
                relation.object.lower(),
                relation.pmid,
                relation.polarity,
            )
            current = grouped.get(key)
            if current is None or relation.confidence > current.confidence:
                grouped[key] = relation
        return sorted(
            grouped.values(),
            key=lambda relation: (
                relation.pmid,
                relation.subject.lower(),
                relation.predicate,
                relation.object.lower(),
            ),
        )

    @staticmethod
    def _article_value(article: Any, field: str, default: Any = None) -> Any:
        if isinstance(article, dict):
            return article.get(field, default)
        return getattr(article, field, default)

    def _article_text(self, article: Any) -> str:
        title = self._article_value(article, "title", "") or ""
        abstract = self._article_value(article, "abstract", "") or ""
        combined = " ".join(part.strip() for part in (title, abstract) if part and part.strip())
        return combined.strip()

    def _article_year(self, article: Any) -> Optional[int]:
        value = (
            self._article_value(article, "publication_year")
            or self._article_value(article, "pub_year")
            or self._article_value(article, "year")
        )
        if value is None:
            pub_date = self._article_value(article, "pub_date")
            if isinstance(pub_date, str):
                match = re.match(r"(\d{4})", pub_date)
                if match:
                    value = match.group(1)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
