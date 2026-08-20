from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field
import collections
import json
import logging
import math
import os
import re

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None

if TYPE_CHECKING:  # pragma: no cover
    from extraction import Relation
    from graph import GapCandidate

logger = logging.getLogger(__name__)

_NEGATION_PATTERN = re.compile(
    r"\b(no association|not associated|no effect|did not|failed to|absence of|unrelated)\b",
    re.IGNORECASE,
)


def _utc_now_iso() -> str:
    datetime_mod = __import__("datetime")
    return datetime_mod.datetime.now(datetime_mod.timezone.utc).isoformat()


def _hash_hypothesis_id(node_a: str, node_b: str, query: str) -> str:
    hashlib_mod = __import__("hashlib")
    raw = f"{node_a}:{node_b}:{query}".encode()
    return hashlib_mod.md5(raw).hexdigest()[:12]


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _get_value(obj: Any, *keys: str, default: Any = None) -> Any:
    mapping = _coerce_mapping(obj)
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
        if key in mapping:
            return mapping[key]
    return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _normalize_type(entity_type: Any) -> str:
    return str(entity_type or "UNKNOWN").strip().upper()


def _canonical_type(entity_type: Any) -> str:
    normalized = _normalize_type(entity_type)
    if normalized in {"CHEMICAL", "DRUG", "COMPOUND", "SMALL_MOLECULE"}:
        return "drug"
    if normalized in {"GENE", "PROTEIN", "TARGET", "ENZYME"}:
        return "gene"
    if normalized in {"DISEASE", "DISORDER", "PHENOTYPE"}:
        return "disease"
    return normalized.lower()


def _extract_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"(19|20)\d{2}", str(value))
    if match:
        return int(match.group(0))
    return None


def _stringify_neighbors(common_neighbors: List[str]) -> str:
    clean = [str(item) for item in common_neighbors if str(item).strip()]
    if not clean:
        return "shared pathway"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    preview = ", ".join(clean[:3])
    if len(clean) > 3:
        preview += ", and related intermediates"
    return preview


def _ensure_evidence_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pmid": str(item.get("pmid", "") or ""),
        "sentence": str(item.get("sentence", "") or ""),
        "year": _extract_year(item.get("year")),
    }


def _dedupe_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        normalized = _ensure_evidence_item(item)
        key = (
            normalized.get("pmid", ""),
            normalized.get("sentence", ""),
            normalized.get("year"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


@dataclass
class Hypothesis:
    hypothesis_id: str
    hypothesis: str
    entities: List[str]
    mechanistic_rationale: str
    supporting_evidence: List[dict] = field(default_factory=list)
    contradicting_evidence: List[dict] = field(default_factory=list)
    novelty_status: str = "INSUFFICIENT_EVIDENCE"
    confidence: float = 0.0
    testable_prediction: str = ""
    suggested_experiment: str = ""
    limitations: str = ""
    alternative_explanations: str = ""
    devils_advocate: str = ""
    falsifiability: str = ""
    evidence_score: float = 0.0
    novelty_score: float = 0.0
    plausibility_score: float = 0.0
    feasibility_score: float = 0.0
    final_score: float = 0.0
    query: str = ""
    generated_at: str = field(default_factory=_utc_now_iso)
    method: str = "rule_based"


class EvidenceScorer:
    """Transparent scoring weights: evidence 0.35, novelty 0.30, plausibility 0.25, feasibility 0.10."""

    NOVELTY_SCORES = {
        "UNEXPLORED": 1.0,
        "INDIRECTLY_SUPPORTED": 0.8,
        "PARTIALLY_SUPPORTED": 0.5,
        "INSUFFICIENT_EVIDENCE": 0.6,
        "KNOWN": 0.0,
        "CONTRADICTED": 0.1,
    }

    @staticmethod
    def compute_evidence_score(supporting: List[dict], contradicting: List[dict]) -> float:
        base = min(len(supporting), 10) / 10.0
        diversity = len({e.get("pmid", "") for e in supporting}) / max(len(supporting), 1)
        contradiction_penalty = min(len(contradicting), 5) / 5.0 * 0.3
        score = base * 0.6 + diversity * 0.4 - contradiction_penalty
        return max(0.0, min(1.0, score))

    @classmethod
    def compute_novelty_score(cls, novelty_status: str) -> float:
        return cls.NOVELTY_SCORES.get(str(novelty_status or "").upper(), 0.6)

    @staticmethod
    def compute_plausibility_score(
        common_neighbors: List[str],
        node_a_type: str,
        node_b_type: str,
    ) -> float:
        score = min(len(common_neighbors), 5) / 5.0
        type_a = _canonical_type(node_a_type)
        type_b = _canonical_type(node_b_type)
        if ({type_a, type_b} == {"gene", "disease"}) or (
            "drug" in {type_a, type_b} and "gene" in {type_a, type_b}
        ):
            score += 0.2
        return max(0.0, min(1.0, score))

    @staticmethod
    def compute_feasibility_score(experiment: str) -> float:
        text = str(experiment or "").lower()
        if any(token in text for token in ("in vitro", "cell line", "assay")):
            return 0.8
        if any(token in text for token in ("mouse model", "animal model")):
            return 0.6
        if any(token in text for token in ("clinical trial", "patient")):
            return 0.4
        return 0.5

    @staticmethod
    def compute_final_score(
        evidence: float,
        novelty: float,
        plausibility: float,
        feasibility: float,
    ) -> float:
        score = 0.35 * evidence + 0.30 * novelty + 0.25 * plausibility + 0.10 * feasibility
        return max(0.0, min(1.0, score))


class RuleBasedHypothesisGenerator:
    DEFAULT_DEVILS_ADVOCATE = (
        "However, the observed network proximity may reflect shared research interest "
        "rather than direct biological interaction."
    )
    DEFAULT_FALSIFIABILITY = (
        "This hypothesis is falsifiable by demonstrating absence of effect in the proposed experiment."
    )
    DEFAULT_LIMITATIONS = (
        "Evidence is indirect (network-based); direct experimental validation is required."
    )

    def generate(
        self,
        candidate: "GapCandidate",
        relations: List["Relation"],
        query: str = "",
    ) -> Hypothesis:
        node_a = str(_get_value(candidate, "node_a", "concept_a", default="") or "")
        node_b = str(_get_value(candidate, "node_b", "concept_b", default="") or "")
        node_a_type = _normalize_type(_get_value(candidate, "node_a_type", "entity_type_a", default="UNKNOWN"))
        node_b_type = _normalize_type(_get_value(candidate, "node_b_type", "entity_type_b", default="UNKNOWN"))
        common_neighbors = [str(item) for item in _as_list(_get_value(candidate, "common_neighbors", "shared_neighbors", default=[]))]

        template = self._select_template(node_a_type, node_b_type)
        pathway_text = _stringify_neighbors(common_neighbors)

        supporting, contradicting = self._find_supporting_evidence(candidate, relations)
        novelty_status = self._infer_novelty_status(supporting, contradicting, common_neighbors)

        hypothesis_text = template["hypothesis"].format(
            node_a=node_a or "Entity A",
            node_b=node_b or "Entity B",
            common_neighbors=pathway_text,
        )
        rationale = template["rationale"].format(
            node_a=node_a or "Entity A",
            node_b=node_b or "Entity B",
            common_neighbors=pathway_text,
        )
        prediction = template["testable_prediction"].format(
            node_a=node_a or "Entity A",
            node_b=node_b or "Entity B",
            common_neighbors=pathway_text,
        )
        experiment = template["experiment"].format(
            node_a=node_a or "Entity A",
            node_b=node_b or "Entity B",
            common_neighbors=pathway_text,
        )

        alternative_explanations = (
            f"Observed overlap between {node_a or 'the entities'} and {node_b or 'the phenotype'} may arise from "
            f"shared upstream regulators such as {pathway_text}, publication bias, or broad pathway co-membership."
        )

        hypothesis = Hypothesis(
            hypothesis_id=_hash_hypothesis_id(node_a, node_b, query),
            hypothesis=hypothesis_text,
            entities=[item for item in [node_a, node_b] if item],
            mechanistic_rationale=rationale,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            novelty_status=novelty_status,
            testable_prediction=prediction,
            suggested_experiment=experiment,
            limitations=self.DEFAULT_LIMITATIONS,
            alternative_explanations=alternative_explanations,
            devils_advocate=self.DEFAULT_DEVILS_ADVOCATE,
            falsifiability=self.DEFAULT_FALSIFIABILITY,
            query=query,
            method="rule_based",
        )
        return self._score_hypothesis(hypothesis, common_neighbors, node_a_type, node_b_type)

    def _score_hypothesis(
        self,
        hypothesis: Hypothesis,
        common_neighbors: List[str],
        node_a_type: str,
        node_b_type: str,
    ) -> Hypothesis:
        evidence_score = EvidenceScorer.compute_evidence_score(
            hypothesis.supporting_evidence,
            hypothesis.contradicting_evidence,
        )
        novelty_score = EvidenceScorer.compute_novelty_score(hypothesis.novelty_status)
        plausibility_score = EvidenceScorer.compute_plausibility_score(
            common_neighbors,
            node_a_type,
            node_b_type,
        )
        feasibility_score = EvidenceScorer.compute_feasibility_score(hypothesis.suggested_experiment)
        final_score = EvidenceScorer.compute_final_score(
            evidence_score,
            novelty_score,
            plausibility_score,
            feasibility_score,
        )

        hypothesis.evidence_score = evidence_score
        hypothesis.novelty_score = novelty_score
        hypothesis.plausibility_score = plausibility_score
        hypothesis.feasibility_score = feasibility_score
        hypothesis.final_score = final_score
        hypothesis.confidence = max(0.0, min(1.0, round((evidence_score + plausibility_score + novelty_score) / 3.0, 4)))
        return hypothesis

    def _select_template(self, node_a_type: str, node_b_type: str) -> dict:
        pair = (_normalize_type(node_a_type), _normalize_type(node_b_type))
        templates = {
            ("GENE", "DISEASE"): {
                "hypothesis": "{node_a} may play a functional role in {node_b} through {common_neighbors} pathway activity",
                "rationale": "Both {node_a} and {node_b} are connected via {common_neighbors}, suggesting a potential mechanistic link.",
                "testable_prediction": "{node_a} knockdown or overexpression should alter {node_b} phenotype in model systems.",
                "experiment": "Perform siRNA knockdown of {node_a} in {node_b} cell lines; measure phenotypic endpoints at 72h.",
            },
            ("CHEMICAL", "GENE"): {
                "hypothesis": "{node_a} may modulate {node_b} expression or activity in relevant biological contexts",
                "rationale": "Shared network neighbors suggest {node_a} and {node_b} may interact through an intermediary.",
                "testable_prediction": "Treatment with {node_a} should significantly alter {node_b} protein levels.",
                "experiment": "Western blot / qPCR analysis of {node_b} in cells treated with {node_a} at IC50 concentrations.",
            },
            ("GENE", "GENE"): {
                "hypothesis": "{node_a} and {node_b} may co-regulate a shared biological process via {common_neighbors}",
                "rationale": "Shared interaction partners suggest functional convergence.",
                "testable_prediction": "Co-immunoprecipitation or proximity ligation assay should reveal interaction.",
                "experiment": "Co-IP followed by mass spectrometry in relevant cell line.",
            },
        }
        if pair in templates:
            return templates[pair]
        reverse_pair = (pair[1], pair[0])
        if reverse_pair in templates:
            return templates[reverse_pair]
        return {
            "hypothesis": "{node_a} may be biologically linked to {node_b} through {common_neighbors}",
            "rationale": "Shared network context involving {common_neighbors} suggests a plausible but unconfirmed relationship between {node_a} and {node_b}.",
            "testable_prediction": "Perturbing {node_a} or {node_b} should alter the shared pathway state associated with {common_neighbors}.",
            "experiment": "Run a targeted perturbation assay in an appropriate in vitro model and quantify pathway readouts linked to {common_neighbors}.",
        }

    def _find_supporting_evidence(
        self,
        candidate: "GapCandidate",
        relations: List["Relation"],
    ) -> Tuple[List[dict], List[dict]]:
        node_a = str(_get_value(candidate, "node_a", "concept_a", default="") or "")
        node_b = str(_get_value(candidate, "node_b", "concept_b", default="") or "")
        common_neighbors = {
            str(item)
            for item in _as_list(_get_value(candidate, "common_neighbors", "shared_neighbors", default=[]))
            if str(item).strip()
        }
        relevant = {node_a, node_b} | common_neighbors

        supporting: List[Dict[str, Any]] = []
        contradicting: List[Dict[str, Any]] = []

        for relation in relations or []:
            src = str(_get_value(relation, "source_entity", "source", default="") or "")
            tgt = str(_get_value(relation, "target_entity", "target", default="") or "")
            if not src and not tgt:
                continue
            endpoints = {src, tgt}
            if not endpoints & relevant:
                continue

            touches_pair = node_a in endpoints and node_b in endpoints if node_a and node_b else False
            touches_bridge = bool(endpoints & common_neighbors) and bool(endpoints & {node_a, node_b})
            if not (touches_pair or touches_bridge):
                continue

            sentence = str(
                _get_value(
                    relation,
                    "sentence",
                    "sentence_context",
                    "context",
                    "evidence_sentence",
                    default="",
                )
                or ""
            )
            rel_type = str(_get_value(relation, "relationship_type", "type", default="") or "")
            year = _extract_year(_get_value(relation, "year", "pub_year", "publication_year", default=None))
            evidence_item = {
                "pmid": str(_get_value(relation, "pmid", "evidence_pmid", default="") or ""),
                "sentence": sentence or rel_type,
                "year": year,
            }
            if _NEGATION_PATTERN.search(sentence) or _NEGATION_PATTERN.search(rel_type):
                contradicting.append(evidence_item)
            else:
                supporting.append(evidence_item)

        return _dedupe_evidence(supporting), _dedupe_evidence(contradicting)

    @staticmethod
    def _infer_novelty_status(
        supporting: List[dict],
        contradicting: List[dict],
        common_neighbors: List[str],
    ) -> str:
        if contradicting and len(contradicting) >= len(supporting):
            return "CONTRADICTED"
        if len(supporting) >= 5:
            return "KNOWN"
        if len(supporting) >= 3:
            return "PARTIALLY_SUPPORTED"
        if len(supporting) >= 1:
            return "INDIRECTLY_SUPPORTED"
        if common_neighbors:
            return "UNEXPLORED"
        return "INSUFFICIENT_EVIDENCE"


class LLMHypothesisGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", temperature: float = 0.3):
        self.api_key = str(api_key or "")
        self.model = model
        self.temperature = temperature
        self.fallback = RuleBasedHypothesisGenerator()
        self.available = bool(genai is not None and self.api_key)
        if self.available:
            genai.configure(api_key=self.api_key)

    def generate(
        self,
        candidate: "GapCandidate",
        relations: List["Relation"],
        query: str = "",
    ) -> Hypothesis:
        if not self.available:
            return self.fallback.generate(candidate, relations, query=query)

        supporting, contradicting = self.fallback._find_supporting_evidence(candidate, relations)
        prompt = self._build_prompt(candidate, supporting, contradicting, query)

        def _call_model() -> str:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature, "response_mime_type": "application/json"},
            )
            return getattr(response, "text", "") or ""

        try:
            response_text = self._retry_with_backoff(_call_model)
            hypothesis = self._parse_response(response_text, candidate, query)
            hypothesis.method = "llm"
            if not hypothesis.supporting_evidence:
                hypothesis.supporting_evidence = supporting
            if not hypothesis.contradicting_evidence:
                hypothesis.contradicting_evidence = contradicting
            return hypothesis
        except Exception as exc:  # pragma: no cover - network/API path
            logger.warning("LLM hypothesis generation failed, using fallback: %s", exc)
            return self.fallback.generate(candidate, relations, query=query)

    def _build_prompt(
        self,
        candidate: Any,
        supporting: List[dict],
        contradicting: List[dict],
        query: str,
    ) -> str:
        node_a = str(_get_value(candidate, "node_a", "concept_a", default="") or "")
        node_b = str(_get_value(candidate, "node_b", "concept_b", default="") or "")
        node_a_type = _normalize_type(_get_value(candidate, "node_a_type", "entity_type_a", default="UNKNOWN"))
        node_b_type = _normalize_type(_get_value(candidate, "node_b_type", "entity_type_b", default="UNKNOWN"))
        common_neighbors = _as_list(_get_value(candidate, "common_neighbors", "shared_neighbors", default=[]))
        payload = {
            "query": query,
            "candidate": {
                "node_a": node_a,
                "node_b": node_b,
                "node_a_type": node_a_type,
                "node_b_type": node_b_type,
                "common_neighbors": common_neighbors,
            },
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
        }
        schema = {
            "hypothesis_id": "string",
            "hypothesis": "string",
            "entities": ["string"],
            "mechanistic_rationale": "string",
            "supporting_evidence": [{"pmid": "string", "sentence": "string", "year": 2024}],
            "contradicting_evidence": [{"pmid": "string", "sentence": "string", "year": 2024}],
            "novelty_status": "UNEXPLORED|INDIRECTLY_SUPPORTED|PARTIALLY_SUPPORTED|INSUFFICIENT_EVIDENCE|KNOWN|CONTRADICTED",
            "confidence": 0.0,
            "testable_prediction": "string",
            "suggested_experiment": "string",
            "limitations": "string",
            "alternative_explanations": "string",
            "devils_advocate": "string",
            "falsifiability": "string",
            "evidence_score": 0.0,
            "novelty_score": 0.0,
            "plausibility_score": 0.0,
            "feasibility_score": 0.0,
            "final_score": 0.0,
            "query": "string",
            "generated_at": "ISO-8601 string",
            "method": "llm",
        }
        return (
            "You are generating scientific hypotheses from explicit evidence only.\n"
            "Rules:\n"
            "1. Do NOT invent facts, papers, PMIDs, years, mechanisms, or results.\n"
            "2. Base every claim only on the provided candidate and evidence.\n"
            "3. If evidence is weak, say so explicitly in limitations or novelty_status.\n"
            "4. Return valid JSON only, with fields matching the requested schema.\n"
            "5. Include alternative_explanations and devils_advocate.\n\n"
            f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            f"OUTPUT JSON SCHEMA EXAMPLE:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

    def _parse_response(self, response_text: str, candidate: Any, query: str) -> Hypothesis:
        cleaned = str(response_text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        node_a = str(_get_value(candidate, "node_a", "concept_a", default="") or "")
        node_b = str(_get_value(candidate, "node_b", "concept_b", default="") or "")

        return Hypothesis(
            hypothesis_id=str(data.get("hypothesis_id") or _hash_hypothesis_id(node_a, node_b, query)),
            hypothesis=str(data.get("hypothesis") or ""),
            entities=[str(item) for item in _as_list(data.get("entities") or [node_a, node_b]) if str(item).strip()],
            mechanistic_rationale=str(data.get("mechanistic_rationale") or ""),
            supporting_evidence=_dedupe_evidence(_as_list(data.get("supporting_evidence"))),
            contradicting_evidence=_dedupe_evidence(_as_list(data.get("contradicting_evidence"))),
            novelty_status=str(data.get("novelty_status") or "INSUFFICIENT_EVIDENCE").upper(),
            confidence=float(data.get("confidence") or 0.0),
            testable_prediction=str(data.get("testable_prediction") or ""),
            suggested_experiment=str(data.get("suggested_experiment") or ""),
            limitations=str(data.get("limitations") or RuleBasedHypothesisGenerator.DEFAULT_LIMITATIONS),
            alternative_explanations=str(data.get("alternative_explanations") or ""),
            devils_advocate=str(data.get("devils_advocate") or RuleBasedHypothesisGenerator.DEFAULT_DEVILS_ADVOCATE),
            falsifiability=str(data.get("falsifiability") or RuleBasedHypothesisGenerator.DEFAULT_FALSIFIABILITY),
            evidence_score=float(data.get("evidence_score") or 0.0),
            novelty_score=float(data.get("novelty_score") or 0.0),
            plausibility_score=float(data.get("plausibility_score") or 0.0),
            feasibility_score=float(data.get("feasibility_score") or 0.0),
            final_score=float(data.get("final_score") or 0.0),
            query=str(data.get("query") or query),
            generated_at=str(data.get("generated_at") or _utc_now_iso()),
            method="llm",
        )

    def _retry_with_backoff(self, fn, max_retries: int = 3):
        time_mod = __import__("time")
        delay = 1.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - network/API path
                last_exc = exc
                if attempt >= max_retries - 1:
                    break
                message = str(exc).lower()
                if any(token in message for token in ("429", "rate", "quota", "resource exhausted")):
                    time_mod.sleep(delay)
                    delay *= 2.0
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM call failed without exception")


class HypothesisPipeline:
    def __init__(self, config_or_api_key, use_llm: bool = False):
        self.config = None if isinstance(config_or_api_key, str) else config_or_api_key
        self.api_key = config_or_api_key if isinstance(config_or_api_key, str) else self._config_get(
            config_or_api_key,
            "api_key",
            "gemini_api_key",
            "GEMINI_API_KEY",
            default="",
        )
        model = self._config_get(config_or_api_key, "model", "gemini_model", default="gemini-2.0-flash")
        temperature = self._config_get(config_or_api_key, "temperature", "gemini_temperature", default=0.3)
        self.use_llm = bool(use_llm)
        self.rule_based = RuleBasedHypothesisGenerator()
        self.generator = (
            LLMHypothesisGenerator(api_key=self.api_key, model=model, temperature=float(temperature))
            if self.use_llm
            else self.rule_based
        )

    @staticmethod
    def _config_get(config: Any, *keys: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            for key in keys:
                if key in config:
                    return config[key]
            return default
        for key in keys:
            if hasattr(config, key):
                return getattr(config, key)
        return default

    def generate_hypotheses(
        self,
        candidates: List["GapCandidate"],
        relations: List["Relation"],
        query: str = "",
        top_k: int = 5,
    ) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        for candidate in (candidates or [])[: max(int(top_k or 0), 0)]:
            hypothesis = self.generator.generate(candidate, relations, query=query)
            common_neighbors = [
                str(item)
                for item in _as_list(_get_value(candidate, "common_neighbors", "shared_neighbors", default=[]))
            ]
            node_a_type = _normalize_type(_get_value(candidate, "node_a_type", "entity_type_a", default="UNKNOWN"))
            node_b_type = _normalize_type(_get_value(candidate, "node_b_type", "entity_type_b", default="UNKNOWN"))
            self._apply_scores(hypothesis, common_neighbors, node_a_type, node_b_type)
            hypotheses.append(hypothesis)
        return self.rank_hypotheses(hypotheses)

    def _apply_scores(
        self,
        hypothesis: Hypothesis,
        common_neighbors: List[str],
        node_a_type: str,
        node_b_type: str,
    ) -> None:
        hypothesis.evidence_score = EvidenceScorer.compute_evidence_score(
            hypothesis.supporting_evidence,
            hypothesis.contradicting_evidence,
        )
        hypothesis.novelty_score = EvidenceScorer.compute_novelty_score(hypothesis.novelty_status)
        hypothesis.plausibility_score = EvidenceScorer.compute_plausibility_score(
            common_neighbors,
            node_a_type,
            node_b_type,
        )
        hypothesis.feasibility_score = EvidenceScorer.compute_feasibility_score(hypothesis.suggested_experiment)
        hypothesis.final_score = EvidenceScorer.compute_final_score(
            hypothesis.evidence_score,
            hypothesis.novelty_score,
            hypothesis.plausibility_score,
            hypothesis.feasibility_score,
        )
        hypothesis.confidence = max(
            0.0,
            min(
                1.0,
                round(
                    (hypothesis.evidence_score + hypothesis.novelty_score + hypothesis.plausibility_score) / 3.0,
                    4,
                ),
            ),
        )

    @staticmethod
    def rank_hypotheses(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        return sorted(hypotheses, key=lambda item: item.final_score, reverse=True)

    @staticmethod
    def export_hypotheses(hypotheses: List[Hypothesis], filepath: str) -> None:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in hypotheses], handle, ensure_ascii=False, indent=2)

    @staticmethod
    def to_dataframe(hypotheses: List[Hypothesis]) -> dict:
        rows = collections.defaultdict(list)
        for item in hypotheses:
            payload = asdict(item)
            for key, value in payload.items():
                rows[key].append(value)
        return dict(rows)


def generate_and_rank(
    candidates,
    relations,
    query: str = "",
    config=None,
    top_k: int = 5,
) -> List[Hypothesis]:
    use_llm = False
    if isinstance(config, dict):
        use_llm = bool(config.get("use_llm", False))
    elif config is not None and hasattr(config, "use_llm"):
        use_llm = bool(getattr(config, "use_llm"))
    pipeline = HypothesisPipeline(config, use_llm=use_llm)
    return pipeline.generate_hypotheses(candidates, relations, query=query, top_k=top_k)
