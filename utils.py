from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import pickle
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

LOGGER_NAME = "bionova"
_JSON_INDENT = 2
_T = TypeVar("_T")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the shared BioNOVA logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger


@dataclass
class BioNOVAConfig:
    entrez_email: str = ""
    entrez_api_key: str = ""
    max_results: int = 200
    batch_size: int = 100
    rate_limit_no_key: float = 3.0
    rate_limit_with_key: float = 10.0
    kg_similarity_threshold: float = 0.75
    min_shared_neighbors: int = 2
    hypothesis_top_gaps: int = 5
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    log_level: str = "INFO"
    random_seed: int = 42
    historical_cutoff_year: Optional[int] = None

    @classmethod
    def load_env(cls) -> "BioNOVAConfig":
        load_dotenv()
        logger = logging.getLogger(LOGGER_NAME)

        def _get_int(name: str, default: int) -> int:
            value = os.getenv(name)
            if value in (None, ""):
                return default
            try:
                return int(value)
            except ValueError:
                logger.warning("Invalid integer for %s=%r; using %r", name, value, default)
                return default

        def _get_float(name: str, default: float) -> float:
            value = os.getenv(name)
            if value in (None, ""):
                return default
            try:
                return float(value)
            except ValueError:
                logger.warning("Invalid float for %s=%r; using %r", name, value, default)
                return default

        cutoff = validate_year(os.getenv("HISTORICAL_CUTOFF_YEAR"))
        return cls(
            entrez_email=os.getenv("ENTREZ_EMAIL", ""),
            entrez_api_key=os.getenv("ENTREZ_API_KEY", ""),
            max_results=_get_int("MAX_RESULTS", 200),
            batch_size=_get_int("BATCH_SIZE", 100),
            rate_limit_no_key=_get_float("RATE_LIMIT_NO_KEY", 3.0),
            rate_limit_with_key=_get_float("RATE_LIMIT_WITH_KEY", 10.0),
            kg_similarity_threshold=_get_float("KG_SIMILARITY_THRESHOLD", 0.75),
            min_shared_neighbors=_get_int("MIN_SHARED_NEIGHBORS", 2),
            hypothesis_top_gaps=_get_int("HYPOTHESIS_TOP_GAPS", 5),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            random_seed=_get_int("RANDOM_SEED", 42),
            historical_cutoff_year=cutoff,
        )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.lower().strip()
    return re.sub(r"\s+", " ", normalized)



def truncate(text: str, max_len: int = 80, suffix: str = "…") -> str:
    if len(text or "") <= max_len:
        return text or ""
    return f"{text[:max_len]}{suffix}"



def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)



def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")



def to_json(obj: Any, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, default=_json_default, ensure_ascii=False, indent=_JSON_INDENT)



def from_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)



def disk_cache(path: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    cache_path = Path(path)

    def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with cache_path.open("rb") as handle:
                    cache = pickle.load(handle)
                    if not isinstance(cache, dict):
                        cache = {}
            except FileNotFoundError:
                cache = {}
            except (pickle.PickleError, EOFError, OSError) as exc:
                logging.getLogger(LOGGER_NAME).warning(
                    "Resetting cache %s after read failure: %s", cache_path, exc
                )
                cache = {}

            try:
                cache_key = hashlib.sha256(
                    pickle.dumps((args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
                ).hexdigest()
            except (pickle.PickleError, TypeError, AttributeError) as exc:
                raise TypeError(f"Arguments for {func.__name__} are not cacheable: {exc}") from exc

            if cache_key in cache:
                return cache[cache_key]

            result = func(*args, **kwargs)
            cache[cache_key] = result
            with cache_path.open("wb") as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return result

        return wrapper

    return decorator


class Timer:
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        self._end = None
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._end = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        end_time = self._end if self._end is not None else time.monotonic()
        return end_time - self._start



def validate_pmid(pmid: str) -> bool:
    return bool(re.fullmatch(r"\d{1,8}", str(pmid or "")))



def validate_year(year: Any) -> Optional[int]:
    if year in (None, ""):
        return None
    try:
        value = int(year)
    except (TypeError, ValueError):
        return None
    from datetime import timezone as _tz
    current_year = datetime.now(_tz.utc).year + 1
    return value if 1900 <= value <= current_year else None



def set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        logging.getLogger(LOGGER_NAME).warning("NumPy is not installed; skipping NumPy seed setup")
        return
    np.random.seed(seed)



def format_evidence_summary(relations: List[dict]) -> str:
    lines: List[str] = []
    for relation in relations or []:
        source = relation.get("source") or relation.get("source_entity") or relation.get("concept_a") or "?"
        target = relation.get("target") or relation.get("target_entity") or relation.get("concept_b") or "?"
        rel_type = relation.get("type") or relation.get("relationship_type") or "related_to"
        pmid = relation.get("pmid") or relation.get("evidence_pmid") or "n/a"
        score = relation.get("confidence") or relation.get("confidence_score")
        score_text = f" ({float(score):.2f})" if score is not None else ""
        lines.append(f"- {source} -> {target} [{rel_type}] PMID:{pmid}{score_text}")
    return "\n".join(lines)



def escape_html(text: str) -> str:
    return html.escape(text or "", quote=True)


__all__ = [
    "BioNOVAConfig",
    "Timer",
    "disk_cache",
    "escape_html",
    "format_evidence_summary",
    "from_json",
    "normalize_text",
    "set_random_seed",
    "setup_logging",
    "slugify",
    "to_json",
    "truncate",
    "validate_pmid",
    "validate_year",
]
