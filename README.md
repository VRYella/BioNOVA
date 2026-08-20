# BioNOVA

**Biomedical Literature Mining and Hypothesis Discovery**

BioNOVA is a research-grade computational framework for biomedical literature mining, evidence-grounded knowledge graph construction, temporal gap discovery, novelty assessment, and hypothesis generation.

---

## Features

- **PubMed retrieval** — batch-fetched articles with rate limiting and retry logic
- **Biomedical entity extraction** — rule-based NER (genes, diseases, chemicals, pathways) with optional SciSpaCy
- **Relation extraction** — predicate-aware extraction with negation, uncertainty, and polarity detection
- **Evidence knowledge graph** — strict separation of biological and semantic/co-occurrence edges
- **Temporal analysis** — publication-year-aware snapshots and historical cutoff support
- **Gap discovery** — Jaccard, Adamic–Adar, Resource Allocation, and common-neighbour link prediction
- **Novelty verification** — explicit classification (KNOWN / PARTIALLY_SUPPORTED / INDIRECTLY_SUPPORTED / UNEXPLORED / CONTRADICTED / INSUFFICIENT_EVIDENCE)
- **Hypothesis generation** — rule-based (deterministic, no API key required) or Gemini LLM-assisted
- **Transparent scoring** — evidence, novelty, plausibility, and feasibility scores with documented formulas
- **Streamlit UI** — professional research interface
- **Local notebook** — offline demo requiring no API keys

---

## Installation

```bash
git clone https://github.com/VRYella/BioNOVA
cd BioNOVA
pip install -r requirements.txt
```

For full NLP enhancement (optional):
```bash
pip install scispacy spacy
python -m spacy download en_core_sci_lg
```

---

## Quick Start

### Streamlit application
```bash
streamlit run app.py
```

### Local notebook (no API keys required)
```bash
jupyter notebook BioNOVA_Local_End_to_End.ipynb
```

### Command-line pipeline
```python
from retrieval import create_sample_articles, DataCleaner
from extraction import ExtractionPipeline
from graph import build_graph
from hypothesis import generate_and_rank

articles = create_sample_articles()
articles = DataCleaner().clean_articles(articles)
entities, relations = ExtractionPipeline().process_articles(articles)
bg = build_graph(relations)
gaps = bg.find_gap_candidates(top_k=10)
hypotheses = generate_and_rank(gaps, relations, query="BRCA1 cancer")
```

---

## Configuration

Create a `.env` file in the repository root:

```env
ENTREZ_EMAIL=your@email.com        # Required for PubMed access
ENTREZ_API_KEY=optional_key        # Optional: 10 req/s instead of 3
GEMINI_API_KEY=your_gemini_key     # Optional: LLM hypothesis generation
GEMINI_MODEL=gemini-2.0-flash      # Optional: default model
LOG_LEVEL=INFO                     # Optional: DEBUG / INFO / WARNING
```

If `ENTREZ_EMAIL` is not set, the demo dataset is used automatically.

---

## Architecture

```
PubMed / local data
        ↓
   retrieval.py         — article fetch, XML parsing, deduplication
        ↓
  extraction.py         — entity NER, relation extraction, negation/uncertainty
        ↓
     graph.py           — biological KG, temporal graph, gap detection, novelty
        ↓
  hypothesis.py         — hypothesis generation, evidence scoring, ranking
        ↓
    utils.py            — config, logging, serialization, validation
        ↓
      app.py            — Streamlit interface (no scientific algorithms)
```

**Critical design principle:** Biological edges (e.g., `drug → inhibits → protein`) and semantic/co-occurrence edges are maintained in separate graphs. Semantic proximity is never automatically treated as biological evidence.

---

## Testing

```bash
python -m pytest test_bionova.py -v
```

Tests require no internet access. All PubMed operations use mock/local data.

Coverage:
- Retrieval: valid/empty/malformed XML, duplicate PMIDs, missing abstracts
- NLP: entity/relation extraction, negation, uncertainty, provenance
- Graph: biological vs semantic separation, edge weighting, temporal snapshots
- Novelty: KNOWN / INDIRECT / CONTRADICTED classification
- Hypothesis: generation, scoring, ranking, reproducibility, export
- End-to-end: full pipeline, cutoff year, empty inputs, malformed relations

---

## Output Format

All outputs are exportable as JSON from the UI or notebook:

- `articles.json` — retrieved articles (PMID, title, abstract, year, journal, authors)
- `relations.json` — extracted relations with provenance (sentence, PMID, year, polarity, certainty)
- `gap_candidates.json` — gap candidates with scores and novelty status
- `hypotheses.json` — ranked hypotheses with full evidence and scoring breakdown

---

## Limitations

- Rule-based NER is heuristic; false positives and missed entities are expected without SciSpaCy
- Novelty classification is based on the retrieved literature corpus; it does not query external ontologies
- Hypothesis generation quality is higher with the Gemini LLM mode
- Future-literature validation (retrospective/temporal) has not been formally evaluated on gold-standard benchmarks
- Extraction performance metrics (precision, recall, F1) require annotated gold-standard data not included in this release

---

## Reproducibility

Every analysis records: query, date, retrieval parameters, publication cutoff, number of articles/entities/relations, algorithm parameters, and software version. Results can be reproduced from saved JSON outputs.

---

## Citation

If you use BioNOVA in your research, please cite:

> BioNOVA: Biomedical Literature Mining and Hypothesis Discovery.  
> [manuscript reference]

---

## License

[See LICENSE file]
