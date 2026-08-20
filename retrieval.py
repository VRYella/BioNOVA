from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional
from xml.etree import ElementTree as ET

import requests

from utils import BioNOVAConfig, normalize_text, validate_pmid, validate_year

logger = logging.getLogger("bionova.retrieval")

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: Optional[int]
    mesh_terms: List[str]
    keywords: List[str]
    doi: Optional[str]


class PubMedRetriever:
    def __init__(self, config: BioNOVAConfig):
        self.config = config
        rate_limit = config.rate_limit_with_key if config.entrez_api_key else config.rate_limit_no_key
        self._min_interval = 1.0 / rate_limit if rate_limit > 0 else 0.0
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.parser = XMLParser()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait_time = self._min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self._last_request_time = time.monotonic()

    def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
        if not query or not query.strip():
            logger.warning("Cannot search PubMed with an empty query")
            return []

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results or self.config.max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.config.entrez_email:
            params["email"] = self.config.entrez_email
        if self.config.entrez_api_key:
            params["api_key"] = self.config.entrez_api_key

        try:
            self._throttle()
            response = self.session.get(_ESEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            ids = payload.get("esearchresult", {}).get("idlist", [])
            return [pmid for pmid in ids if validate_pmid(pmid)]
        except requests.RequestException as exc:
            logger.error("PubMed search request failed for %r: %s", query, exc)
        except (ValueError, TypeError, AttributeError) as exc:
            logger.error("PubMed search response parsing failed for %r: %s", query, exc)
        return []

    def fetch_records(self, pmids: List[str]) -> str:
        if not pmids:
            return ""

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.config.entrez_email:
            params["email"] = self.config.entrez_email
        if self.config.entrez_api_key:
            params["api_key"] = self.config.entrez_api_key

        self._throttle()
        response = self.session.get(_EFETCH_URL, params=params, timeout=60)
        response.raise_for_status()
        if not response.text.strip():
            raise ValueError("NCBI efetch returned an empty response")
        return response.text

    def fetch_with_progress(
        self,
        query: str,
        max_results: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Article]:
        pmids = self.search(query, max_results or self.config.max_results)
        if not pmids:
            logger.warning("No PubMed IDs found for query=%r", query)
            return []

        articles: List[Article] = []
        total = len(pmids)
        fetched = 0

        for start in range(0, total, self.config.batch_size):
            batch = pmids[start : start + self.config.batch_size]
            xml_str = self._fetch_batch_with_retry(batch)
            if xml_str:
                articles.extend(self.parser.parse_xml(xml_str))
            fetched += len(batch)
            if progress_callback is not None:
                progress_callback(fetched, total)
        return articles

    def _fetch_batch_with_retry(self, pmids: List[str], max_retries: int = 3) -> str:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                return self.fetch_records(pmids)
            except (requests.RequestException, ValueError) as exc:
                if attempt >= max_retries:
                    logger.error(
                        "Failed to fetch batch after %d attempts for PMIDs starting with %s: %s",
                        max_retries,
                        pmids[:3],
                        exc,
                    )
                    return ""
                logger.warning(
                    "Retrying failed batch fetch %d/%d for PMIDs starting with %s in %.1fs: %s",
                    attempt,
                    max_retries,
                    pmids[:3],
                    delay,
                    exc,
                )
                time.sleep(delay)
                delay *= 2
        return ""


class XMLParser:
    def parse_xml(self, xml_str: str) -> List[Article]:
        if not xml_str or not xml_str.strip():
            return []
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as exc:
            logger.error("Failed to parse PubMed XML: %s", exc)
            return []

        articles: List[Article] = []
        for article_elem in root.findall(".//PubmedArticle"):
            article = self._parse_article(article_elem)
            if article is not None:
                articles.append(article)
        return articles

    def _parse_article(self, article_elem: ET.Element) -> Optional[Article]:
        try:
            medline_elem = article_elem.find("./MedlineCitation")
            article_meta = article_elem.find("./MedlineCitation/Article")
            if medline_elem is None or article_meta is None:
                logger.warning("Skipping malformed article record without MedlineCitation/Article")
                return None

            pmid = self._text(medline_elem, "./PMID")
            if not validate_pmid(pmid):
                logger.warning("Skipping article with invalid PMID: %r", pmid)
                return None

            title = self._inner_text(article_meta.find("./ArticleTitle"))
            abstract = self._extract_abstract(article_meta)
            journal = self._text(article_meta, "./Journal/Title") or self._text(medline_elem, "./MedlineJournalInfo/MedlineTA")
            year = self._extract_year(medline_elem)
            authors = self._extract_authors(medline_elem)
            mesh_terms = self._extract_mesh(medline_elem)
            keywords = self._extract_keywords(medline_elem)
            doi = self._extract_doi(article_elem)

            return Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                mesh_terms=mesh_terms,
                keywords=keywords,
                doi=doi,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed article record: %s", exc)
            return None

    def _extract_year(self, medline_elem: ET.Element) -> Optional[int]:
        direct_year = self._text(medline_elem, "./Article/Journal/JournalIssue/PubDate/Year")
        valid_direct_year = validate_year(direct_year)
        if valid_direct_year is not None:
            return valid_direct_year

        medline_date = self._text(medline_elem, "./Article/Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date:
            match = re.search(r"\b(\d{4})\b", medline_date)
            if match:
                return validate_year(match.group(1))

        article_date = self._text(medline_elem, "./Article/ArticleDate/Year")
        return validate_year(article_date)

    def _extract_authors(self, medline_elem: ET.Element) -> List[str]:
        authors: List[str] = []
        for author_elem in medline_elem.findall(".//AuthorList/Author"):
            collective = self._text(author_elem, "./CollectiveName")
            if collective:
                authors.append(collective)
                continue
            last_name = self._text(author_elem, "./LastName")
            fore_name = self._text(author_elem, "./ForeName")
            if last_name and fore_name:
                authors.append(f"{last_name}, {fore_name}")
            elif last_name:
                authors.append(last_name)
            elif fore_name:
                authors.append(fore_name)
        return authors

    def _extract_mesh(self, medline_elem: ET.Element) -> List[str]:
        mesh_terms: List[str] = []
        for heading in medline_elem.findall(".//MeshHeadingList/MeshHeading"):
            descriptor = self._inner_text(heading.find("./DescriptorName"))
            if not descriptor:
                continue
            qualifiers = [self._inner_text(elem) for elem in heading.findall("./QualifierName") if self._inner_text(elem)]
            if qualifiers:
                mesh_terms.extend([f"{descriptor} / {qualifier}" for qualifier in qualifiers])
            else:
                mesh_terms.append(descriptor)
        return mesh_terms

    def _extract_keywords(self, medline_elem: ET.Element) -> List[str]:
        keywords: List[str] = []
        for keyword_elem in medline_elem.findall(".//KeywordList/Keyword"):
            keyword = self._inner_text(keyword_elem)
            if keyword:
                keywords.append(keyword)
        return keywords

    def _extract_abstract(self, article_meta: ET.Element) -> str:
        abstract_elem = article_meta.find("./Abstract")
        if abstract_elem is None:
            return ""
        sections: List[str] = []
        for text_elem in abstract_elem.findall("./AbstractText"):
            text = self._inner_text(text_elem)
            label = (text_elem.attrib.get("Label") or "").strip()
            if text and label:
                sections.append(f"{label}: {text}")
            elif text:
                sections.append(text)
        return "\n".join(sections).strip()

    def _extract_doi(self, article_elem: ET.Element) -> Optional[str]:
        for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if (article_id.attrib.get("IdType") or "").lower() == "doi":
                doi = self._inner_text(article_id)
                return doi or None
        for location_id in article_elem.findall(".//ELocationID"):
            if (location_id.attrib.get("EIdType") or "").lower() == "doi":
                doi = self._inner_text(location_id)
                return doi or None
        return None

    def _text(self, parent: Optional[ET.Element], path: str) -> str:
        if parent is None:
            return ""
        child = parent.find(path)
        return self._inner_text(child)

    def _inner_text(self, elem: Optional[ET.Element]) -> str:
        if elem is None:
            return ""
        return "".join(elem.itertext()).strip()


class DataCleaner:
    def clean_articles(self, articles: List[Article]) -> List[Article]:
        cleaned: List[Article] = []
        seen_pmids = set()
        for article in articles:
            if not validate_pmid(article.pmid):
                logger.warning("Dropping article with invalid PMID during cleaning: %r", article.pmid)
                continue
            if article.pmid in seen_pmids:
                continue
            normalized = self.normalize_article(article)
            if not normalized.title and not normalized.abstract:
                continue
            seen_pmids.add(normalized.pmid)
            cleaned.append(normalized)
        return cleaned

    def normalize_article(self, article: Article) -> Article:
        def _dedupe(values: List[str]) -> List[str]:
            seen = set()
            ordered: List[str] = []
            for value in values:
                if value and value not in seen:
                    seen.add(value)
                    ordered.append(value)
            return ordered

        normalized_authors = _dedupe([normalize_text(author) for author in article.authors if author])
        normalized_mesh = _dedupe([normalize_text(term) for term in article.mesh_terms if term])
        normalized_keywords = _dedupe([normalize_text(keyword) for keyword in article.keywords if keyword])
        return Article(
            pmid=str(article.pmid).strip(),
            title=normalize_text(article.title),
            abstract=normalize_text(article.abstract),
            authors=normalized_authors,
            journal=normalize_text(article.journal),
            year=validate_year(article.year),
            mesh_terms=normalized_mesh,
            keywords=normalized_keywords,
            doi=normalize_text(article.doi) if article.doi else None,
        )


class MockRetriever:
    def __init__(self, articles: Optional[List[Article]] = None):
        self.articles = list(articles) if articles is not None else create_sample_articles()

    def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
        limit = max_results or len(self.articles)
        query_terms = {term for term in normalize_text(query).split() if term}
        if not query_terms:
            return [article.pmid for article in self.articles[:limit]]

        matched = []
        for article in self.articles:
            haystack = " ".join(
                [article.title, article.abstract, article.journal, " ".join(article.keywords), " ".join(article.mesh_terms)]
            )
            normalized_haystack = normalize_text(haystack)
            if all(term in normalized_haystack for term in query_terms):
                matched.append(article.pmid)
        return matched[:limit]

    def fetch_records(self, pmids: List[str]) -> str:
        selected = [article for article in self.articles if article.pmid in set(pmids)]
        root = ET.Element("PubmedArticleSet")
        for article in selected:
            root.append(self._article_to_xml(article))
        return ET.tostring(root, encoding="unicode")

    def fetch_with_progress(
        self,
        query: str,
        max_results: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Article]:
        pmids = self.search(query, max_results)
        selected = [article for article in self.articles if article.pmid in set(pmids)]
        total = len(selected)
        if progress_callback is not None:
            progress_callback(total, total)
        return selected

    def _article_to_xml(self, article: Article) -> ET.Element:
        pubmed_article = ET.Element("PubmedArticle")
        medline = ET.SubElement(pubmed_article, "MedlineCitation")
        ET.SubElement(medline, "PMID").text = article.pmid
        article_meta = ET.SubElement(medline, "Article")
        ET.SubElement(article_meta, "ArticleTitle").text = article.title
        abstract = ET.SubElement(article_meta, "Abstract")
        ET.SubElement(abstract, "AbstractText").text = article.abstract
        journal = ET.SubElement(article_meta, "Journal")
        ET.SubElement(journal, "Title").text = article.journal
        issue = ET.SubElement(journal, "JournalIssue")
        pub_date = ET.SubElement(issue, "PubDate")
        if article.year is not None:
            ET.SubElement(pub_date, "Year").text = str(article.year)
        author_list = ET.SubElement(article_meta, "AuthorList")
        for author_name in article.authors:
            author = ET.SubElement(author_list, "Author")
            if "," in author_name:
                last_name, fore_name = [part.strip() for part in author_name.split(",", 1)]
                ET.SubElement(author, "LastName").text = last_name
                ET.SubElement(author, "ForeName").text = fore_name
            else:
                ET.SubElement(author, "CollectiveName").text = author_name
        if article.mesh_terms:
            mesh_heading_list = ET.SubElement(medline, "MeshHeadingList")
            for mesh_term in article.mesh_terms:
                heading = ET.SubElement(mesh_heading_list, "MeshHeading")
                if " / " in mesh_term:
                    descriptor, qualifier = [part.strip() for part in mesh_term.split(" / ", 1)]
                    ET.SubElement(heading, "DescriptorName").text = descriptor
                    ET.SubElement(heading, "QualifierName").text = qualifier
                else:
                    ET.SubElement(heading, "DescriptorName").text = mesh_term
        if article.keywords:
            keyword_list = ET.SubElement(medline, "KeywordList")
            for keyword in article.keywords:
                ET.SubElement(keyword_list, "Keyword").text = keyword
        pubmed_data = ET.SubElement(pubmed_article, "PubmedData")
        article_id_list = ET.SubElement(pubmed_data, "ArticleIdList")
        ET.SubElement(article_id_list, "ArticleId", {"IdType": "pubmed"}).text = article.pmid
        if article.doi:
            ET.SubElement(article_id_list, "ArticleId", {"IdType": "doi"}).text = article.doi
        return pubmed_article



def load_local_articles(filepath: str) -> List[Article]:
    path = Path(filepath)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        logger.error("Local article file not found: %s", filepath)
        raise exc
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in local article file %s: %s", filepath, exc)
        raise exc

    if isinstance(payload, dict):
        raw_articles = payload.get("articles", [])
    elif isinstance(payload, list):
        raw_articles = payload
    else:
        raise ValueError("Local article payload must be a list or a dict containing an 'articles' key")

    articles: List[Article] = []
    for raw_article in raw_articles:
        if not isinstance(raw_article, dict):
            logger.warning("Skipping non-dictionary article entry in %s", filepath)
            continue
        articles.append(
            Article(
                pmid=str(raw_article.get("pmid", "")).strip(),
                title=str(raw_article.get("title", "")),
                abstract=str(raw_article.get("abstract", "")),
                authors=[str(author) for author in raw_article.get("authors", [])],
                journal=str(raw_article.get("journal", "")),
                year=validate_year(raw_article.get("year")),
                mesh_terms=[str(term) for term in raw_article.get("mesh_terms", [])],
                keywords=[str(keyword) for keyword in raw_article.get("keywords", [])],
                doi=str(raw_article.get("doi")) if raw_article.get("doi") else None,
            )
        )
    return articles



def create_sample_articles() -> List[Article]:
    return [
        Article(
            pmid="10000001",
            title="BRCA1 loss rewires homologous recombination and replication stress signaling in triple-negative breast cancer",
            abstract=(
                "BRCA1-deficient breast cancer cells accumulated replication-associated DNA damage, activated ATR-CHK1 signaling, "
                "and showed increased sensitivity to PARP inhibition. Transcriptomic profiling linked BRCA1 loss to RAD51 depletion, "
                "genomic instability, and inflammatory cGAS-STING pathway activation in basal-like tumors."
            ),
            authors=["Smith, Elena", "Patel, Rohan", "Nguyen, Mai"],
            journal="Cancer Discovery",
            year=2023,
            mesh_terms=["BRCA1 Protein", "DNA Repair", "Breast Neoplasms / genetics"],
            keywords=["BRCA1", "homologous recombination", "PARP inhibitor", "triple-negative breast cancer"],
            doi="10.1000/cdisc.2023.001",
        ),
        Article(
            pmid="10000002",
            title="Mutant TP53 drives chromatin remodeling and immune evasion in high-grade serous ovarian cancer",
            abstract=(
                "Gain-of-function TP53 variants promoted NF-kB signaling, increased PD-L1 expression, and remodeled enhancer accessibility "
                "through interaction with the SWI/SNF complex. The resulting transcriptional program supported cytokine secretion, mesenchymal "
                "transition, and resistance to platinum chemotherapy."
            ),
            authors=["Garcia, Lucia", "Brown, Aiden"],
            journal="Nature Cancer",
            year=2024,
            mesh_terms=["Tumor Suppressor Protein p53", "Ovarian Neoplasms / pathology", "Chromatin Remodeling"],
            keywords=["TP53", "immune evasion", "PD-L1", "SWI/SNF"],
            doi="10.1000/natc.2024.117",
        ),
        Article(
            pmid="10000003",
            title="Cross-talk between PI3K-AKT-mTOR and p53 pathways shapes endocrine resistance in ER-positive breast tumors",
            abstract=(
                "Endocrine-resistant estrogen receptor positive tumors displayed coordinated activation of PI3K-AKT-mTOR signaling and attenuation "
                "of p53-dependent apoptosis. Integrated phosphoproteomics identified AKT-mediated MDM2 phosphorylation, reduced CDKN1A induction, "
                "and enhanced cyclin D1 translation as determinants of proliferative escape."
            ),
            authors=["Lee, Hannah", "Smith, Elena", "Osei, Kofi"],
            journal="Clinical Cancer Research",
            year=2022,
            mesh_terms=["Phosphatidylinositol 3-Kinases", "Breast Neoplasms / drug therapy", "Tumor Suppressor Protein p53"],
            keywords=["PI3K", "AKT", "mTOR", "p53", "endocrine resistance"],
            doi="10.1000/ccr.2022.208",
        ),
        Article(
            pmid="10000004",
            title="ATM and BRCA1 coordinate DNA damage checkpoints with interferon signaling after radiotherapy",
            abstract=(
                "Radiation exposed mammary tumor models required ATM and BRCA1 to stabilize stalled replication forks and sustain G2-M checkpoint control. "
                "Loss of either factor enhanced micronuclei formation, interferon-stimulated gene expression, and macrophage recruitment through STING-dependent cytokines."
            ),
            authors=["Ahmed, Sara", "Patel, Rohan", "Kim, Daniel"],
            journal="Molecular Cell",
            year=2021,
            mesh_terms=["ATM Serine-Threonine Kinases", "BRCA1 Protein", "Interferons / metabolism"],
            keywords=["ATM", "BRCA1", "radiotherapy", "STING", "DNA damage checkpoint"],
            doi="10.1000/molcel.2021.442",
        ),
        Article(
            pmid="10000005",
            title="Synthetic lethal targeting of PARP and ATR uncovers vulnerabilities in TP53 and BRCA1 co-altered tumors",
            abstract=(
                "Combined PARP and ATR inhibition induced replication catastrophe in organoid models harboring TP53 mutation together with BRCA1 deficiency. "
                "The combination suppressed RAD51 foci, amplified double-strand breaks, and triggered caspase-dependent apoptosis in aggressive cancer clones."
            ),
            authors=["Johnson, Priya", "Garcia, Lucia", "Chen, Wei"],
            journal="Cell Reports Medicine",
            year=2024,
            mesh_terms=["Poly(ADP-ribose) Polymerase Inhibitors", "ATR Kinase", "Synthetic Lethality"],
            keywords=["PARP", "ATR", "TP53", "BRCA1", "synthetic lethality"],
            doi="10.1000/crmed.2024.509",
        ),
    ]



def retrieve_and_clean(
    query: str,
    config: BioNOVAConfig,
    max_results: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Article]:
    retriever = PubMedRetriever(config)
    cleaner = DataCleaner()
    articles = retriever.fetch_with_progress(query, max_results=max_results, progress_callback=progress_callback)
    return cleaner.clean_articles(articles)


__all__ = [
    "Article",
    "DataCleaner",
    "MockRetriever",
    "PubMedRetriever",
    "XMLParser",
    "create_sample_articles",
    "load_local_articles",
    "retrieve_and_clean",
]
