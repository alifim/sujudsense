"""
tests/test_kb_audit.py

Knowledge Base Ingestion Audit Suite.

Purpose: Determine whether retrieval failures are caused by:
  1. MISSING CONTENT (term not in source at all) -> fix ingestion
  2. RETRIEVAL FAILURE (term exists but search doesn't surface it) -> fix retrieval

Run: pytest tests/test_kb_audit.py -v
"""

import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
import re


# =============================================================================
# CONFIGURATION
# =============================================================================

REQUIRED_TERMS = [
    "knee", "sujud", "back", "ruku", "shoulder", "wrist", "prostration",
    "elbow", "ankle", "neck", "prayer", "spine", "hip", "chair", "leg",
]

POSTURE_TERMS = ["qiyam", "ruku", "sujud", "julus", "prostration", "bowing", "standing", "sitting"]

BODY_PART_TERMS = [
    "knee", "back", "shoulder", "wrist", "elbow", "ankle", "neck",
    "spine", "hip", "leg", "foot", "feet", "hand", "hands", "arm", "arms", "head", "forehead",
]

SOURCE_CONFIG = {
    "txt_biomechanics": {
        "path": "sources/v1_synthetic/biomechanics.txt",
        "label": "TXT: Biomechanics",
        "domain": "biomechanics",
        "type": "synthetic",
    },
    "txt_fiqh": {
        "path": "sources/v1_synthetic/fiqh.txt",
        "label": "TXT: Fiqh",
        "domain": "fiqh",
        "type": "synthetic",
    },
    "pdf_biomechanics": {
        "path": "sources/v2_raw/nazish_kalra_2018_salat_biomechanics.pdf",
        "label": "PDF: Biomechanics (Nazish & Kalra 2018)",
        "domain": "biomechanics",
        "type": "verified_peer_reviewed",
    },
    "pdf_fiqh": {
        "path": "sources/v2_raw/gaiae_2015_salatul_mareed.pdf",
        "label": "PDF: Fiqh (GAIAE 2015)",
        "domain": "fiqh",
        "type": "verified_official_fatwa",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SourceAudit:
    key: str
    label: str
    domain: str
    source_type: str
    path: str
    exists: bool
    chars: int = 0
    words: int = 0
    term_counts: Dict[str, int] = field(default_factory=dict)
    missing_terms: List[str] = field(default_factory=list)
    posture_counts: Dict[str, int] = field(default_factory=dict)
    body_part_counts: Dict[str, int] = field(default_factory=dict)
    found_body_parts: List[str] = field(default_factory=list)
    noise_score: float = 0.0
    structure_score: int = 0

    def __post_init__(self):
        if self.term_counts is None:
            self.term_counts = {}
        if self.missing_terms is None:
            self.missing_terms = []
        if self.posture_counts is None:
            self.posture_counts = {}
        if self.body_part_counts is None:
            self.body_part_counts = {}
        if self.found_body_parts is None:
            self.found_body_parts = []


# =============================================================================
# LOADER FUNCTIONS
# =============================================================================

def load_txt(path: str) -> str:
    """Load raw text from TXT file."""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def load_pdf_text(path: str) -> str:
    """Extract raw text from PDF using PDFPlumber."""
    try:
        from langchain_community.document_loaders import PDFPlumberLoader
    except ImportError:
        pytest.skip("PDFPlumberLoader not available (langchain_community)")

    p = Path(path)
    if not p.exists():
        return ""

    loader = PDFPlumberLoader(str(p))
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])


def load_source(config: dict) -> str:
    """Dispatch to appropriate loader based on file extension."""
    path = config["path"]
    if path.endswith(".pdf"):
        return load_pdf_text(path)
    else:
        return load_txt(path)


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def audit_source(key: str, config: dict) -> SourceAudit:
    """Perform full audit on a single source."""
    path = config["path"]
    p = Path(path)

    audit = SourceAudit(
        key=key,
        label=config["label"],
        domain=config["domain"],
        source_type=config["type"],
        path=path,
        exists=p.exists(),
    )

    if not audit.exists:
        return audit

    text = load_source(config)
    text_lower = text.lower()

    audit.chars = len(text)
    audit.words = len(text.split())

    # Required terms
    for term in REQUIRED_TERMS:
        count = text_lower.count(term)
        audit.term_counts[term] = count
        if count == 0:
            audit.missing_terms.append(term)

    # Posture terms
    for posture in POSTURE_TERMS:
        audit.posture_counts[posture] = text_lower.count(posture)

    # Body parts
    for part in BODY_PART_TERMS:
        count = text_lower.count(part)
        audit.body_part_counts[part] = count
        if count > 0:
            audit.found_body_parts.append(part)

    # Noise detection (PDF artifacts)
    noise_patterns = {
        "et_al": text_lower.count("et al"),
        "reference_list": len(re.findall(r"^\s*\d+\.\s+\w+", text, re.MULTILINE)),
        "page_headers": text_lower.count("nabeela nazish") + text_lower.count("journal of"),
        "footer_artifacts": text_lower.count("international journal") + text_lower.count("vol.") + text_lower.count("issue"),
        "citation_numbers": len(re.findall(r"\[\d+\]", text)),
    }
    audit.noise_score = sum(noise_patterns.values())

    # Structure detection
    audit.structure_score = (
        text.count("#") + 
        text.count("##") + 
        text.count("Biomechanics") + 
        text.count("Fiqh") +
        text.count("Rulings") +
        text.count("Section")
    )

    return audit


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def all_audits() -> Dict[str, SourceAudit]:
    """Run audit on all configured sources once per session."""
    results = {}
    for key, config in SOURCE_CONFIG.items():
        results[key] = audit_source(key, config)
    return results


# =============================================================================
# TESTS: SOURCE EXISTENCE
# =============================================================================

class TestSourceExistence:
    """Verify all expected source files are present."""

    @pytest.mark.parametrize(
        "key,config", 
        [(k, v) for k, v in SOURCE_CONFIG.items()], 
        ids=[k for k in SOURCE_CONFIG.keys()]
    )
    def test_source_exists(self, key, config):
        path = Path(config["path"])
        assert path.exists(), f"Source not found: {config['path']}"
        print(f"\n  {config['label']}: {path.stat().st_size:,} bytes")


# =============================================================================
# TESTS: TERM COVERAGE
# =============================================================================

class TestTermCoverage:
    """Check which required terms exist in each source."""

    @pytest.mark.parametrize("term", REQUIRED_TERMS, ids=lambda t: t)
    def test_term_across_all_sources(self, term, all_audits):
        """A term must exist in at least one source (any source)."""
        total = sum(audit.term_counts.get(term, 0) for audit in all_audits.values() if audit.exists)
        assert total > 0, f"Term '{term}' is completely missing from ALL sources"

    @pytest.mark.parametrize("source_key", [k for k in SOURCE_CONFIG.keys()], ids=lambda k: k)
    def test_source_missing_terms(self, source_key, all_audits):
        """Log which terms each individual source is missing."""
        audit = all_audits[source_key]
        if not audit.exists:
            pytest.skip(f"Source {source_key} does not exist")

        # This test always passes; it just logs missing terms for analysis
        if audit.missing_terms:
            print(f"\n  {audit.label} missing: {audit.missing_terms}")
        else:
            print(f"\n  {audit.label}: ALL terms present ✓")


# =============================================================================
# TESTS: BODY PART COVERAGE
# =============================================================================

class TestBodyPartCoverage:
    """Analyze which body parts are covered across sources."""

    def test_body_part_coverage_summary(self, all_audits):
        """Print coverage matrix for all body parts across all sources."""
        print("\n")
        print("=" * 70)
        print("BODY PART COVERAGE MATRIX")
        print("=" * 70)
        print(f"{'Body Part':<15}", end="")
        for key in SOURCE_CONFIG.keys():
            print(f"{key:<20}", end="")
        print()
        print("-" * 70)

        for part in BODY_PART_TERMS:
            print(f"{part:<15}", end="")
            for key in SOURCE_CONFIG.keys():
                audit = all_audits[key]
                if not audit.exists:
                    print(f"{'N/A':<20}", end="")
                else:
                    count = audit.body_part_counts.get(part, 0)
                    marker = "✓" if count > 0 else "✗"
                    print(f"{marker} ({count:3d}){'':<12}", end="")
            print()

        # Always pass — this is informational
        assert True

    @pytest.mark.parametrize("part", ["shoulder", "wrist", "elbow", "ankle", "leg", "hip"], ids=lambda p: p)
    def test_critical_body_part_present(self, part, all_audits):
        """Critical body parts (the ones your tests fail on) must exist somewhere."""
        total = sum(audit.body_part_counts.get(part, 0) for audit in all_audits.values() if audit.exists)
        assert total > 0, f"Critical body part '{part}' missing from ALL sources — this will cause retrieval failures"


# =============================================================================
# TESTS: CROSS-SOURCE COMPARISON
# =============================================================================

class TestCrossSourceComparison:
    """Compare TXT vs PDF sources to decide which to use."""

    def test_pdf_vs_txt_coverage(self, all_audits):
        """PDF sources should cover at least as many terms as TXT sources."""
        txt_terms = set()
        pdf_terms = set()

        for key, audit in all_audits.items():
            if not audit.exists:
                continue
            found = {t for t, c in audit.term_counts.items() if c > 0}
            if audit.source_type == "synthetic":
                txt_terms.update(found)
            else:
                pdf_terms.update(found)

        txt_only = txt_terms - pdf_terms
        pdf_only = pdf_terms - txt_terms

        print("\n")
        print("=" * 50)
        print("CROSS-SOURCE TERM COMPARISON")
        print("=" * 50)
        print(f"Terms only in TXT (lost in PDF):  {sorted(txt_only) or 'None'}")
        print(f"Terms only in PDF (gained):       {sorted(pdf_only) or 'None'}")
        print(f"Shared terms:                     {sorted(txt_terms & pdf_terms)}")

        # PDF should not lose critical terms that TXT had
        critical_lost = txt_only & {"knee", "sujud", "back", "ruku", "prayer", "chair"}
        assert not critical_lost, f"PDF is missing terms that TXT had: {critical_lost}"

    def test_pdf_noise_level(self, all_audits):
        """PDF sources should not be dominated by noise."""
        for key, audit in all_audits.items():
            if not audit.exists or audit.source_type == "synthetic":
                continue

            # Noise ratio: noise markers per 1000 words
            noise_ratio = (audit.noise_score / max(audit.words, 1)) * 1000
            print(f"\n  {audit.label}: noise_ratio={noise_ratio:.1f} per 1000 words")

            # Warn if noise is high, but don't fail — this is diagnostic
            if noise_ratio > 10:
                print(f"  ⚠️  HIGH NOISE: PDF may need cleaning before ingestion")


# =============================================================================
# TESTS: INGESTION READINESS
# =============================================================================

class TestIngestionReadiness:
    """Determine if current sources are ready for embedding, or need work."""

    def test_overall_kb_health(self, all_audits):
        """Final verdict: can we embed these sources as-is, or fix first?"""
        print("\n")
        print("=" * 70)
        print("KB INGESTION READINESS REPORT")
        print("=" * 70)

        total_chars = 0
        total_noise = 0

        for key, audit in all_audits.items():
            if not audit.exists:
                print(f"\n  {audit.label}: ❌ MISSING")
                continue

            total_chars += audit.chars
            total_noise += audit.noise_score

            health = "✅ HEALTHY" if not audit.missing_terms else "⚠️  GAPS"
            if audit.noise_score > 20 and audit.source_type != "synthetic":
                health = "🔧 NEEDS CLEANING"

            print(f"\n  {audit.label}")
            print(f"    Size: {audit.chars:,} chars, {audit.words:,} words")
            print(f"    Missing terms: {audit.missing_terms or 'None'}")
            print(f"    Body parts: {len(audit.found_body_parts)}/{len(BODY_PART_TERMS)}")
            print(f"    Noise score: {audit.noise_score}")
            print(f"    Structure score: {audit.structure_score}")
            print(f"    Verdict: {health}")

        # Calculate terms missing from the COMBINED corpus
        all_present_terms = set()
        for audit in all_audits.values():
            if not audit.exists:
                continue
            for term, count in audit.term_counts.items():
                if count > 0:
                    all_present_terms.add(term)
        
        all_missing = set(REQUIRED_TERMS) - all_present_terms

        print(f"\n  OVERALL:")
        print(f"    Total content: {total_chars:,} chars")
        print(f"    Total noise markers: {total_noise}")
        print(f"    Terms present somewhere: {len(all_present_terms)}/{len(REQUIRED_TERMS)}")
        print(f"    Universally missing terms: {sorted(all_missing) or 'None'}")

        if all_missing:
            print(f"\n  🔴 ACTION NEEDED: Add content for {sorted(all_missing)}")
            print(f"     These terms WILL cause retrieval failures regardless of search method.")
        else:
            print(f"\n  🟢 ALL TERMS COVERED — ready for retrieval experiments")

        assert True
