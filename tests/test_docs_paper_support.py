from __future__ import annotations

from pathlib import Path


DOCS = (
    "docs/CLAIMS_AND_EVIDENCE.md",
    "docs/PUBLICATION_NOTES.md",
    "docs/MANUSCRIPT_OUTLINE.md",
    "docs/VERIFIED_REFERENCE_POOL.md",
)


def test_readme_links_new_paper_support_docs() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    for rel in DOCS:
        assert rel in text



def test_new_docs_exist() -> None:
    for rel in DOCS:
        assert Path(rel).exists(), rel



def test_publication_notes_stay_venue_agnostic() -> None:
    text = Path("docs/PUBLICATION_NOTES.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "reproducible logic-aware" in lowered
    assert ("sa" + "iv") not in lowered
    assert ("nf" + "m") not in lowered
    assert "monitoring-only" in lowered



def test_manuscript_outline_stays_generic() -> None:
    text = Path("docs/MANUSCRIPT_OUTLINE.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "candidate manuscript claim" in lowered
    assert "repo-to-manuscript mapping" in lowered
    assert ("sa" + "iv") not in lowered
    assert ("nf" + "m") not in lowered



def test_dataset_recommendations_capture_cmapss_access_ambiguity() -> None:
    text = Path("docs/DATASET_RECOMMENDATIONS.md").read_text(encoding="utf-8")
    assert "CMAPSSData.zip" in text
    assert "availability is unstable enough that the paper should not" in text
    assert "currently unavailable for direct download" not in text
    assert "federal catalog entry" in text



def test_reference_pool_contains_checked_core_set() -> None:
    text = Path("docs/VERIFIED_REFERENCE_POOL.md").read_text(encoding="utf-8")
    assert "## A. Checked core references" in text
    assert "## B. Candidate expansion references" in text
    assert "NASA Thermal Protection Systems overview" in text



def test_claims_doc_reinforces_monitoring_only_distinction() -> None:
    text = Path("docs/CLAIMS_AND_EVIDENCE.md").read_text(encoding="utf-8")
    assert "monitoring result, not a full spatially constrained training benchmark" in text
    assert "Avoid these formulations" in text
