"""
test_pipeline.py
────────────────────────────────────────────────────────────────────────────
Phase 15 — Testing Suite for the RAG Ingestion Pipeline.

Contains 6 required test cases enforcing key behaviors as specified in
RAG_INGESTION_PIPELINE_PROMPT.md.

Run with: `pytest ingestion/tests/test_pipeline.py`
"""

import pytest

from ingestion.language_filter import filter_hindi
from ingestion.batch_processor import get_batch_ranges
from ingestion.chunker import chunk_document
from ingestion.llm_client import _TEMPERATURE


def test_devanagari_filter():
    """
    Test 1: Feed mixed Hindi+English paragraph, assert Hindi removed,
    English preserved.
    """
    mixed_text = (
        "This is an English sentence.\n\n"
        "यह हिंदी में एक वाक्य है जो देवनागरी लिपि का उपयोग करता है।\n\n"
        "This is another English sentence with some data: 12345."
    )
    
    filtered = filter_hindi(mixed_text)
    
    # English should be preserved
    assert "This is an English sentence." in filtered
    assert "This is another English sentence" in filtered
    
    # Hindi (Devanagari) should be stripped out completely
    assert "यह हिंदी में" not in filtered
    assert "देवनागरी लिपि" not in filtered


def test_batch_ranges():
    """
    Test 2: Test get_batch_ranges(40) → [(1,40)], get_batch_ranges(60) → [(1,25),(26,50),(51,60)].
    """
    assert get_batch_ranges(40, batch_size=25, threshold=50) == [(1, 40)]
    assert get_batch_ranges(60, batch_size=25, threshold=50) == [(1, 25), (26, 50), (51, 60)]


@pytest.fixture
def dummy_structured_md() -> str:
    return (
        "# Main Document Title\n\n"
        "This is the introductory text. It is quite long to ensure it exceeds "
        "the fifty character limit required by the chunker.\n\n"
        "## Section 1\n\n"
        "This is the content for Section 1. It is also purposefully made long "
        "enough to pass the fifty character minimum threshold safely.\n\n"
        "### Subsection 1.1\n\n"
        "Short.\n\n"  # < 50 chars, will be merged into parent
        "## Section 2\n\n"
        "Section 2 has sufficient text to be standalone. "
        "It also contains a markdown table.\n"
        "| Header A | Header B |\n"
        "| -------- | -------- |\n"
        "| Val 1    | Val 2    |"
    )


@pytest.fixture
def dummy_metadata_json() -> dict:
    return {
        "title": "Main Document Title",
        "total_sections": 4,
        "sections": [
            {
                "id": "section_001",
                "heading": "Main Document Title",
                "level": 1,
                "heading_path": "Main Document Title",
                "parent_id": None,
                "children": ["section_002", "section_004"],
                "approximate_page_range": [1, 2]
            },
            {
                "id": "section_002",
                "heading": "Section 1",
                "level": 2,
                "heading_path": "Main Document Title > Section 1",
                "parent_id": "section_001",
                "children": ["section_003"],
                "approximate_page_range": [2, 2]
            },
            {
                "id": "section_003",
                "heading": "Subsection 1.1",
                "level": 3,
                "heading_path": "Main Document Title > Section 1 > Subsection 1.1",
                "parent_id": "section_002",
                "children": [],
                "approximate_page_range": [2, 2]
            },
            {
                "id": "section_004",
                "heading": "Section 2",
                "level": 2,
                "heading_path": "Main Document Title > Section 2",
                "parent_id": "section_001",
                "children": [],
                "approximate_page_range": [3, 3]
            }
        ]
    }


def test_chunk_metadata_fields(dummy_structured_md, dummy_metadata_json):
    """
    Test 3: Assert all required fields present in every chunk.
    """
    chunks = chunk_document(dummy_structured_md, dummy_metadata_json)
    
    assert len(chunks) > 0, "No chunks were produced."
    
    required_fields = {
        "chunk_id", "source_file", "page_range", "heading_path",
        "heading_path_list", "section_level", "section_heading", "chunk_index",
        "content", "content_length", "token_estimate", "has_table",
        "parent_heading", "child_headings"
    }
    
    for chunk in chunks:
        missing = required_fields - set(chunk.keys())
        assert not missing, f"Chunk is missing required fields: {missing}"


def test_no_empty_chunks(dummy_structured_md, dummy_metadata_json):
    """
    Test 4: Assert no chunk has content_length < 50 characters.
    The short subsection (which just has 'Short.') should be merged into its parent.
    """
    chunks = chunk_document(dummy_structured_md, dummy_metadata_json)
    
    # Verify that the short section was either merged or padded/handled,
    # and no chunk ever hits the disk with less than 50 chars of content.
    for chunk in chunks:
        # Some chunks might just be headings with no body content, but the chunker
        # spec says: "Minimum chunk content length: 50 characters — if section content 
        # is shorter, merge it into parent section chunk".
        assert chunk["content_length"] >= 50, (
            f"Found chunk with length < 50: {chunk['chunk_id']} "
            f"(len={chunk['content_length']})\nContent: {chunk['content']}"
        )


def test_heading_path_format(dummy_structured_md, dummy_metadata_json):
    """
    Test 5: Assert all heading_path strings use ` > ` separator.
    """
    chunks = chunk_document(dummy_structured_md, dummy_metadata_json)
    
    for chunk in chunks:
        path = chunk["heading_path"]
        level = chunk["section_level"]
        
        # If level > 1, it must be a nested section and therefore contain the ' > ' separator
        if level > 1:
            assert " > " in path, f"Missing ' > ' separator in path: {path}"


def test_temperature_zero():
    """
    Test 6: Assert LLM calls use temperature parameter = 0.
    In our implementation, the unified llm_client enforces this rigidly via a constant.
    """
    assert _TEMPERATURE == 0, "LLM temperature must be explicitly set to 0 for ingestion tasks."
