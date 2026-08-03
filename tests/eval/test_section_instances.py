from fda_device_rag.eval.section_instances import build_section_instances
from fda_device_rag.models import Chunk, ChunkMetadata


def _chunk(id_, doc, section, text):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(
            source_type="guidance_pdf",
            source_url="https://example.com/doc.pdf",
            document_title=doc,
            section_name=section,
            retrieved_date="2026-08-03",
        ),
    )


def test_groups_consecutive_same_section_chunks_into_one_instance():
    chunks = [
        _chunk("doc-0", "doc", "WARNINGS", "WARNINGS\nFirst piece."),
        _chunk("doc-1", "doc", "WARNINGS", "Second piece."),
    ]

    instances = build_section_instances(chunks)

    assert len(instances) == 1
    assert instances[0].document_title == "doc"
    assert instances[0].section_name == "WARNINGS"
    assert instances[0].instance_ordinal == 1
    assert instances[0].chunk_ids == ["doc-0", "doc-1"]
    assert instances[0].text == "WARNINGS\nFirst piece.\nSecond piece."


def test_non_adjacent_repeat_of_same_section_name_is_a_separate_instance():
    chunks = [
        _chunk("doc-0", "doc", "GETTING STARTED", "GETTING STARTED\nPage 1 content."),
        _chunk("doc-1", "doc", "MAINTENANCE", "MAINTENANCE\nMaintenance content."),
        _chunk("doc-2", "doc", "GETTING STARTED", "GETTING STARTED\nPage 2 content."),
    ]

    instances = build_section_instances(chunks)

    assert len(instances) == 3
    getting_started = [i for i in instances if i.section_name == "GETTING STARTED"]
    assert [i.instance_ordinal for i in getting_started] == [1, 2]
    assert [i.chunk_ids for i in getting_started] == [["doc-0"], ["doc-2"]]


def test_instance_ordinal_counter_is_scoped_per_document():
    chunks = [
        _chunk("docA-0", "docA", "WARNINGS", "WARNINGS\nContent A."),
        _chunk("docB-0", "docB", "WARNINGS", "WARNINGS\nContent B."),
    ]

    instances = build_section_instances(chunks)

    assert [i.instance_ordinal for i in instances] == [1, 1]
