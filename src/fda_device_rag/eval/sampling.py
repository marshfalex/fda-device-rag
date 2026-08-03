import random

from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.eval.section_instances import build_section_instances
from fda_device_rag.eval.furniture import is_page_furniture

RECALL_FLOOR_CHARS = 100
EVENT_FLOOR_CHARS = 100
PDF_SECTION_FLOOR_CHARS = 72

RECALL_CATEGORIES = [
    "Process control",
    "Device Design",
    "Under Investigation by firm",
    "Other",
    "Nonconforming Material/Component",
    "Employee error",
    "Software design",
    "Material/Component Contamination",
    "Error in labeling",
    "Packaging process control",
    "Use error",
    "Radiation Control for Health and Safety Act",
    "Labeling design",
]

EVENT_CATEGORIES = [
    "Patient Device Interaction Problem",
    "Adverse Event Without Identified Device or Use Problem",
    "Insufficient Device Problem Information",
    "Defective Device",
    "Pumping Stopped",
    "Mechanical Problem",
    "Defective Component",
    "No Apparent Adverse Event",
    "Material Integrity Problem",
    "Failure to Power Up",
    "Device Alarm System",
    "Patient-Device Incompatibility",
    "Device Dislodged or Dislocated",
]


class EmptyCandidatePoolError(Exception):
    """Raised when a category or document has zero eligible candidates after
    the floor filter -- a real data problem that must surface immediately,
    never silently skip a benchmark slot."""


class SectionDiversityError(Exception):
    """Raised when two sampled PDF candidates within the same document
    share a section_name -- an invariant that should already hold given
    without-replacement sampling over distinct section instances, but is
    verified explicitly per design doc section 8 step 2 rather than
    assumed."""


def check_section_diversity(candidates: list[dict]) -> None:
    """Raises SectionDiversityError if any two candidates sharing the same
    document_title also share the same section_name. `candidates` is the
    list of PDF-candidate dicts (each with document_title/section_name
    keys) produced by sample_pdf_section_candidates for one or more
    documents."""
    seen = {}
    for c in candidates:
        key = (c["document_title"], c["section_name"])
        if key in seen:
            raise SectionDiversityError(
                f"document {c['document_title']!r} has two candidates sharing "
                f"section_name {c['section_name']!r} -- section-diversity "
                f"invariant violated"
            )
        seen[key] = c


def sample_recall_candidates(records, retrieved_date, base_seed, categories=RECALL_CATEGORIES, floor=RECALL_FLOOR_CHARS):
    """One candidate per category: filter records in `records` matching
    `category` (exact match on root_cause_description) to those whose built
    chunk text meets `floor`, sort by stable record id, then draw with a
    seed derived deterministically from (base_seed, category) so each
    category's draw is reproducible and independent of the others."""
    results = []
    for category in categories:
        pool = []
        for record in records:
            if record.get("root_cause_description") != category:
                continue
            chunk = recall_to_chunk(record, retrieved_date=retrieved_date)
            if chunk is None or len(chunk.text) < floor:
                continue
            pool.append(record)
        pool.sort(key=lambda r: str(r.get("product_res_number", "")))

        if not pool:
            raise EmptyCandidatePoolError(f"recall category {category!r} has zero eligible candidates")

        rng = random.Random(f"{base_seed}-recall-{category}")
        chosen = rng.choice(pool)
        results.append({
            "source_type": "recall",
            "category": category,
            "record_id": str(chosen.get("product_res_number", "")),
            "record": chosen,
        })
    return results


def sample_event_candidates(records, retrieved_date, base_seed, categories=EVENT_CATEGORIES, floor=EVENT_FLOOR_CHARS):
    """Same pipeline as sample_recall_candidates, but category membership is
    list-containment on product_problems (an event can carry several problem
    categories), not equality -- so the same record may legitimately be
    eligible for, and drawn into, more than one category's pool."""
    results = []
    for category in categories:
        pool = []
        for record in records:
            if category not in (record.get("product_problems") or []):
                continue
            chunk = event_to_chunk(record, retrieved_date=retrieved_date)
            if chunk is None or len(chunk.text) < floor:
                continue
            pool.append(record)
        pool.sort(key=lambda r: str(r.get("report_number", "")))

        if not pool:
            raise EmptyCandidatePoolError(f"MAUDE category {category!r} has zero eligible candidates")

        rng = random.Random(f"{base_seed}-maude-{category}")
        chosen = rng.choice(pool)
        results.append({
            "source_type": "maude",
            "category": category,
            "record_id": str(chosen.get("report_number", "")),
            "record": chosen,
        })
    return results


def _flag_furniture(eligible):
    """Returns {index_of_furniture_instance: index_of_first_instance_it_duplicates}
    for every instance in `eligible` that is a page-furniture duplicate of an
    earlier instance in the same list. Computed pool-wide (all pairs against
    instances kept so far), matching the design doc's full-pool validation
    methodology -- not scoped to what a live draw has selected so far."""
    furniture_of = {}
    kept = []
    for idx, instance in enumerate(eligible):
        match = next(
            (
                k for k in kept
                if is_page_furniture(
                    eligible[k].text, eligible[k].section_name, instance.text, instance.section_name
                )
            ),
            None,
        )
        if match is not None:
            furniture_of[idx] = match
        else:
            kept.append(idx)
    return furniture_of


def sample_pdf_section_candidates(document_title, chunks, quota, base_seed, floor=PDF_SECTION_FLOOR_CHARS):
    """Selects `quota` distinct, non-furniture section instances from this
    document's chunks. Furniture exclusion happens before the random draw
    (pool-wide, per _flag_furniture), then `quota` instances are drawn
    without replacement from what remains."""
    instances = build_section_instances(chunks)
    eligible = [i for i in instances if len(i.text) >= floor]

    furniture_of = _flag_furniture(eligible)
    skip_log = [
        (
            f"{document_title}:{eligible[idx].section_name}#{eligible[idx].instance_ordinal}",
            f"{document_title}:{eligible[match].section_name}#{eligible[match].instance_ordinal}",
        )
        for idx, match in furniture_of.items()
    ]

    non_furniture = [inst for i, inst in enumerate(eligible) if i not in furniture_of]

    if len(non_furniture) < quota:
        raise EmptyCandidatePoolError(
            f"document {document_title!r} has only {len(non_furniture)} non-furniture "
            f"eligible section instances, needed {quota}"
        )

    rng = random.Random(f"{base_seed}-pdf-{document_title}")
    selected = rng.sample(non_furniture, quota)
    return selected, skip_log
