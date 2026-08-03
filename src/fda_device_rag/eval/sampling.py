import random

from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk

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
