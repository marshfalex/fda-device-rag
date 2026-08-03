from fda_device_rag.eval.furniture import is_page_furniture


def test_identical_footer_differing_only_by_page_number_is_furniture():
    a = "MAINTENANCE\nZ-800F Instructions for Use.  15 \nP/N 800F-IFU-2602, Rev. O"
    b = "MAINTENANCE\nZ-800F Instructions for Use.  21 \nP/N 800F-IFU-2602, Rev. O"

    assert is_page_furniture(a, b) is True


def test_short_leading_page_number_difference_is_furniture():
    a = "GETTING STARTED\n16  Z-800F Instructions for Use \nP/N 800F-IFU-2602, Rev. O"
    b = "GETTING STARTED\n18  Z-800F Instructions for Use \nP/N 800F-IFU-2602, Rev. O"

    assert is_page_furniture(a, b) is True


def test_distinct_citations_differing_by_a_long_digit_run_are_not_furniture():
    # Real guidance-doc citations that a naive digit-stripping rule wrongly
    # collapsed (design doc S4, round 2) -- different URLs, different topics,
    # coincidentally identical non-digit template text.
    a = ("Compliance on Off-the-Shelf Software Use in Medical Devices\n"
         "(http://www.fda.gov/MedicalDevices/DeviceRegulationandGuidance/GuidanceDocuments/ucm073778.htm).")
    b = ("Medical Device Design\n"
         "(http://www.fda.gov/MedicalDevices/DeviceRegulationandGuidance/GuidanceDocuments/ucm259748.htm).")

    assert is_page_furniture(a, b) is False


def test_distinct_spec_tables_with_long_differing_item_numbers_are_not_furniture():
    # Real needle-set spec tables that a naive digit-stripping rule wrongly
    # collapsed (design doc S4, round 2) -- different item numbers and
    # residual volumes, same table template.
    single_needle = ("Single-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                      "6 mm RMS12406 0.4 ml 20\n9 mm RMS12409 0.4 ml 20\n"
                      "12 mm RMS12412 0.4 ml 20\n14 mm RMS12414 0.4 ml 20")
    two_needle = ("Two-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                  "6 mm RMS22406 0.7 ml 10\n9 mm RMS22409 0.7 ml 10\n"
                  "12 mm RMS22412 0.7 ml 10\n14 mm RMS22414 0.7 ml 10")

    assert is_page_furniture(single_needle, two_needle) is False


def test_distinct_spec_tables_with_differing_digit_run_counts_are_not_furniture():
    # Real needle-set spec tables where one member has an extra wrapped
    # digit fragment, giving the two texts a different total digit-run count
    # (design doc S4, round 2's second false-positive class).
    single_needle = ("Single-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                      "4 mm RMS12604 0.1 ml 20\n6 mm RMS12606 0.1 ml 20\n"
                      "9 mm RMS12609 0.1 ml 20\n12 mm RMS12612 0.1 ml 20\n"
                      "14 mm RMS12614 0.1 ml 20")
    six_needle = ("Six-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                  "4 mm RMS62604 0.6 ml 10\n6 mm RMS62606 0.6 ml 10\n"
                  "9 mm RMS62609 0.6 ml 10\n12 mm RMS62612 0.6 ml 10\n"
                  "14 mm RMS62614 0.6 ml 10\n16")

    assert is_page_furniture(single_needle, six_needle) is False


def test_completely_unrelated_text_with_no_digits_is_not_furniture():
    assert is_page_furniture("WARNINGS\nDo not reuse this device.", "CAUTION\nKeep away from heat.") is False


def test_distinct_hazard_table_entries_with_matching_digit_run_counts_are_not_furniture():
    # Real hazard-analysis/FMEA table entries from data/raw/guidance_pdfs/78369.pdf
    # (document title "78369", section_name "Supply Voltage Error" vs
    # "Hazard Potential Causes") that a digit-run-only rule wrongly collapsed
    # (Task 8 real-corpus run): two entirely unrelated table rows that
    # coincidentally carry the same count of short, page-number-scale digit
    # runs (one embedded page number each: "15" vs "14"), but have
    # completely different skeletons (surrounding words). This pins the bug
    # where digit-run count/scale matching alone -- without first requiring
    # skeleton equality -- produced a false positive: verified below that
    # the pre-fix logic (digit-run comparison with no skeleton check) would
    # incorrectly call this pair furniture, while the fixed
    # is_page_furniture correctly rejects it on skeleton mismatch.
    instance_a = (
        "Supply Voltage Error\nAC supply exceeds limits \nBattery voltage exceeds limits \n"
        "Battery depleted \nVoltage conversion failed \nBattery Failure Battery voltage too low \n"
        "Battery depleted \nBattery overcharged \nLeakage Current too high Inadequate shielding \n"
        " \n 15 \nShort circuit"
    )
    instance_b = (
        "Hazard Potential Causes\nAir in Infusion Line Incorrect/incomplete priming processes \n"
        "Broken, loose, or unsealed delivery path \nThe pump is unable to release gas or air \n"
        "The pump is set up with an incompatible infusion set \nOcclusion  Delivery path obstructed, "
        "e.g., kinked tubes \nChemical precipitation inside the delivery path \n"
        "Bolus occurring after an occlusion \nUncontrolled Flow of Infusate (e.g. \n"
        "free flow) \nValves in the delivery path are broken \n"
        "The pump is positioned much higher than the infusion \nsite, causing unintentional drug flow \n"
        "The delivery path is damaged, creating a vent on the path \nthat allows unintentional gravity flow \n"
        "Retrograde Flow of Infusate (e.g. The pump is positioned much lower than the infusion \n"
        " \n 14 \nReverse Flow)"
    )

    assert is_page_furniture(instance_a, instance_b) is False


def test_identical_zero_digit_text_is_furniture():
    # Real repeating fragment from data/raw/guidance_pdfs/188844.pdf -- the
    # section "Operations" recurs 8 times with byte-identical text and no
    # digits anywhere. The old zero-digit-run guard wrongly excluded this
    # genuine furniture from ever being flagged (Task 8 real-corpus run).
    text = (
        "Operations\nRisk-Based Analysis Assurance Activities Establishing the appropriate \nrecord"
    )

    assert is_page_furniture(text, text) is True


def test_identical_skeleton_with_differing_long_digit_runs_are_not_furniture():
    # Two texts with identical skeleton (same words, digits stripped) but
    # where one digit-run exceeds MAX_FURNITURE_DIGIT_RUN_LEN (3 digits).
    # This directly tests the digit-run length guard (line 40 of furniture.py),
    # which is not exercised by any existing "not furniture" test case.
    # Both texts could be furniture (page numbers), but the embedded
    # serial numbers differ in scale (3 vs 5 digits), indicating they are
    # genuinely different content, not page-number variants.
    a = "Serial Number: SN999 verified"
    b = "Serial Number: SN99999 verified"

    # Skeletons are identical: both -> "Serial Number: SN verified"
    # But digit-runs differ in length: ["999"] vs ["99999"]
    # Since max(3, 5) > MAX_FURNITURE_DIGIT_RUN_LEN, should return False
    assert is_page_furniture(a, b) is False
