from app.stage_consensus import choose_stage_by_count_then_confidence


def test_more_vehicles_always_wins():
    stage, confidence, _ = choose_stage_by_count_then_confidence(
        {0: 3, 1: 2},
        {0: 1.8, 1: 1.98},
    )
    assert stage == 0
    assert round(confidence, 2) == 0.60


def test_equal_counts_use_average_confidence():
    stage, confidence, _ = choose_stage_by_count_then_confidence(
        {0: 2, 1: 2},
        {0: 1.40, 1: 1.64},
    )
    assert stage == 1
    assert round(confidence, 2) == 0.82


def test_exact_confidence_tie_uses_higher_stage():
    stage, confidence, _ = choose_stage_by_count_then_confidence(
        {1: 2, 2: 2},
        {1: 1.60, 2: 1.60},
    )
    assert stage == 2
    assert round(confidence, 2) == 0.80

