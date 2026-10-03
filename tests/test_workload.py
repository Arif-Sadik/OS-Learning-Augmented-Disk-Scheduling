from src.workload import generate_workload, generate_workload_shift


def test_workload_generation_is_deterministic():
    first = generate_workload("uniform", seed=123, count=20)
    second = generate_workload("uniform", seed=123, count=20)

    assert first.requests == second.requests
    assert first.initial_head == 50


def test_workloads_stay_inside_cylinder_range():
    for workload_type in ("uniform", "localized", "directional", "bursty"):
        sample = generate_workload(workload_type, seed=42, count=100)
        assert len(sample.requests) == 100
        assert min(sample.requests) >= 0
        assert max(sample.requests) <= 199


def test_localized_workload_is_more_concentrated_than_uniform_for_same_seed():
    uniform = generate_workload("uniform", seed=44, count=80)
    localized = generate_workload("localized", seed=44, count=80)

    assert max(localized.requests) - min(localized.requests) < max(uniform.requests) - min(uniform.requests)


def test_directional_workload_has_strong_net_direction():
    sample = generate_workload("directional", seed=99, count=50)
    transitions = [right - left for left, right in zip(sample.requests, sample.requests[1:])]
    positives = sum(step > 0 for step in transitions)
    negatives = sum(step < 0 for step in transitions)

    assert max(positives, negatives) / len(transitions) > 0.65


def test_workload_shift_records_phase_boundary():
    samples = generate_workload_shift(seed=700, windows_per_phase=3, count=10)

    assert len(samples) == 6
    assert [sample.phase for sample in samples[:3]] == ["phase_1_directional"] * 3
    assert [sample.phase for sample in samples[3:]] == ["phase_2_uniform"] * 3
    assert [sample.workload_type for sample in samples] == ["directional"] * 3 + ["uniform"] * 3
