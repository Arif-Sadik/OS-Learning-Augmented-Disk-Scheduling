import pytest

from src.schedulers import best_scheduler, cscan, fcfs, scan, sstf


def test_fcfs_services_arrival_order_and_total_movement():
    result = fcfs([82, 170, 43, 140, 24, 16, 190], initial_head=50)

    assert result.service_order == [82, 170, 43, 140, 24, 16, 190]
    assert result.movement_path == [50, 82, 170, 43, 140, 24, 16, 190]
    assert result.total_movement == 642
    assert result.average_movement == pytest.approx(642 / 7)


def test_sstf_chooses_nearest_request_with_lower_tie_break():
    result = sstf([60, 40, 70], initial_head=50)

    assert result.service_order == [40, 60, 70]
    assert result.total_movement == 40


def test_sstf_classic_mixed_workload():
    result = sstf([82, 170, 43, 140, 24, 16, 190], initial_head=50)

    assert result.service_order == [43, 24, 16, 82, 140, 170, 190]
    assert result.movement_path == [50, 43, 24, 16, 82, 140, 170, 190]
    assert result.total_movement == 208


def test_scan_upward_reaches_endpoint_before_reversing():
    result = scan([82, 170, 43, 140, 24, 16, 190], initial_head=50, direction="up")

    assert result.service_order == [82, 140, 170, 190, 43, 24, 16]
    assert result.movement_path == [50, 82, 140, 170, 190, 199, 43, 24, 16]
    assert result.total_movement == 332


def test_scan_downward_reaches_endpoint_before_reversing():
    result = scan([82, 170, 43, 140, 24, 16, 190], initial_head=50, direction="down")

    assert result.service_order == [43, 24, 16, 82, 140, 170, 190]
    assert result.movement_path == [50, 43, 24, 16, 0, 82, 140, 170, 190]
    assert result.total_movement == 240


def test_cscan_upward_counts_endpoint_and_wrap_distance():
    result = cscan([82, 170, 43, 140, 24, 16, 190], initial_head=50, direction="up")

    assert result.service_order == [82, 140, 170, 190, 16, 24, 43]
    assert result.movement_path == [50, 82, 140, 170, 190, 199, 0, 16, 24, 43]
    assert result.total_movement == 391


def test_cscan_downward_counts_endpoint_and_wrap_distance():
    result = cscan([82, 170, 43, 140, 24, 16, 190], initial_head=50, direction="down")

    assert result.service_order == [43, 24, 16, 190, 170, 140, 82]
    assert result.movement_path == [50, 43, 24, 16, 0, 199, 190, 170, 140, 82]
    assert result.total_movement == 366


def test_requests_equal_to_head_and_duplicates_are_serviced():
    requests = [50, 40, 50, 60, 40]

    assert fcfs(requests, 50).service_order == requests
    assert sstf(requests, 50).service_order == [50, 50, 40, 40, 60]
    assert scan(requests, 50, direction="up").service_order == [50, 50, 60, 40, 40]
    assert cscan(requests, 50, direction="up").service_order == [50, 50, 60, 40, 40]


def test_empty_request_list_has_zero_movement():
    for scheduler in (fcfs, sstf, scan, cscan):
        result = scheduler([], initial_head=50)
        assert result.service_order == []
        assert result.movement_path == [50]
        assert result.total_movement == 0
        assert result.average_movement == 0.0


def test_single_request_and_boundaries():
    assert fcfs([0], 50).total_movement == 50
    assert sstf([199], 50).total_movement == 149
    assert scan([0, 199], 50, direction="up").movement_path == [50, 199, 0]
    assert cscan([0, 199], 50, direction="up").movement_path == [50, 199, 0]


def test_ascending_and_descending_inputs():
    ascending = [60, 70, 80]
    descending = [80, 70, 60]

    assert fcfs(ascending, 50).total_movement == 30
    assert fcfs(descending, 50).total_movement == 50
    assert sstf(descending, 50).service_order == [60, 70, 80]


def test_invalid_cylinder_is_rejected():
    with pytest.raises(ValueError):
        fcfs([-1], 50)
    with pytest.raises(ValueError):
        scan([200], 50)
    with pytest.raises(ValueError):
        cscan([10], 50, direction="sideways")


def test_best_scheduler_uses_deterministic_tie_order():
    results = {
        "FCFS": fcfs([50], 50),
        "SSTF": sstf([50], 50),
        "SCAN": scan([50], 50),
        "C-SCAN": cscan([50], 50),
    }

    assert best_scheduler(results) == "FCFS"
