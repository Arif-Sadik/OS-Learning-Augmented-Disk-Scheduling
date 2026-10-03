"""Classical disk scheduling algorithms for a simulated cylinder range.

Convention used in this project:

* Disk cylinders are inclusive, normally 0 through 199.
* Total head movement is the sum of absolute differences along the recorded
  movement path.
* SCAN is implemented as true elevator SCAN, not LOOK. It moves to the disk
  endpoint in the selected direction before reversing.
* C-SCAN is implemented as true circular SCAN, not C-LOOK. It moves to the
  endpoint, wraps to the opposite endpoint, and counts the wrap distance.
* Requests equal to the initial head are serviced first at zero movement.
* SSTF breaks equal-distance ties by choosing the lower cylinder first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


Direction = Literal["up", "down"]


@dataclass(frozen=True)
class ScheduleResult:
    """Result from one scheduler execution."""

    scheduler: str
    initial_head: int
    service_order: list[int]
    movement_path: list[int]
    total_movement: int
    average_movement: float


def _validate_inputs(
    requests: Iterable[int],
    initial_head: int,
    min_cylinder: int,
    max_cylinder: int,
) -> list[int]:
    if min_cylinder >= max_cylinder:
        raise ValueError("min_cylinder must be smaller than max_cylinder")
    if not min_cylinder <= initial_head <= max_cylinder:
        raise ValueError("initial_head is outside the disk cylinder range")

    request_list = list(requests)
    for request in request_list:
        if not min_cylinder <= request <= max_cylinder:
            raise ValueError(f"request {request} is outside the disk cylinder range")
    return request_list


def _movement(path: list[int]) -> int:
    return sum(abs(right - left) for left, right in zip(path, path[1:]))


def _result(name: str, initial_head: int, service_order: list[int], path: list[int]) -> ScheduleResult:
    total = _movement(path)
    average = total / len(service_order) if service_order else 0.0
    return ScheduleResult(name, initial_head, service_order, path, total, average)


def _append_path(path: list[int], cylinder: int) -> None:
    if not path or path[-1] != cylinder:
        path.append(cylinder)


def fcfs(
    requests: Iterable[int],
    initial_head: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> ScheduleResult:
    """First-Come, First-Served: service requests in arrival order."""

    request_list = _validate_inputs(requests, initial_head, min_cylinder, max_cylinder)
    path = [initial_head]
    for request in request_list:
        _append_path(path, request)
    return _result("FCFS", initial_head, request_list, path)


def sstf(
    requests: Iterable[int],
    initial_head: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> ScheduleResult:
    """Shortest Seek Time First with deterministic lower-cylinder tie-breaking."""

    pending = _validate_inputs(requests, initial_head, min_cylinder, max_cylinder)
    path = [initial_head]
    order: list[int] = []
    current = initial_head

    while pending:
        index, request = min(
            enumerate(pending),
            key=lambda item: (abs(item[1] - current), item[1], item[0]),
        )
        order.append(request)
        _append_path(path, request)
        current = request
        pending.pop(index)

    return _result("SSTF", initial_head, order, path)


def scan(
    requests: Iterable[int],
    initial_head: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
    direction: Direction = "up",
) -> ScheduleResult:
    """True SCAN scheduler.

    In the upward direction the head services requests at the current cylinder,
    then cylinders above the head in ascending order, moves to ``max_cylinder``,
    reverses, and services lower requests in descending order. The downward
    direction is symmetric and moves to ``min_cylinder`` before reversing.
    Endpoint movement is included even when the endpoint itself is not requested.
    """

    request_list = _validate_inputs(requests, initial_head, min_cylinder, max_cylinder)
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    if not request_list:
        return _result("SCAN", initial_head, [], [initial_head])

    equal = sorted([request for request in request_list if request == initial_head])
    below = sorted([request for request in request_list if request < initial_head])
    above = sorted([request for request in request_list if request > initial_head])

    path = [initial_head]
    order: list[int] = []

    for request in equal:
        order.append(request)
        _append_path(path, request)

    if direction == "up":
        for request in above:
            order.append(request)
            _append_path(path, request)
        _append_path(path, max_cylinder)
        for request in reversed(below):
            order.append(request)
            _append_path(path, request)
    else:
        for request in reversed(below):
            order.append(request)
            _append_path(path, request)
        _append_path(path, min_cylinder)
        for request in above:
            order.append(request)
            _append_path(path, request)

    return _result("SCAN", initial_head, order, path)


def cscan(
    requests: Iterable[int],
    initial_head: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
    direction: Direction = "up",
) -> ScheduleResult:
    """True C-SCAN scheduler with counted circular wrap-around movement."""

    request_list = _validate_inputs(requests, initial_head, min_cylinder, max_cylinder)
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    if not request_list:
        return _result("C-SCAN", initial_head, [], [initial_head])

    equal = sorted([request for request in request_list if request == initial_head])
    below = sorted([request for request in request_list if request < initial_head])
    above = sorted([request for request in request_list if request > initial_head])

    path = [initial_head]
    order: list[int] = []

    for request in equal:
        order.append(request)
        _append_path(path, request)

    if direction == "up":
        for request in above:
            order.append(request)
            _append_path(path, request)
        _append_path(path, max_cylinder)
        if below:
            _append_path(path, min_cylinder)
            for request in below:
                order.append(request)
                _append_path(path, request)
    else:
        for request in reversed(below):
            order.append(request)
            _append_path(path, request)
        _append_path(path, min_cylinder)
        if above:
            _append_path(path, max_cylinder)
            for request in reversed(above):
                order.append(request)
                _append_path(path, request)

    return _result("C-SCAN", initial_head, order, path)


SCHEDULERS = {
    "FCFS": fcfs,
    "SSTF": sstf,
    "SCAN": scan,
    "C-SCAN": cscan,
}


def run_all_schedulers(
    requests: Iterable[int],
    initial_head: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
    direction: Direction = "up",
) -> dict[str, ScheduleResult]:
    """Run the four schedulers on the same workload instance."""

    request_list = list(requests)
    return {
        "FCFS": fcfs(request_list, initial_head, min_cylinder, max_cylinder),
        "SSTF": sstf(request_list, initial_head, min_cylinder, max_cylinder),
        "SCAN": scan(request_list, initial_head, min_cylinder, max_cylinder, direction),
        "C-SCAN": cscan(request_list, initial_head, min_cylinder, max_cylinder, direction),
    }


def best_scheduler(results: dict[str, ScheduleResult]) -> str:
    """Return the minimum-movement scheduler with deterministic tie-breaking."""

    tie_order = {"FCFS": 0, "SSTF": 1, "SCAN": 2, "C-SCAN": 3}
    return min(results, key=lambda name: (results[name].total_movement, tie_order[name]))
