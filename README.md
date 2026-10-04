# Learning-Augmented Disk Scheduling Under Changing Workload Patterns

This repository contains a reproducible Operating Systems simulation project. It compares four classical disk scheduling algorithms with a lightweight learned selector that chooses one scheduler per workload window.

The project is a simulation. It does not modify a kernel, control physical disk hardware, or measure real HDD latency. The main metric is simulated disk-head movement in cylinders.

## Algorithms

- FCFS: services requests in arrival order.
- SSTF: services the closest pending request; equal-distance ties choose the lower cylinder first.
- SCAN: true elevator SCAN, not LOOK.
- C-SCAN: true circular SCAN, not C-LOOK.

## Learned Selector

The learned selector uses `sklearn.tree.DecisionTreeClassifier`. For each training workload window, the experiment:

1. extracts workload features,
2. runs FCFS, SSTF, SCAN, and C-SCAN on the same requests,
3. labels the window with the scheduler that gives the lowest total movement,
4. trains a decision tree to predict that label from workload features.

Scheduler movement values are not used as prediction features.

## Project Structure

```text
src/
  schedulers.py       Classical scheduler implementations
  workload.py         Synthetic workload generators
  features.py         Workload feature extraction
  model.py            Decision tree training and prediction
  experiment.py       Reproducible experiment pipeline
  plots.py            Figure generation from CSV files
  write_paper.py      LaTeX paper generator from result CSVs
tests/
  test_schedulers.py
  test_workload.py
  test_features.py
results/
  raw_results.csv
  summary_results.csv
  model_metrics.csv
  workload_shift_results.csv
  training_dataset.csv
  confusion_matrix.csv
  experiment_parameters.json
figures/
  classical_scheduler_comparison.png
  learned_selector_comparison.png
  workload_shift_behavior.png
  decision_tree_confusion_matrix.png
run_all.ps1
requirements.txt
```

## Windows Setup

From PowerShell in the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To run only the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

To reproduce the full project:

```powershell
.\run_all.ps1
```

The script installs Python dependencies into `.venv`, runs tests, regenerates CSV results, regenerates figures, and regenerates the LaTeX paper source.

## Experiment Parameters

- Cylinder range: 0 to 199, inclusive
- Initial head position: 50
- Requests per workload window: 40
- Workload types: uniform, localized, directional, bursty
- Training windows: 640 total, 160 per workload type
- Evaluation windows: 200 total, 50 per workload type
- Workload shift: 12 directional windows followed by 12 uniform windows
- SCAN/C-SCAN direction: upward
- Decision tree: `max_depth=4`, `min_samples_leaf=8`, `random_state=307`
- Train/test split: 75 percent / 25 percent
- Label tie order: FCFS, then SSTF, then SCAN, then C-SCAN

## Scheduler Conventions

SCAN moves to the disk endpoint before reversing. With upward direction, it services requests at the current head, then requests above the head in ascending order, moves to cylinder 199, reverses, and services lower requests in descending order. Endpoint movement is counted even when the endpoint is not requested.

C-SCAN moves in one direction, reaches cylinder 199, wraps to cylinder 0, and continues servicing lower requests in ascending order. The wrap-around distance is counted as head movement. Requests equal to the initial head are serviced first at zero movement.

## Result Summary

The generated decision tree test accuracy is 84.38 percent. Training-label distribution is:

- FCFS: 93
- SCAN: 12
- SSTF: 535
- C-SCAN: 0

C-SCAN receives no best-scheduler labels under this experiment because the true C-SCAN wrap distance is counted and total head movement is the metric.

Mean total head movement highlights from `results/summary_results.csv`:

- Uniform: FCFS 2680.20, SSTF 303.26, learned selector 303.26, oracle 302.18
- Localized: FCFS 524.24, SSTF 93.32, learned selector 93.32, oracle 93.32
- Directional: FCFS 192.22, SSTF 225.28, learned selector 211.16, oracle 187.98
- Bursty: FCFS 600.80, SSTF 206.86, learned selector 206.86, oracle 206.80

The workload-shift experiment records per-window predictions in `results/workload_shift_results.csv`. In the directional phase, the learned selector averaged 214.50 cylinders against an oracle average of 193.25. In the uniform phase, it averaged 287.58 cylinders against an oracle average of 286.08.
