# Orbis



#### Retrospective Surprise Measure

For the **Retrospective Surprise Measure** experiment. The workflow extracts surprise maps from target clips across multiple architectural heads (`detailed`, `semantic`, and `combined`) and timesteps, computes calibration metrics across non-OOD baseline samples, normalizes scores using the baseline statistics, and generates spatial visual overlays.

---

## Experiment Replication Workflow

Step 1: Dataset Preparation

```python
python scripts/Dota_prepare.py
```

Step 2: Generate and Cache Raw Scores

```python
python DiagnosticProbes/scoring/cache_scores.py --max_samples=3000
```

Step 3: Compute Baseline Calibration Statistics

```python
python DiagnosticProbes/scoring/calibration.py
```

Step 4: Normalize Surprise Scores

```python
python DiagnosticProbes/scoring/normalize.py \
    --cache_path results/sample_scores.pt \
    --calib_path results_pt/calib_stats_combined.pt \
    --out_name normalized_combined_scores.pt
```

Step 5: Visualize and Overlay Anomalous Maps

```python
python DiagnosticProbes/scoring/plotter.py
```