# Out-of-Distribution Evaluation of World Models - Orbis, VJEPA 2.1
Review the original Readme file for complete details - [readme_original.md](readme_original.md)

## 0. Curating Benchmarks
Since DoTA collects data at a sampling frequency of 10 fps, we subsampled 5 frames of context at 5 fps, for each of the training and test sequences. This is to stay consistent with the original experimental setup on ORBIS, and to not introduce any additional variance in the studies. We used a train-validation split of 4:1 and utilized 3000 data sequences from the DoTA dataset, to derive OOD and ID samples from each sequence, by leveraging the temporal annotations provided by the dataset. We also normalized the data (Layer Norm) before feeding the data into the classifier/self-attention blocks.

Each clip in DOTA has some segment (ie. some number of frames) which is marked as anomalous and we used this annotation to generate anomalous and non-anomalous frames for a single clip. Also since we need to sample data at 5 fps we picked the alternate frames to convert it from 10 to 5 fps. When we didn't have enough clips after the end of the anomalous section, we picked from start.

DOTA has 9 classes in total - oncoming, leave-to-left, leave-to-right, lateral, moving-ahead-or-waiting, turning, unknown, start-stop-or-stationary, pedestrian

among these we ignored the class unknown and also skipped night frames using annotations!

For the surprise detection experiment, we chose 3000 in-distribution samples to get a calibrated mean and std-dev and used these to normalize our scores for Out-of-Distribution sample scores.

We also used a sample size of 3000 samples and this sample was used to evaluate our classifiers for Orbis activations, Surprise attention and Vjepa activations.

---
## 1. Methods
### 1.1 Retrospective Surprise Measure

For the **Retrospective Surprise Measure** experiment. The workflow extracts surprise maps from target clips across multiple architectural heads (`detailed`, `semantic`, and `combined`) and timesteps, computes calibration metrics across non-OOD baseline samples, normalizes scores using the baseline statistics, and generates spatial visual overlays.

### 1.2 Orbis Activations

For the **Retrospective Surprise Measure** experiment. The workflow extracts surprise maps from target clips across multiple architectural heads (`detailed`, `semantic`, and `combined`) and timesteps, computes calibration metrics across non-OOD baseline samples, normalizes scores using the baseline statistics, and generates spatial visual overlays.

### 1.3 VJEPA Probes

For the **Retrospective Surprise Measure** experiment. The workflow extracts surprise maps from target clips across multiple architectural heads (`detailed`, `semantic`, and `combined`) and timesteps, computes calibration metrics across non-OOD baseline samples, normalizes scores using the baseline statistics, and generates spatial visual overlays.

---
## 2. Run Experiments
### 2.1 Experiment Replication for Surprise Workflow

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

### 2.2 Experiment Replication for Orbis Diagnostic Probes

Step 1: Dataset Preparation - generates 'DoTA_training' folder with all the subsampled frames from the original DoTA dataset, along with a manifest file to capture all the metadata like ID, source_class, binary_label, multiclass_label corresponding to each of the training sequences 
```python
python DiagnosticProbes/scripts/dota.py
```

Step 2: Generate and Cache Activations/Embeddings

```python
python DiagnosticProbes/scoring/cache_orbis_activations --max_samples=3000

python DiagnosticProbes/scoring/cache_orbis_embeddings.py --max_samples=3000
```

Step 3: Train the corresponding Linear Probes and Attention Probes

```python
python DiagnosticProbes/<Activations/Encoder/Surprise>/linear_<maxpool/attention>_<probe/encoder>_<binary/mc>.py_
```

Step 4: Generate stats, visualize the metrics and the corresponding heatmaps

```python
python DiagnosticProbes/scripts/stats.py \
    --cache_path ./cached_features/cached_normalized_surprise_scores_3000.pt \
    --output_path reports/stats_report_surprise_combined_t3.md \
    --plot_dir reports/confusion_matrices_surprise_combined_t3

```

Step 5: Generate comparison plots for the summary results

```python
python DiagnosticProbes/scripts/plot_comparison.py
```

### 2.3 Experiment Replication for VJEPA Diagnostic Probes (review dota_vjepa_implementation branch for full details)

---

## 3 Appendix


## 3.1 Issues faced with MPS implementation
MPS issues - NotImplementedError: The operator 'aten::_upsample_bicubic2d_aa.out' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
