# Train the burglary / normal / vandalism model

Open `notebooks/Train_Crime_Colab.ipynb` in Google Colab using **File > Upload notebook**.
Choose **Runtime > Change runtime type > GPU**, then run cells in order. The notebook
includes a snapshot of this project's Python source; no GitHub push or private-repo
authentication is needed. It does not contain a trained model.

## Dataset and storage

The notebook uses version 1 of
[kizitoryan/burglary-and-vandalism-detection-dataset](https://www.kaggle.com/datasets/kizitoryan/burglary-and-vandalism-detection-dataset/data).
Its published CSVs list 100 burglary, 950 normal and 50 vandalism videos. The supplied
training set has 80/760/40; test has 20/190/10 respectively. These are metadata counts,
not a completed audit of the actual video archive.

The five `dataset.7z.001` through `.005` files together occupy about **82 GB / 76 GiB**.
All five parts must be available together. The extracted videos need additional space.
The notebook checks local free space before the large download, then reads the archive's
uncompressed sizes before extraction. It does not download the entire Kaggle dataset ZIP
on top of the five archive files. It downloads the parts individually.

In the configuration cell choose one storage arrangement:

- If everything fits, leave archives and extracted videos on the Colab runtime disk.
- If runtime storage is too small, put `ARCHIVE_DIR` on Google Drive with enough quota,
  and extract videos to the runtime. Mounted Drive free-space reports do not accurately
  describe your account quota: verify it in Drive before downloading 82 GB.
- If you already extracted the dataset, set `VIDEO_ROOT_OVERRIDE` to the directory
  containing `burglary/`, `normal/`, and `vandalism/`. This skips archive downloads.

Free Colab resources are not guaranteed to fit this dataset. Source archives on Drive
and videos on local disk are useful when sufficient Drive quota is available. Reading
videos directly from Drive is also supported, but can be slow. An interrupted runtime
loses `/content` files; remount Drive, restore/extract videos and rerun the notebook.
Persistent feature caches and epoch checkpoints avoid repeating completed model work.

For Kaggle authentication, use a Colab secret called `KAGGLE_API_TOKEN` if required.
Do not paste tokens into notebook cells. Public downloads may not require login.

## Model and labels

The class order is fixed: **0=burglary, 1=normal, 2=vandalism**. It is different from
the old binary violence pipeline. Neither checkpoint is interchangeable with the other.

The baseline freezes the pretrained R3D-18 video encoder and samples 32 evenly spaced
windows across each recording. Each window has 16 frames sampled at 15 FPS. The existing
TRACE resize/crop/normalization recipe is shared with violence inference. A trainable
attention layer pools the window features into a video representation, followed by a
three-class linear classifier. This is an attention-based multiple-instance-learning
baseline: it learns from one label for the complete video without declaring every
sampled window positive. Attention is **not** a validated timestamp or event confidence.

Feature extraction uses the GPU in Colab. The much smaller classifier trains on CPU in
full batches. Training uses inverse-frequency class weights, training-only feature
normalization, validation macro F1 checkpoint selection, and validation-loss tie breaking.
Early stopping defaults to 25 epochs without improvement. There is no accuracy target
or measured real-data result until the notebook is run.

## Splits and audit

The importer preserves the supplied test assignment and takes 20% of the supplied
training groups for validation (approximately 64% train / 16% val / 20% test overall).
The publisher generated its train/test split randomly and stratified by class; it is
not a verified camera-independent or original-UCF benchmark split.

Exact duplicates within the same source split and label are deduplicated. Cross-split
duplicates, conflicting labels, folder/label mismatches, missing files and groups that
cross splits stop preparation. Full SHA-256 hashing is deliberate and can take time.
Video decode failures stop training with the offending path; they are not silently
removed from evaluation. Review failed videos and create a new audited run if data changes.

By default each file is its own recording group. This cannot identify adjacent clips,
re-encoded copies or shared scenes. Supply an optional `GROUPS_CSV` with `video_path,group`
covering every retained video to group related recordings. Grouping that crosses the
publisher's train/test boundary requires a deliberately revised dataset split; the tool
will not silently move test examples into training. Filenames resemble UCF-Crime, but
provenance and overlap have not been verified. Do not use overlapping UCF clips as an
independent external test.

The dataset supplies whole-video labels, not event intervals. Sparse sampling can miss
events even with 32 windows. For better temporal supervision, annotate event intervals
in training footage. For deployment, evaluate on continuous footage from new cameras.

## Outputs and resuming

The notebook saves caches, audit files, encoder and run outputs under `MyDrive/TRACE_crime/`:

- `runs/crime_baseline/best.pt`: selected model, normalization, class names and sampling recipe.
- `runs/crime_baseline/last.pt`: last complete epoch, optimizer and selected model for resume.
- `metrics.json`: per-class precision/recall/F1, macro F1, balanced accuracy and confusion matrix.
- `predictions.csv`: scores for each evaluated video.
- `history.json`, `config.json`: training history and reproducibility settings.
- `audit/manifest.csv`, `audit/dataset_report.json`: split assignments and checksums.

Features are cached per video hash, encoder hash and sampling recipe. Writes use a
temporary file followed by replacement, so interrupted writes can be retried. Use the
same settings and `--resume` after an interruption. Dataset, source code, encoder or
optimization-setting changes require a new run name. The epoch limit can be increased
on resume, but an already early-stopped run remains stopped. CUDA/runtime changes may
still introduce numerical variation in newly extracted features.

The test split is evaluated only with `--evaluate-test`; it never selects a checkpoint.
The notebook defaults to final test evaluation after selection. Turn that option off
while developing settings, and do not choose settings based on repeated test results.

## Run the trained model on Windows

Download the exported `crime_baseline.zip` (the exact run name may differ) from Drive.
Extract its contents into the `T.R.A.C.E` project directory. It supplies:

```text
models/r3d_18-b3b3357e.pth
runs/crime_baseline/best.pt
```

If the Python environment is not installed yet:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then double-click `run_crime.cmd`, or select a checkpoint explicitly:

```powershell
.\.venv\Scripts\python.exe -m trace.predict_crime --source "C:\path\to\clip.mp4" --checkpoint runs\crime_baseline\best.pt
```

The prediction is a whole-video class with uncalibrated scores. A `normal` prediction
is not proof of no event. A `burglary` prediction is a possible event for review, not
proof of forced entry or unlawful intent. This model is not yet connected to the live
zone-monitoring window; the existing violence model and launcher remain separate.

## Local training commands

These also run in Colab/Linux with `python` instead of the Windows executable:

```powershell
.\.venv\Scripts\python.exe -m trace.crime_data --train-csv data\source\train.csv --test-csv data\source\test.csv --video-root data\source\dataset --output data\crime_audit
.\.venv\Scripts\python.exe -m trace.train_crime --manifest data\crime_audit\manifest.csv --video-root data\source\dataset --output runs\crime_baseline --device cpu --evaluate-test
```

When updating source code, regenerate the self-contained notebook with
`python scripts/build_crime_notebook.py` before uploading it again. The notebook records
its source checksums and exports its source snapshot with the results.

References: [KaggleHub download/authentication](https://github.com/Kaggle/kagglehub),
[Colab resource limits](https://research.google.com/colaboratory/faq.html),
[R3D-18 encoder](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html).
