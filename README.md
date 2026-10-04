# TRACE prototype

## Train the new burglary / normal / vandalism model

Upload [notebooks/Train_Crime_Colab.ipynb](notebooks/Train_Crime_Colab.ipynb) to Google Colab,
select a GPU runtime, and follow the cells. The notebook includes the current training
code, dataset storage checks, resumable feature extraction/training, and a model export
for this project. Read [the setup and storage guide](docs/CRIME_TRAINING.md) first: the
dataset archive alone is about 82 GB. After training and importing the export, use
`run_crime.cmd` to classify a video. This is a separate three-class video model; no new
trained weights or real-data accuracy results are included yet.

This is a small working prototype for the CCTV concept in `hih.pptx`. It detects people with a pretrained YOLO model, tracks them with ByteTrack, and raises one event when a tracked person remains inside a user-selected restricted zone for the configured dwell time. Each event gets a snapshot, JSONL record, and SQLite row. It deliberately does not identify faces or infer intent.

## Run it

For the supplied violence-video dataset, the trained model is ready: double-click `run_violence.cmd` to classify a short clip. See the [measured results](docs/VIOLENCE_RESULTS.md) and [violence training guide](docs/VIOLENCE_TRAINING.md). That pipeline trains a binary video classifier; the commands below run the person-and-zone prototype.

From PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m trace.monitor --source "C:\path\to\video.mp4" --select-zone --save-video
```

Click three or more zone corners, press Enter, and press Q to stop. The run is written under `outputs/`. The saved `zone.local.json` can be reused with `--zone zone.local.json` for another video from the same camera framing. For a webcam, use `--source 0`; for an RTSP camera, pass its URL and omit `--select-zone` after saving a zone from a representative frame.

The default is CPU inference. This environment has an RTX 4060, but the installed PyTorch wheel is CPU-only, so use `--device 0` only after installing a CUDA-enabled PyTorch build. Reducing `--imgsz` to 320 improves CPU speed; do not treat that as an accuracy result.

## Data and training

The first useful training set is footage from the actual cameras. Sample frames with:

```powershell
.\.venv\Scripts\python.exe scripts\extract_frames.py --video "C:\path\to\video.mp4" --output data\camera1_monday --every-seconds 2
```

Annotate every visible person with a person bounding box using CVAT, Label Studio, or Roboflow, export YOLO format, and make empty label files for reviewed frames with no people. Keep frames from one recording session in one split. Fill `examples/manifest.example.csv` with paths relative to the manifest, then validate and copy the dataset:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --manifest data\manifest.csv --output data\person_dataset
```

The manifest validator rejects missing labels, invalid normalized boxes, duplicate frames, and leakage where one recording group crosses train/validation/test. Train a small baseline on CPU:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data data\person_dataset\dataset.yaml --device cpu --epochs 30 --batch 2
```

Before committing to a long run, verify the pipeline only:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data data\person_dataset\dataset.yaml --smoke
```

The `data/coco8.zip` file is only a tiny plumbing sample. COCO is useful for a generic pretrained person detector, CrowdHuman is useful for dense crowds, and MOT17 is useful for validating tracking. Neither COCO nor CrowdHuman teaches “restricted area entry”; the event rule is defined by geometry and time. A later anomaly model can use UCF-Crime, but its video-level anomaly labels do not directly supervise this zone event.

## Suggested data plan

Start with 2–4 hours per camera covering daylight, night, empty scenes, normal movement near the boundary, partial occlusion, and the real restricted-area entries. Sample every 1–2 seconds, annotate roughly 1,000–3,000 diverse frames, and keep entire sessions together when splitting. After the baseline, review false alerts and add those exact scenes to a held-out test set. Measure person detection precision/recall and event precision, recall, alert delay, and false alerts per camera-hour.

## Safety and privacy boundaries

The prototype stores event snapshots locally, keeps camera credentials out of `settings.json`, and has no face recognition or identity tracking. Add retention limits, access control, human review, and a documented deletion process before connecting it to real CCTV. Check the license of every dataset and model before commercial deployment. Ultralytics YOLO is AGPL-3.0 by default, with separate licensing options for some commercial uses.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

