# Solar Event Tracking with Deep Regression Networks

> **Paper:** [Solar Event Tracking with Deep Regression Networks: A Proof of Concept Evaluation](https://ieeexplore.ieee.org/document/9006273)
> IEEE International Conference on Big Data (Big Data), 2019

<div style="display:flex; gap:10px; margin: 16px 0;">
  <img src="https://raw.githubusercontent.com/toqitahamid/solar-event-tracking/main/videos/HMI_7269_labels/106.jpg" style="width:400px;">
  <img src="https://raw.githubusercontent.com/toqitahamid/solar-event-tracking/main/videos/HMI_7269_labels/151.jpg" style="width:400px;">
</div>

*White rectangle: ground truth &nbsp;|&nbsp; Red rectangle: predicted bounding box*

[Watch the demo video on YouTube](https://www.youtube.com/watch?v=jgumuJfT5Pc)

---

## Overview

This project implements an automated pipeline for tracking solar events in Solar Dynamics Observatory (SDO) imagery using deep regression networks. Events such as coronal holes and active regions are sourced from NASA's Heliospheric Events Knowledgebase (HEK) and tracked across image sequences using the [GOTURN](https://github.com/davheld/GOTURN) deep regression tracker.

**Pipeline:**

```
HEK Event Metadata  →  SDO Image Download  →  GOTURN Tracking  →  Evaluation
```

---

## Repository Structure

```
solar-event-tracking/
├── data-download/
│   ├── hek-event-download/
│   │   └── hek_event_download_CH_SPoCA.py   # Query HEK for solar event metadata
│   └── image-download/
│       ├── download_image_AR_dump.py         # Download SDO images via Helioviewer
│       ├── convert_jp2_to_jpg.py             # Convert JPEG2000 → JPG
│       ├── convert_hpc_to_pixel.py           # Transform HPC coordinates to pixels
│       ├── save_all_image_information_ar_12_13.py
│       ├── find_events_with_double_image_name.py
│       └── check_image_folder*.py            # Data validation utilities
├── evaluation-tracking/
│   └── evaluate_all_csv.py                   # Compute tracking metrics
├── videos/
│   ├── label-video.py                        # Extract labeled frames from AVI
│   ├── test_evaluate.py                      # Evaluation test script
│   └── HMI_7269_labels/                      # Sample labeled output frames
└── README.md
```

---

## Dependencies

- Python 3
- [sunpy](https://sunpy.org/) — HEK API queries and JP2 image handling
- [OpenCV](https://opencv.org/) (`cv2`) — image processing and visualization
- [pandas](https://pandas.pydata.org/) — data manipulation and CSV I/O
- [numpy](https://numpy.org/) — numerical computations
- pgmagick — JPEG2000 image conversion
- Pillow — image utilities

---

## Usage

### 1. Download Solar Event Metadata

Query the [Heliospheric Events Knowledgebase (HEK)](https://www.lmsal.com/hek/) for coronal hole and sunspot events:

```bash
cd data-download/hek-event-download
python hek_event_download_CH_SPoCA.py
```

This outputs CSV files containing event metadata (bounding boxes, timestamps, solar coordinates).

### 2. Download Solar Images

Using the HEK event CSV files, download the corresponding SDO images from [Helioviewer](https://student.helioviewer.org/):

```bash
cd data-download/image-download
python download_image_AR_dump.py
```

**Preprocessing utilities in the same folder:**

| Script | Purpose |
|--------|---------|
| `convert_jp2_to_jpg.py` | Convert SDO's native JPEG2000 format to JPG |
| `convert_hpc_to_pixel.py` | Transform solar (HPC) coordinates to image pixel coordinates |
| `find_events_with_double_image_name.py` | Detect and remove duplicate events |
| `check_image_folder*.py` | Validate downloaded image sets |

### 3. Train the Tracker

Train the deep regression network using [GOTURN](https://github.com/davheld/GOTURN) on the downloaded image sequences. Refer to the GOTURN repository for setup and training instructions.

### 4. Evaluate

Compute tracking accuracy metrics by comparing GOTURN output against ground-truth bounding boxes:

```bash
cd evaluation-tracking
python evaluate_all_csv.py
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| IoU | Intersection over Union per frame |
| F-score | Precision/recall balance |
| OTA | Overall Tracking Accuracy |
| OTP | Overall Tracking Precision |
| ATA | Average Tracking Accuracy |
| Deviation | Centroid distance between predicted and ground truth |
| PBM | Partial Bounding Match |

---

## Data Sources

- **Solar events:** [Heliospheric Events Knowledgebase (HEK)](https://www.lmsal.com/hek/)
- **Solar images:** [Solar Dynamics Observatory (SDO)](https://sdo.gsfc.nasa.gov/) via [Helioviewer](https://student.helioviewer.org/)
- **Event types:** Coronal Holes (CH), Active Regions (AR) — from SPoCA and NOAA SWPC Observer

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@INPROCEEDINGS{9006273,
  author={Sarker, Toqi Tahamid and Banda, Juan M.},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)},
  title={Solar Event Tracking with Deep Regression Networks: A Proof of Concept Evaluation},
  year={2019},
  pages={4942-4949},
  doi={10.1109/BigData47090.2019.9006273}
}
```
