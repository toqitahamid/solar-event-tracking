# [Solar Event Tracking with Deep Regression Networks](https://ieeexplore.ieee.org/document/9006273)

<div style="display:flex;">


<div style="display:flex;">
  <img src="https://raw.githubusercontent.com/toqitahamid/solar-event-tracking/main/videos/HMI_7269_labels/106.jpg" style="width:400px;margin-right:10px;">
  <img src="https://raw.githubusercontent.com/toqitahamid/solar-event-tracking/main/videos/HMI_7269_labels/151.jpg" style="width:400px;">
</div>

`White rectangle is the ground truth and the red rectangle is the generated bounding box`

### Event Download
To download the solar event from [Heliospheric Events Knowledgebase (HEK)](https://www.lmsal.com/hek/) go to `data-download>hek-event-download` and run the [hek_event_download_CH_SPoCA.py](https://github.com/toqitahamid/solar-event-tracking/blob/main/data-download/hek-event-download/hek_event_download_CH_SPoCA.py)


### Solar Image Download
To download the images from [Solar Dynamics Observatory (SDO)](https://student.helioviewer.org/) go to `data_download>image-download` and run [download_image_AR_dump.py](https://github.com/toqitahamid/solar-event-tracking/blob/main/data-download/image-download/download_image_AR_dump.py) to download the solar images using the HEK event files.

The `image-download` folder has codes for data pre-processing such as convert JP2 to JPG, finding duplicate events etc.


### Training
To train the deep regression network use [GOTURN](https://github.com/davheld/GOTURN)

### Evaluation
To evaluate the model go to `evaluation-tracking` folder and run the [evaluate_all_csv.py](https://github.com/toqitahamid/solar-event-tracking/blob/main/evaluation-tracking/evaluate_all_csv.py)

Watch the [video](https://www.youtube.com/watch?v=jgumuJfT5Pc)


```
@INPROCEEDINGS{9006273,
  author={Sarker, Toqi Tahamid and Banda, Juan M.},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)}, 
  title={Solar Event Tracking with Deep Regression Networks: A Proof of Concept Evaluation}, 
  year={2019},
  pages={4942-4949},
  doi={10.1109/BigData47090.2019.9006273}}

```

