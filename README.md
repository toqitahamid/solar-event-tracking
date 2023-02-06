# [Solar Event Tracking with Deep Regression Networks](https://ieeexplore.ieee.org/document/9006273)

<img src="https://raw.githubusercontent.com/toqitahamid/solar-event-tracking/main/videos/HMI_7269_labels/106.jpg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px; width: 400px" />


### Event Download
To download the solar event from [Heliospheric Events Knowledgebase (HEK)](https://www.lmsal.com/hek/) go to `data-download>hek-event-download` and run the `[hek_event_download_CH_SPoCA.py](https://github.com/toqitahamid/solar-event-tracking/blob/main/data-download/hek-event-download/hek_event_download_CH_SPoCA.py)`


### Solar Image Download
To download the images from [Solar Dynamics Observatory (SDO)](https://student.helioviewer.org/) go to `data_download>image-download` and run `[download_image_AR_dump.py](https://github.com/toqitahamid/solar-event-tracking/blob/main/data-download/image-download/download_image_AR_dump.py)` to download the solar images using the HEK event files.

The `image-download` folder has codes for data pre-processing such as convert JP2 to JPG, finding duplicate events etc.


### Training
To train the deep regression network use [GOTURN](https://github.com/davheld/GOTURN)

### Evaluation
To evaluate the model go to `evaluation-tracking` folder

Watch the [video](https://www.youtube.com/watch?v=jgumuJfT5Pc)

