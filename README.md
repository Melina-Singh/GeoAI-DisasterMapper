# 🛰️ GeoAI-DisasterMapper

> Automated Building Damage Detection from Satellite Imagery using Deep Learning and Climate Vulnerability Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Overview

GeoAI-DisasterMapper is an end-to-end GeoAI research project that combines deep learning, remote sensing, and climate analytics to automatically detect building damage from satellite imagery and assess climate vulnerability in affected regions.

The system uses before-and-after satellite image pairs to identify structural damage caused by natural disasters such as earthquakes, floods, hurricanes, wildfires, and tsunamis. Predicted damage regions are exported as geospatial GeoJSON files with real-world coordinates and further analyzed using NASA POWER climate datasets.

This project was built as a research portfolio project for Graduate Research Assistant applications in GeoAI, Remote Sensing, and Climate Change.

---

# 🗺️ Key Features

* Automated building damage detection from satellite imagery
* Deep learning segmentation using U-Net + ResNet50
* Multi-class damage classification
* GeoJSON export with exact geographic coordinates
* Climate vulnerability analysis using NASA POWER API
* Interactive Streamlit dashboard
* Geospatial visualization support for QGIS and ArcGIS
* Disaster assessment workflow for remote sensing research

---

# 🏗️ Project Pipeline

Satellite Imagery (Before + After)
↓
Preprocessing and Image Tiling
↓
U-Net Deep Learning Model Training
↓
Damage Segmentation Inference
↓
GeoJSON Damage Polygon Generation
↓
NASA POWER Climate Data Integration
↓
Climate Vulnerability Assessment
↓
Interactive Streamlit Web Application

---

# 🛠️ Tech Stack

| Component              | Technology                     |
| ---------------------- | ------------------------------ |
| Deep Learning          | PyTorch                        |
| Segmentation Framework | segmentation-models-pytorch    |
| Model Architecture     | U-Net with ResNet50 Encoder    |
| Dataset                | xBD / xView2 Challenge Dataset |
| Geospatial Processing  | Rasterio, GeoPandas, Shapely   |
| Satellite Imagery      | Landsat 8                      |
| Climate Data           | NASA POWER API                 |
| Visualization          | Folium, QGIS                   |
| Web Framework          | Streamlit                      |
| Training Environment   | Google Colab T4 GPU            |

---

# 📦 Installation

## 1. Clone the Repository

git clone https://github.com/Melina-Singh/GeoAI-DisasterMapper.git

cd GeoAI-DisasterMapper

---

## 2. Create Virtual Environment

### Windows

venv\Scripts\activate

### Linux / Mac

source venv/bin/activate

---

## 3. Install Dependencies

pip install -r requirements.txt

---

## 4. Download Trained Model

Download:
best_model_resnet50.pth

from the Releases section and place it inside:

model/

---

## 5. Run the Application

streamlit run app.py

---

# 🚀 Usage

## Streamlit Dashboard

Run:

streamlit run app.py

Then open:

http://localhost:8501

---

## Damage Detection Workflow

1. Upload a before-disaster satellite image
2. Upload an after-disaster satellite image
3. Click RUN DAMAGE DETECTION
4. View predicted damage segmentation
5. Download GeoJSON damage polygons

---

## Climate Vulnerability Workflow

1. Select a disaster region or enter coordinates
2. Fetch NASA POWER climate data
3. Analyze:

   * Temperature trends
   * Rainfall patterns
   * Humidity changes
4. Assess long-term vulnerability risk

---

# 🧠 Deep Learning Model

## Model Architecture

* U-Net semantic segmentation network
* ResNet50 encoder backbone
* Multi-class building damage segmentation

---

## Damage Classes

| Class ID | Label        |
| -------- | ------------ |
| 0        | Background   |
| 1        | No Damage    |
| 2        | Minor Damage |
| 3        | Major Damage |
| 4        | Destroyed    |

---

# 🌍 Disaster Coverage

The model was evaluated on multiple disaster events from the xBD dataset.

| Disaster            | Type              | Year |
| ------------------- | ----------------- | ---- |
| Guatemala Volcano   | Volcanic Eruption | 2018 |
| Hurricane Florence  | Hurricane         | 2018 |
| Hurricane Harvey    | Hurricane         | 2017 |
| Hurricane Matthew   | Hurricane         | 2016 |
| Hurricane Michael   | Hurricane         | 2018 |
| Mexico Earthquake   | Earthquake        | 2017 |
| Midwest Flooding    | Flood             | 2019 |
| Palu Tsunami        | Tsunami           | 2018 |
| Santa Rosa Wildfire | Wildfire          | 2017 |
| SoCal Fire          | Wildfire          | 2017 |

---

# 🌡️ Climate Vulnerability Analysis

The project integrates NASA POWER climate datasets to analyze long-term environmental vulnerability in disaster-affected areas.

## Climate Parameters

| Parameter   | Description             | Unit   |
| ----------- | ----------------------- | ------ |
| T2M         | Air Temperature at 2m   | °C     |
| PRECTOTCORR | Corrected Precipitation | mm/day |
| RH2M        | Relative Humidity       | %      |

---

## Why Climate Analysis Matters

Disaster-affected areas often face secondary climate risks after infrastructure damage.

Examples include:

* Flash flooding in damaged urban zones
* Landslides on weakened terrain
* Heat stress in areas with vegetation loss
* Long-term vulnerability from rising temperatures

This project connects immediate disaster impact with future climate resilience planning.

---

Supported Platforms:

* QGIS
* ArcGIS
* Google Earth
* Folium
* GeoPandas

---

# 🔮 Future Improvements

* Fine-tune on Nepal-specific disaster imagery
* Add Sentinel-2 multispectral support
* Integrate Himalayan GLOF risk analysis
* Deploy cloud inference API
* Add temporal disaster progression analysis
* Improve small-building segmentation accuracy

---

# 📚 References

Gupta, R., et al. (2019).
Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery.
CVPR Workshops.

Ronneberger, O., et al. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.

NASA POWER Project
https://power.larc.nasa.gov

xView2 Challenge
https://xview2.org


---

# 📄 License

MIT License

This project is open-source and free to use for research and educational purposes.
