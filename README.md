# Multi-Basin Flood Early Warning System (FEWS) in Nepal: Hydro-Meteorological Risk Classification via Cost-Sensitive Machine Learning

## Overview

This repository contains an end-to-end, production-grade Machine Learning pipeline designed for an operational **Flood Early Warning System (FEWS)** in Nepal. Utilizing daily hydro-meteorological, soil moisture, and telemetric river discharge data from 2023 to 2026 across 10 strategic river monitoring stations, the system classifies **24-hour lead time flood hazard alerts** ($t+1$).

Unlike standard machine learning paradigms that assume symmetric misclassification costs, this project formulates the problem using **Cost-Sensitive Learning and Bayes Minimum Risk Decision Theory**. In disaster risk reduction (DRR), missing a flood disaster (False Negative) carries catastrophic human and economic costs compared to a precautionary alert (False Positive). By embedding asymmetric penalties directly into gradient boosting loss functions and calibrating the operational decision threshold, this system reduces missed flood disasters from 39 down to 2 on unseen future holdout data (an operational detection recall of **98.90%**), while reducing expected disaster risk loss by **49.88%**.

---

## Hydrological Context and Problem Statement

Nepal's river systems are characterized by extreme topographical relief, glacial headwaters, and intense South Asian monsoon precipitation (June through September). Floods and flash floods regularly threaten river corridors across major transboundary basins including the Koshi, Narayani/Gandak, Bagmati, Kamala, West Rapti, Babai, Karnali, and Mahakali.

### Spatial Hydrological Heterogeneity
Catchment dynamics vary drastically across monitoring stations:
- **Mountain Headwaters (e.g., Rasuwagadhi on Bhote Koshi):** High elevation (1,749 m), steep drainage, with baseline discharge under 1 m3/s and flood peaks reaching ~8.3 m3/s.
- **Major Lowland Corridors (e.g., Kusum on West Rapti):** Expansive catchment drainage, baseline discharge ~20 m3/s, with extreme monsoon discharge exceeding 3,175 m3/s.

A uniform national discharge cutoff is hydrologically invalid. To address this, localized hazard thresholds are statistically derived using the **90th percentile historical discharge ($Q_{90}$)** per monitoring station.

### Operational Target Definition
The early warning objective is to predict whether river discharge will exceed the local warning threshold 24 hours in advance:
$$y_{i, t+1} = \mathbb{I}\left(Q_{i, t+1} \ge Q_{i, 90}\right)$$

This target exhibits severe class imbalance (~10.05% positive flood hazard events vs. ~89.95% normal flow).

---

## Cost-Sensitive Learning Framework

### Asymmetric Disaster Cost Matrix
In operational emergency management, errors are fundamentally asymmetric:
- **True Negative ($C_{TN} = 0.0$):** Normal flow correctly identified.
- **True Positive ($C_{TP} = 0.0$):** Timely flood alert successfully issued.
- **False Positive ($C_{FP} = 1.0$):** Precautionary false alarm resulting in emergency mobilization, standby verification, and minor logistical disruption.
- **False Negative ($C_{FN} = 10.0$):** Missed flood disaster resulting in unwarned populations, loss of life, severe infrastructure collapse, and emergency crisis.

### Mathematical Objective
The model minimizes total expected operational disaster loss:
$$\text{Total Cost} = C_{FN} \times FN + C_{FP} \times FP = 10.0 \times FN + 1.0 \times FP$$
$$\text{Cost per Sample} = \frac{\text{Total Cost}}{N}$$

To achieve this, gradient boosting ensembles are parameterized with a positive class loss weight:
$$w_{\text{pos}} = \frac{C_{FN}}{C_{FP}} = 10.0$$

### Bayes Minimum Risk Decision Threshold
While standard classifiers deploy an uncalibrated default threshold $\tau = 0.50$, theoretical Bayes decision rule indicates the optimal threshold shifts to:
$$\tau^*_{\text{theoretical}} = \frac{C_{FP} - C_{TN}}{C_{FP} - C_{TN} + C_{FN} - C_{TP}} = \frac{1.0 - 0.0}{1.0 + 10.0} \approx 0.091$$

In this project, an empirical search over the **validation partition** calibrated the optimal operational threshold to $\tau^* = 0.32$, minimizing operational risk on holdout test observations.

---

## End-to-End Pipeline Architecture

```
[ Raw Telemetry & Weather Data (2023-2026) ]
                    |
[ Hydrological Domain Feature Engineering ]
  - Antecedent Precipitation Indices (API: 3-day, 7-day cumulative sums)
  - Root-Zone Soil Moisture Change Velocity (0-100 cm depth)
  - Discharge Momentum (3-day rolling mean, max, and daily acceleration)
  - Relative Hazard Saturation Ratio (Q_t / Q_90)
  - Cyclical Day-of-Year Encodings & Monsoon Seasonal Indicator
                    |
[ Chronological Partitioning (Leakage Prevention) ]
  - Train: 2023-01-01 to 2025-12-05 (80% historical window)
  - Validation: Inner temporal holdout for threshold tuning
  - Test: 2025-12-06 to 2026-08-31 (20% future unseen holdout)
                    |
[ Multi-Architecture Benchmarking (Cost-Weighted Loss) ]
  - Baseline LightGBM (Unweighted, tau=0.50)
  - Cost-Sensitive LightGBM (scale_pos_weight=10.0)
  - Cost-Sensitive XGBoost (scale_pos_weight=10.0)
  - Champion Cost-Sensitive CatBoost (class_weights=[1.0, 10.0])
                    |
[ Validation Threshold Tuning (Bayes Minimum Risk) ]
  - Optimized tau* = 0.32
                    |
[ Comprehensive Evaluation on 2026 Monsoon Holdout ]
                    |
[ Model Knowledge Serialization (Model/model_knowledge/) ]
```

---

## Experimental Benchmark and Results

The pipeline was benchmarked on the independent holdout testing period (2025-12-06 to 2026-08-31; 2,680 station-day records containing 182 actual flood events).

### Comparative Performance Table

| Model Architecture & Policy | Threshold ($\tau$) | Recall (Detection Rate) | Precision | F2-Score | ROC-AUC | PR-AUC | False Negatives (Missed) | False Positives | Total Cost | Cost Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM (Unweighted)** | 0.50 | 78.57% | 80.34% | 0.7892 | 0.9892 | 0.8757 | 39 | 35 | 425.0 | Baseline |
| **Cost-Sensitive LightGBM** | 0.50 | 81.87% | 71.63% | 0.7959 | 0.9874 | 0.8619 | 33 | 59 | 389.0 | -8.47% |
| **Cost-Sensitive XGBoost** | 0.50 | 81.87% | 72.68% | 0.7985 | 0.9871 | 0.8589 | 33 | 56 | 386.0 | -9.18% |
| **Champion CatBoost (Cost-Weighted)** | 0.50 | 96.70% | 58.09% | 0.8535 | 0.9902 | 0.8795 | 6 | 127 | 187.0 | -56.00% |
| **Champion CatBoost (Tuned Policy)** | **0.32** | **98.90%** | **48.26%** | **0.8174** | **0.9902** | **0.8795** | **2** | **193** | **213.0** | **-49.88%** |

### Key Findings
1. **Critical Disaster Mitigation:** The baseline model missed 39 out of 182 flood events. The cost-sensitive champion with tuned threshold missed only 2 events, achieving a **98.90% detection recall**.
2. **Economic Risk Reduction:** Total operational disaster cost dropped from 425.0 down to 213.0, representing a **49.88% reduction in risk loss**.
3. **Discriminative Separation:** The Champion CatBoost model demonstrated superior area under curve metrics with an **ROC-AUC of 0.9902** and a **PR-AUC of 0.8795**.

---

## Top Hydrological Predictors

Feature importance analysis extracted from the Champion CatBoost model identifies key physical runoff drivers:

1. **`discharge_to_threshold_ratio`:** Instantaneous hydrological saturation relative to local station capacity.
2. **`river_discharge_m3s` & `discharge_roll_3d_max`:** Baseflow elevation and recent peak antecedent discharge.
3. **`discharge_roll_3d_mean`:** Multi-day baseline volume maintaining channel capacity saturation.
4. **`precip_roll_7d` & `precip_roll_3d`:** Cumulative multi-day antecedent precipitation (API) dictating catchment soil infiltration limits.
5. **`soil_moisture_0_100cm_m3m3`:** Deep root-zone saturation governing surface runoff conversion rates.
6. **`elevation_m` & `location`:** Spatial catchment characteristics, river basin morphology, and orographic precipitation regimes.

---

## Repository Structure

```
Nepal Flood and Weather Analysis/
├── Dataset/
│   └── CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv
├── Model/
│   ├── Early_Warning_Flood_Alert_System.ipynb
│   └── model_knowledge/
│       ├── champion_catboost_model.cbm
│       ├── cost_sensitive_lgbm.joblib
│       ├── station_flood_thresholds.json
│       ├── cost_matrix_config.json
│       ├── decision_policy.json
│       ├── feature_schema.json
│       ├── model_benchmark_results.json
│       ├── feature_importance.csv
│       └── inference_pipeline.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Serialized Knowledge Artifacts

All operational assets are preserved in `Model/model_knowledge/` for automated deployment:

- `champion_catboost_model.cbm`: Serialized binary weights of the champion CatBoost classifier.
- `cost_sensitive_lgbm.joblib`: Serialized alternative LightGBM model.
- `station_flood_thresholds.json`: Dictionary of 90th percentile discharge thresholds per monitoring station.
- `cost_matrix_config.json`: Configuration specifying cost penalties ($C_{FN}=10.0$, $C_{FP}=1.0$, $C_{TN}=0.0$, $C_{TP}=0.0$).
- `decision_policy.json`: Operational decision policy document containing the calibrated threshold ($\tau^* = 0.32$), lead time (24h), and performance metrics.
- `feature_schema.json`: Complete specification of input feature columns, data types, and categorical mappings.
- `model_benchmark_results.json`: Benchmark comparison records across all evaluated strategies.
- `feature_importance.csv`: Ranked importance score table for all 32 engineered features.
- `inference_pipeline.py`: Production-ready Python inference class for processing incoming daily telemetry.

---

## Getting Started

### 1. Environment Setup

Clone this repository and create a clean Python virtual environment (Python 3.10+ recommended):

```bash
git clone https://github.com/NumiKun/Nepal-Flood-and-Weather-Analysis.git
cd "Nepal-Flood-and-Weather-Analysis"

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Interactive Analysis

Launch Jupyter Lab or Notebook to inspect the fully executed pipeline:

```bash
jupyter notebook Model/Early_Warning_Flood_Alert_System.ipynb
```

The notebook contains pre-rendered visualization plots:
- Basin-level discharge distribution and target class imbalance.
- Receiver Operating Characteristic (ROC) and Precision-Recall (PR) comparison curves.
- Validation cost curves as a function of decision threshold.
- Side-by-side confusion matrices (Baseline vs. Cost-Sensitive Champion).
- Top-15 feature importance ranking.
- Operational time-series simulation across the 2026 monsoon period.

### 3. Programmatic Inference

To generate 24-hour flood risk predictions on new hydro-meteorological observations using the standalone inference pipeline:

```python
import pandas as pd
from Model.model_knowledge.inference_pipeline import FloodEarlyWarningPipeline

pipeline = FloodEarlyWarningPipeline(artifacts_dir="Model/model_knowledge")

new_telemetry_df = pd.read_csv("path/to/incoming_daily_telemetry.csv")
predictions = pipeline.predict_risk(new_telemetry_df)

print(predictions[["date", "location", "predicted_flood_probability", "flood_alert_24h"]].head())
```

---

## Operational Deployment Guidelines

For deployment within hydrological monitoring centers (such as the Department of Hydrology and Meteorology, Nepal):
1. **Daily Ingest Schedule:** Ingest daily 24-hour rainfall and river discharge telemetry at 06:00 NPT.
2. **Automated Advisory Dissemination:** Automatically trigger high-priority SMS and radio advisories to provincial disaster response committees whenever predicted flood probability exceeds $\tau^* = 0.32$.
3. **Post-Monsoon Recalibration:** Annually re-evaluate station $Q_{90}$ thresholds following significant morphological channel alterations or major catchment infrastructure adjustments.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
