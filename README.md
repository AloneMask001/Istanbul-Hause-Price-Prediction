[README.md](https://github.com/user-attachments/files/28794784/README.md)
<p align="center">
  <h1 align="center">🏠 Istanbul House Price Prediction</h1>
  <p align="center">
    <strong>Predicting Istanbul real estate prices using Machine Learning</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
    <img src="https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
    <img src="https://img.shields.io/badge/Jupyter-Notebook-red?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  </p>
</p>

---

## ✨ About The Project

A machine learning project that analyzes **~2983 real estate listings** from Istanbul and builds an **XGBoost Regressor** model to predict house prices. The project covers the full data science pipeline — from data cleaning and feature engineering to model training and evaluation.

> 💡 **Key Insight:** The model performs significantly better on regular-priced homes. Error rates increase for extreme/luxury prices due to their rarity in the dataset.

---

## 📊 Quick Stats

<table>
  <tr>
    <td align="center"><b>📦 Dataset</b><br>~2983 listings</td>
    <td align="center"><b>🏙️ City</b><br>Istanbul</td>
    <td align="center"><b>🏘️ Districts</b><br>13 districts</td>
    <td align="center"><b>📍 Neighborhoods</b><br>210 neighborhoods</td>
  </tr>
  <tr>
    <td align="center"><b>🤖 Model</b><br>XGBoost</td>
    <td align="center"><b>📈 CV R²</b><br>0.762</td>
    <td align="center"><b>🎯 Test R²</b><br>0.763</td>
    <td align="center"><b>📐 Features</b><br>16 engineered</td>
  </tr>
</table>

---

## 🎯 Model Performance by Price Segment

The model's accuracy varies across different price ranges. It performs **best on regular-priced homes** and less accurately on extreme/luxury prices, which are underrepresented in the dataset.

| Price Segment | Listing Count | MAE (TRY) | Accuracy |
|:---:|:---:|:---:|:---:|
| 🟢 **0 – 2M TRY** | 1,662 | ~317K | ✅ Most Accurate |
| 🟡 **2M – 10M TRY** | 1,012 | ~1.6M | ⚠️ Moderate |
| 🔴 **10M+ TRY** | 226 | ~8.2M | ❌ Less Accurate |

> ⚠️ **Why do extreme prices have higher error rates?**  
> Luxury properties (10M+ TRY) make up only **~8%** of the dataset. With so few samples, the model struggles to learn the unique pricing patterns of these properties — such as premium locations, special amenities, or architectural features that justify extreme prices.

### Overall Metrics

| Metric | Value |
|:---|:---:|
| **CV R²** (log prices) | `0.762` |
| **Test R²** (log prices) | `0.763` |
| **R² Score** (actual prices) | `0.613` |
| **MAE** | `~1.26M TRY` |

---

## 🏆 Feature Importance

The features that matter most when predicting house prices:

```
Yaşam_endeksi        ████████████████████████████  30.6%  🏘️ Living Index
Banyo_Sayısı         █████████████████████         21.8%  🚿 Bathrooms
İlçe_Mahalle_target  ████████                       8.4%  📍 Location
Nüfus                ███████                        6.9%  👥 Population
Net_Metrekare        ██████                         6.6%  📐 Area (m²)
Oda_Sayısı           █████                          4.4%  🚪 Rooms
Binanın_Kat_Sayısı   ████                           3.8%  🏗️ Total Floors
kat_oran             ███                            2.2%  📊 Floor Ratio
```

> 🔑 **Top 3 price determinants:** Living Index, Bathrooms, and Location account for over **66%** of the prediction power.

---

## 📋 Dataset Features

| Feature | Description | Type |
|:---|:---|:---:|
| `Net_Metrekare` | Net area in square meters | Numeric |
| `Binanın_Yaşı` | Building age (0–21+) | Categorical → Numeric |
| `Binanın_Kat_Sayısı` | Total number of floors | Numeric |
| `Bulunduğu_Kat` | Floor the apartment is on | Categorical → Numeric |
| `Oda_Sayısı` | Number of rooms (e.g. 3+1, 4+1) | Categorical → Encoded |
| `Banyo_Sayısı` | Number of bathrooms | Numeric |
| `Isıtma_Tipi` | Heating type (Natural Gas, AC, etc.) | Categorical → Encoded |
| `Kullanım_Durumu` | Occupancy status | Categorical → Encoded |
| `Krediye_Uygunluk` | Mortgage eligibility | Categorical → Encoded |
| `Site_İçerisinde` | In a residential complex? | Binary |
| `yaka` | European / Asian Side | Binary |
| `Yaşam_endeksi` | Living index score | Numeric |
| `Nüfus` | Area population (thousands) | Numeric |
| `kat_oran` | Floor ratio (engineered) | Numeric |
| `İlçe_Mahalle_target` | Target encoded location with smoothing (engineered) | Numeric |
| **`Fiyatı`** | **Property price (TRY) — Target** | **Numeric** |

---

## 🔧 Data Pipeline

```
📥 Raw Data (27 columns, 2983 rows)
    │
    ├── 1️⃣  Drop irrelevant columns (İlan_Numarası, Brüt_Metrekare, etc.)
    │
    ├── 2️⃣  Clean whitespace from categorical values
    │
    ├── 3️⃣  Convert categoricals → numericals
    │       ├── Custom functions (Binanın_Yaşı, Bulunduğu_Kat)
    │       └── LabelEncoder (Kullanım_Durumu, Isıtma_Tipi, etc.)
    │
    ├── 4️⃣  Feature Engineering
    │       ├── kat_oran = floor / total_floors
    │       └── İlçe_Mahalle_target (Target Encoding + Smoothing, m=15)
    │
    ├── 5️⃣  Outlier Removal (price > 50M, floor > building floors)
    │
    └── 6️⃣  Log Transform target variable (np.log1p)
            │
            ▼
📊 Clean Data (16 features, ~2919 rows) → 🤖 XGBoost Model
```

---

## 🤖 Model Configuration

```python
XGBRegressor(
    n_estimators=500,       # 🌲 Number of trees
    learning_rate=0.01,     # 📉 Step size shrinkage
    max_depth=6,            # 📏 Max tree depth
    min_child_weight=5,     # 🍃 Min sum of instance weight
    subsample=0.7,          # 🎲 Row sampling ratio
    colsample_bytree=0.8,   # 🔢 Feature sampling ratio
    gamma=0.3,              # ✂️ Min loss reduction to split
    random_state=42         # 🎲 Reproducibility
)
```

---

## 🛠️ Technologies

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="40"/><br><b>Python</b></td>
    <td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="40"/><br><b>NumPy</b></td>
    <td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="40"/><br><b>Pandas</b></td>
    <td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" width="40"/><br><b>Matplotlib</b></td>
    <td align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="40"/><br><b>Scikit-learn</b></td>
  </tr>
</table>

---

## 🚀 Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/AloneMask001/EV-Fiyat-Tahmini.git
cd EV-Fiyat-Tahmini

# 2. Install dependencies
pip install numpy pandas matplotlib scikit-learn xgboost jupyter

# 3. Run the notebook
jupyter notebook EV_tahmini.ipynb
```

> ⚠️ **Note:** The `ev_verisi.csv` dataset file must be placed in the same directory as the notebook.

---

## 📁 Project Structure

```
EV-Fiyat-Tahmini/
│
├── 📓 EV_tahmini.ipynb    # Main analysis & modeling notebook
├── 📊 ev_verisi.csv        # Dataset (Istanbul real estate listings)
└── 📄 README.md            # Project documentation
```

---

## 👤 Developer

**Semih Erdem Verep**

---

## 📄 License

This project was developed for educational purposes.

---

<p align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
</p>
