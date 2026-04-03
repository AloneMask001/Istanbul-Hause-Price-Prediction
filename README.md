<p align="center">
  <h1 align="center">🏠 Istanbul House Price Prediction</h1>
  <p align="center">
    <strong>Predicting Istanbul real estate prices using Machine Learning</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
    <img src="https://img.shields.io/badge/Jupyter-Notebook-red?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  </p>
</p>

---

## ✨ About The Project

A machine learning project that analyzes **~2983 real estate listings** from Istanbul and builds a **Random Forest Regressor** model to predict house prices. The project covers the full data science pipeline — from data cleaning and feature engineering to model training and evaluation.

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
    <td align="center"><b>🤖 Model</b><br>Random Forest</td>
    <td align="center"><b>📈 OOB Score</b><br>0.778</td>
    <td align="center"><b>🎯 Test R²</b><br>0.611</td>
    <td align="center"><b>📐 Features</b><br>16 engineered</td>
  </tr>
</table>

---

## 🎯 Model Performance by Price Segment

The model's accuracy varies across different price ranges. It performs **best on regular-priced homes** and less accurately on extreme/luxury prices, which are underrepresented in the dataset.

| Price Segment | Listing Count | MAE (TRY) | Accuracy |
|:---:|:---:|:---:|:---:|
| 🟢 **0 – 2M TRY** | 1,662 | Lower | ✅ Most Accurate |
| 🟡 **2M – 10M TRY** | 1,012 | ~1.5M | ⚠️ Moderate |
| 🔴 **10M+ TRY** | 226 | Higher | ❌ Less Accurate |

> ⚠️ **Why do extreme prices have higher error rates?**  
> Luxury properties (10M+ TRY) make up only **~8%** of the dataset. With so few samples, the model struggles to learn the unique pricing patterns of these properties — such as premium locations, special amenities, or architectural features that justify extreme prices.

### Overall Metrics

| Metric | Value |
|:---|:---:|
| **OOB Score** (log prices) | `0.778` |
| **Test Score** (log prices) | `0.781` |
| **R² Score** (actual prices) | `0.611` |
| **RMSE** | `~3.1M TRY` |
| **MAE** | `~1.26M TRY` |

---

## 🏆 Feature Importance

The features that matter most when predicting house prices:

```
İlçe_Mahalle_target  ████████████████████████████  26.7%  📍 Location
Yaşam_endeksi        ███████████████████           19.0%  🏘️ Living Index
Net_Metrekare        ██████████████████            15.4%  📐 Area (m²)
Nüfus                █████████                      8.7%  👥 Population
Banyo_Sayısı         ████████                       8.2%  🚿 Bathrooms
Oda_Sayısı           ██████                         6.0%  🚪 Rooms
Binanın_Kat_Sayısı   ████                           4.4%  🏗️ Total Floors
kat_oran             ███                            3.1%  📊 Floor Ratio
```

> 🔑 **Top 3 price determinants:** Location, Living Index, and Size account for over **61%** of the prediction power.

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
| `İlçe_Mahalle_target` | Target encoded location (engineered) | Numeric |
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
    │       └── İlçe_Mahalle_target (Target Encoding)
    │
    ├── 5️⃣  Outlier Removal (price > 50M, floor > building floors)
    │
    └── 6️⃣  Log Transform target variable (np.log1p)
            │
            ▼
📊 Clean Data (16 features, ~2919 rows) → 🤖 Random Forest Model
```

---

## 🤖 Model Configuration

```python
RandomForestRegressor(
    n_estimators=300,       # 🌲 Number of trees
    random_state=42,        # 🎲 Reproducibility
    oob_score=True,         # 📊 Out-of-bag evaluation
    max_depth=25,           # 📏 Max tree depth
    min_samples_leaf=2,     # 🍃 Min samples per leaf
    max_features="sqrt"     # 🔢 Features per split
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
pip install numpy pandas matplotlib scikit-learn jupyter

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
