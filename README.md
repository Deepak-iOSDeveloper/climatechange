# 🌏 India Climate Watch — Full-Stack Climate Intelligence Dashboard

> **The best Climate Change in India project** — Built with Django, Scikit-learn, NumPy, Pandas, Matplotlib & Seaborn

---

## 📸 What This Project Does

A full-stack Django web application that:

1. **Fetches 73 years of real climate data** from Open-Meteo's free ERA5 Historical API (no API key required)
2. **Runs ML models** — Linear Regression, Polynomial Regression (°3), Polynomial Regression (°5)
3. **Generates 6 publication-quality charts** using Matplotlib + Seaborn
4. **Forecasts India's temperature up to 2050** using Scikit-learn pipelines
5. **Displays everything** in a stunning dark-themed Django dashboard

---

## 🗂 Project Structure

```
climatechange/
├── climate/
│   ├── analysis.py          ← ALL ML + chart generation logic
│   ├── views.py             ← Django views
│   ├── urls.py              ← URL routing
│   └── templates/climate/
│       └── index.html       ← Full frontend dashboard
├── media/charts/            ← Generated chart PNGs
├── climatechange/
│   ├── settings.py
│   └── urls.py
└── manage.py
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install django pandas numpy matplotlib seaborn scikit-learn requests
```

### Step 2 — Run migrations
```bash
python manage.py migrate
```

### Step 3 — Start the server
```bash
python manage.py runserver
```

### Step 4 — Open browser
```
http://127.0.0.1:8000
```

### Step 5 — Run analysis
- Select a city (Delhi, Mumbai, Chennai, etc.)
- Click **⚡ Run Full Analysis**
- Watch all 6 charts generate in real-time!

---

## 📊 6 Charts Generated

| # | Chart | What It Shows |
|---|-------|---------------|
| 1 | **Temperature Trend + Forecast** | Annual temp + Polynomial regression curve + 2050 prediction |
| 2 | **City Heatmap** | Monthly temperature grid for 8 major cities |
| 3 | **Decade Comparison** | Violin plots showing temp distribution shift per decade |
| 4 | **Multi-City Warming Race** | All cities' warming anomaly on one plot vs 1.5°C Paris threshold |
| 5 | **Precipitation Analysis** | Monsoon patterns + Temperature vs Rainfall correlation |
| 6 | **Model Comparison** | Linear vs Poly°3 vs Poly°5 with R² scores |

---

## 🧠 Machine Learning Concepts Used

- **Polynomial Regression** (degree 1, 3, 5) via `sklearn.preprocessing.PolynomialFeatures`
- **Linear Regression** via `sklearn.linear_model.LinearRegression`
- **Ridge Regression** for precipitation correlation
- **R² Score** evaluation via `sklearn.metrics.r2_score`
- **Sklearn Pipelines** combining preprocessor + model
- **NumPy** for array ops, trend computation, synthetic data generation
- **Pandas** for time-series aggregation, rolling means, groupby operations

---

## 📡 Data Source

**Open-Meteo Historical API** — Free, open-source, no API key required
- ERA5 Reanalysis dataset (ECMWF)
- Data from 1940 to present
- 9 km spatial resolution
- URL: `https://archive-api.open-meteo.com/v1/archive`

The app has a **smart fallback** — if the API is unavailable (firewall, offline), it generates realistic synthetic data with actual city-specific temperature profiles and a 0.18°C/decade warming trend based on IPCC reports.

---

## 🎨 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Django 4.x |
| ML | Scikit-learn, NumPy, Pandas |
| Charts | Matplotlib, Seaborn |
| Data | Open-Meteo ERA5 API |
| Frontend | HTML/CSS/JS (no frameworks needed) |
| Fonts | Syne, Space Grotesk, JetBrains Mono |

---

## 💡 Key Academic Insights

1. **Why Polynomial > Linear for Climate**: Warming acceleration is non-linear due to greenhouse gas feedback loops. Polynomial regression captures this curve.
2. **Why R² matters**: Higher R² = model explains more variance. Poly°3 consistently outperforms Linear.
3. **Paris Agreement context**: The 1.5°C threshold line is shown on the multi-city chart — several Indian cities are already approaching it.
4. **Urban Heat Islands**: Cities like Delhi and Ahmedabad warm faster than coastal cities (Mumbai, Chennai) due to urbanisation.

---

*Built with ❤️ using free, open-source tools only.*
