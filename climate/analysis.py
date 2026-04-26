import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os, warnings, inspect
warnings.filterwarnings('ignore')

# ── Teal-Forest theme ─────────────────────────────────────────────────────────
BG=    "#0a0f0d"; CARD=  "#111816"; CARD2= "#182420"; BORDER="#1e2e28"
T1=    "#2dd4aa"; T2=    "#a3e635"; T3=    "#38bdf8"; T4=    "#fb923c"; T5="#f472b6"
TEXT=  "#ecfdf5"; MUTED= "#6b9e8a"; DIM=   "#2d4a3e"

DATASET = os.path.join(os.path.dirname(__file__), "modified_data.csv")
MEDIA   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media", "charts")
os.makedirs(MEDIA, exist_ok=True)

NUM_FEATURES = [
    'maxtempC','mintempC','totalSnow_cm','sunHour','uvIndex',
    'moon_illumination','DewPointC','FeelsLikeC','HeatIndexC',
    'WindChillC','WindGustKmph','cloudcover','humidity',
    'precipMM','pressure','visibility','winddirDegree','windspeedKmph'
]
TARGET = 'tempC'
CITIES = ['Pune','Bombay','Delhi','Hyderabad','Jaipur','Kanpur','Nagpur','Bengaluru']
CITY_COLORS = [T1,T2,T3,T4,T5,'#c084fc','#fb7185','#fbbf24']

def load_data(sample=80000):
    df = pd.read_csv(DATASET, parse_dates=['date_time','date'])
    df['year']  = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['hour']  = df['date_time'].dt.hour
    df['season']= df['month'].map({
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
        10:'Autumn',11:'Autumn'
    })
    for c in NUM_FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if sample and len(df)>sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    return df

def _sf(fig): fig.patch.set_facecolor(BG)
def _sa(ax, title='', xl='', yl=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    if title: ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    if xl: ax.set_xlabel(xl, color=MUTED, fontsize=8)
    if yl: ax.set_ylabel(yl, color=MUTED, fontsize=8)
    ax.grid(True, color=DIM, linewidth=0.5, alpha=0.7)
def _save(fig, name):
    plt.savefig(os.path.join(MEDIA,f'{name}.png'), dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

# CHART 1 — Scatter
def chart_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(14,6)); _sf(fig)
    cmap = dict(zip(CITIES, CITY_COLORS))
    for city, color in cmap.items():
        sub = df[df['city']==city]
        axes[0].scatter(sub['humidity'], sub['tempC'], c=color, s=3, alpha=0.3, label=city, rasterized=True)
    _sa(axes[0],'🌡  Humidity vs Temperature by City','Humidity (%)','Temperature (°C)')
    axes[0].legend(markerscale=5, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=7, ncol=2)
    for city, color in cmap.items():
        sub = df[df['city']==city]
        axes[1].scatter(sub['windspeedKmph'], sub['FeelsLikeC'], c=color, s=3, alpha=0.3, rasterized=True)
    _sa(axes[1],'💨  Wind Speed vs Feels-Like Temp','Wind Speed (km/h)','Feels Like (°C)')
    plt.tight_layout()
    fig.suptitle('SCATTER ANALYSIS — 771K Hourly Records · 8 Indian Cities · 2009–2019', color=T1, fontsize=10, fontweight='bold', y=1.01)
    _save(fig,'scatter')

# CHART 2 — Line
def chart_line(df):
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(14,9)); _sf(fig)
    annual = df.groupby(['city','year'])['tempC'].mean().reset_index()
    for city, color in zip(CITIES, CITY_COLORS):
        sub = annual[annual['city']==city]
        ax1.plot(sub['year'], sub['tempC'], color=color, linewidth=2.2, marker='o', markersize=4, label=city)
    _sa(ax1,'📈  Annual Mean Temperature Trend (2009–2019)','Year','Mean Temp (°C)')
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, ncol=4)
    monthly_avg = df.groupby(['city','month'])['tempC'].mean().reset_index()
    mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for city, color in zip(CITIES, CITY_COLORS):
        sub = monthly_avg[monthly_avg['city']==city].sort_values('month')
        ax2.plot(sub['month'], sub['tempC'], color=color, linewidth=2, marker='s', markersize=3.5, label=city)
    ax2.set_xticks(range(1,13)); ax2.set_xticklabels(mnames, color=MUTED, fontsize=8)
    _sa(ax2,'📅  Monthly Temperature Profile (avg across 2009–2019)','Month','Mean Temp (°C)')
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, ncol=4)
    plt.tight_layout()
    fig.suptitle('LINE ANALYSIS — Temperature Trends Across Time', color=T2, fontsize=10, fontweight='bold', y=1.01)
    _save(fig,'line')

# CHART 3 — Bar
def chart_bar(df):
    city_stats = df.groupby('city').agg(
        mean_temp=('tempC','mean'), mean_humidity=('humidity','mean'),
        mean_wind=('windspeedKmph','mean'), mean_uv=('uvIndex','mean'),
        mean_precip=('precipMM','mean'),
    ).reset_index()
    fig, axes = plt.subplots(2,3, figsize=(15,8)); _sf(fig); axes=axes.flatten()
    metrics = [
        ('mean_temp','🌡 Mean Temperature (°C)'),
        ('mean_humidity','💧 Mean Humidity (%)'),
        ('mean_wind','💨 Mean Wind Speed (km/h)'),
        ('mean_uv','☀️ Mean UV Index'),
        ('mean_precip','🌧 Mean Precipitation (mm)'),
    ]
    for i,(col,title) in enumerate(metrics):
        bars = axes[i].bar(city_stats['city'], city_stats[col], color=CITY_COLORS, width=0.65, edgecolor=BORDER, linewidth=0.7)
        for bar,val in zip(bars, city_stats[col]):
            axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:.1f}', ha='center', color=TEXT, fontsize=7, fontweight='bold')
        _sa(axes[i], title, 'City', '')
        axes[i].tick_params(axis='x', rotation=30, colors=MUTED)
    season_city = df.groupby(['city','season']).size().unstack(fill_value=0)
    season_city.plot(kind='bar', ax=axes[5], color=[T1,T3,T4,T5], width=0.65, edgecolor=BORDER, linewidth=0.5)
    _sa(axes[5],'🍂 Season Distribution per City','City','Records')
    axes[5].tick_params(axis='x', rotation=30, colors=MUTED)
    axes[5].legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=7)
    plt.tight_layout()
    fig.suptitle('BAR ANALYSIS — City-wise Climate Metrics Comparison', color=T4, fontsize=10, fontweight='bold', y=1.01)
    _save(fig,'bar')

# CHART 4 — Heatmap
def chart_heatmap(df):
    cols = ['tempC','maxtempC','mintempC','humidity','windspeedKmph','pressure',
            'cloudcover','precipMM','FeelsLikeC','DewPointC','uvIndex','visibility','WindGustKmph','sunHour']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(13,10)); _sf(fig); ax.set_facecolor(CARD)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(145, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, ax=ax, cmap=cmap, annot=True, fmt='.2f',
                annot_kws={'size':8,'color':'white','weight':'bold'},
                linewidths=0.5, linecolor=BG, vmin=-1, vmax=1,
                cbar_kws={'label':'Correlation','shrink':0.8})
    ax.set_title('🔥  Feature Correlation Heatmap — All Climate Variables', color=TEXT, fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors=TEXT, labelsize=9)
    cb = ax.collections[0].colorbar
    cb.set_label('Correlation', color=MUTED, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
    cb.outline.set_color(BORDER)
    plt.tight_layout(); _save(fig,'heatmap')

# CHART 5 — Gradient KDE
def chart_gradient(df):
    s = df.sample(min(8000,len(df)), random_state=1)
    fig, axes = plt.subplots(1,3, figsize=(16,6)); _sf(fig)
    for ax in axes: ax.set_facecolor(CARD)
    sns.kdeplot(data=s, x='humidity', y='tempC', ax=axes[0], fill=True, cmap='YlOrRd', thresh=0.05, levels=15)
    _sa(axes[0],'🌈  KDE: Humidity vs Temperature','Humidity (%)','Temp (°C)')
    sns.kdeplot(data=s, x='windspeedKmph', y='pressure', ax=axes[1], fill=True, cmap='GnBu', thresh=0.05, levels=15)
    _sa(axes[1],'🌊  KDE: Wind Speed vs Pressure','Wind Speed (km/h)','Pressure (hPa)')
    sns.kdeplot(data=s, x='DewPointC', y='FeelsLikeC', ax=axes[2], fill=True, cmap='PuRd', thresh=0.05, levels=15)
    _sa(axes[2],'💜  KDE: Dew Point vs Feels-Like','Dew Point (°C)','Feels Like (°C)')
    plt.tight_layout()
    fig.suptitle('GRADIENT DENSITY MAPS — Seaborn KDE (Kernel Density Estimation)', color=T3, fontsize=10, fontweight='bold', y=1.01)
    _save(fig,'gradient')

def run_charts():
    print("Loading dataset…"); df = load_data(80000)
    print(f"  ✓ {len(df)} rows")
    print("[1/5] Scatter"); chart_scatter(df)
    print("[2/5] Line");    chart_line(df)
    print("[3/5] Bar");     chart_bar(df)
    print("[4/5] Heatmap"); chart_heatmap(df)
    print("[5/5] Gradient"); chart_gradient(df)
    print("✅ Charts done.")
    return {
        'total_rows':771264, 'total_cities':8, 'year_range':'2009–2019',
        'mean_temp':round(float(df['tempC'].mean()),2),
        'max_temp':round(float(df['maxtempC'].max()),1),
        'min_temp':round(float(df['mintempC'].min()),1),
        'mean_humidity':round(float(df['humidity'].mean()),1),
        'hottest_city':df.groupby('city')['tempC'].mean().idxmax(),
        'coolest_city':df.groupby('city')['tempC'].mean().idxmin(),
    }

# ── ML PREDICTION ENGINE ─────────────────────────────────────────────────────
ALGO_MAP = {
    'linear': LinearRegression,
    'ridge':  Ridge,
    'lasso':  Lasso,
    'rf':     RandomForestRegressor,
    'gbm':    GradientBoostingRegressor,
    'dt':     DecisionTreeRegressor,
    'knn':    KNeighborsRegressor,
}

def run_prediction(payload: dict) -> dict:
    algo      = payload.get('algorithm','rf')
    hparams   = payload.get('hyperparams',{})
    scaler_t  = payload.get('use_scaler','standard')
    do_impute = payload.get('use_imputer', True)
    do_onehot = payload.get('use_onehot', False)
    input_data= payload.get('input_data',{})
    target    = payload.get('target', TARGET)

    df = load_data(60000)
    features = [f for f in NUM_FEATURES if f != target]

    if do_onehot:
        df = pd.get_dummies(df, columns=['city'], drop_first=True)
        features += [c for c in df.columns if c.startswith('city_')]

    X = df[features].copy()
    y = df[target].copy()
    if do_impute:
        X = X.fillna(X.median())
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = None
    if scaler_t == 'standard': scaler = StandardScaler()
    elif scaler_t == 'minmax':  scaler = MinMaxScaler()
    if scaler:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=features)

    ModelClass = ALGO_MAP.get(algo, RandomForestRegressor)
    sig = inspect.signature(ModelClass.__init__).parameters
    valid = {}
    for k, v in hparams.items():
        if k in sig:
            try: valid[k] = int(v) if '.' not in str(v) else float(v)
            except: valid[k] = v

    model = ModelClass(**valid)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = round(float(r2_score(y_test, y_pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    mae  = round(float(mean_absolute_error(y_test, y_pred)), 4)

    importances = {}
    if hasattr(model,'feature_importances_'):
        top = sorted(zip(features, model.feature_importances_), key=lambda x:-x[1])[:8]
        importances = {k: round(float(v),5) for k,v in top}
    elif hasattr(model,'coef_'):
        top = sorted(zip(features, np.abs(model.coef_)), key=lambda x:-x[1])[:8]
        importances = {k: round(float(v),5) for k,v in top}

    user_prediction = None
    if input_data:
        row = {f: float(input_data.get(f, X[f].median())) for f in features}
        row_df = pd.DataFrame([row])
        if scaler:
            row_df = pd.DataFrame(scaler.transform(row_df), columns=features)
        user_prediction = round(float(model.predict(row_df)[0]), 2)

    # Residual chart
    n = min(500, len(y_test))
    yt = np.array(y_test)[:n]; yp = y_pred[:n]
    fig, (a1,a2) = plt.subplots(1,2, figsize=(12,5)); _sf(fig)
    a1.scatter(yt, yp, color=T1, s=8, alpha=0.4, rasterized=True)
    mn,mx = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    a1.plot([mn,mx],[mn,mx], color=T4, linewidth=2, linestyle='--', label='Perfect fit')
    _sa(a1,'✅  Actual vs Predicted (°C)','Actual','Predicted')
    a1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    residuals = yt - yp
    a2.hist(residuals, bins=40, color=T2, edgecolor=BORDER, linewidth=0.5, alpha=0.85)
    a2.axvline(0, color=T4, linewidth=2, linestyle='--')
    _sa(a2,'📊  Residual Distribution','Residual (°C)','Frequency')
    plt.tight_layout()
    fig.suptitle(f'ML DIAGNOSTICS — {algo.upper()} | R²={r2} | RMSE={rmse}', color=T1, fontsize=10, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(MEDIA,'residuals.png'), dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

    return {
        'algorithm':algo, 'r2':r2, 'rmse':rmse, 'mae':mae,
        'importances':importances, 'user_prediction':user_prediction,
        'target':target, 'train_size':len(X_train), 'test_size':len(X_test),
        'features_used':len(features), 'scaler':scaler_t, 'hyperparams':valid,
    }
