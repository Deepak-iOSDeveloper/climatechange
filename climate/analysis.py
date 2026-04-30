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
from scipy import stats
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
            'cloudcover','precipMM','FeelsLikeC','DewPointC','uvIndex','visibility',
            'WindGustKmph','sunHour']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    temp_corr = corr['tempC'].drop('tempC')
    sorted_corr = temp_corr.abs().sort_values(ascending=False)
    top_feature = sorted_corr.index[0]
    top_value = temp_corr[top_feature]
    print(f"\n🔥 Most correlated feature with tempC: {top_feature} ({top_value:.2f})")
    print("\n📊 Top 5 correlations with tempC:")
    print(temp_corr.loc[sorted_corr.index].head(5))
    fig, ax = plt.subplots(figsize=(13,10))
    _sf(fig)
    ax.set_facecolor(CARD)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(145, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap=cmap, annot=True, fmt='.2f',
        annot_kws={'size':8,'color':'white','weight':'bold'},
        linewidths=0.5, linecolor=BG, vmin=-1, vmax=1,
        cbar_kws={'label':'Correlation','shrink':0.8}
    )
    x_idx = cols.index(top_feature)
    y_idx = cols.index('tempC')
    ax.text(x_idx + 0.5, y_idx + 0.5, "★", ha='center', va='center', color='yellow', fontsize=16, fontweight='bold')
    ax.set_title('🔥 Feature Correlation Heatmap — All Climate Variables', color=TEXT, fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors=TEXT, labelsize=9)
    cb = ax.collections[0].colorbar
    cb.set_label('Correlation', color=MUTED, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
    cb.outline.set_color(BORDER)
    plt.tight_layout()
    _save(fig, 'heatmap')

# CHART 5 — Box Plot
def chart_boxplot(df):
    fig, axes = plt.subplots(1, 3, figsize=(16,6)); _sf(fig)
    for ax in axes:
        ax.set_facecolor(CARD)
    sns.boxplot(data=df, x='city', y='tempC', ax=axes[0], palette=CITY_COLORS)
    _sa(axes[0], '🌡 Temperature Distribution by City (Outliers)', 'City', 'Temp (°C)')
    axes[0].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df, y='humidity', ax=axes[1], color=T2)
    _sa(axes[1], '💧 Humidity Distribution (Outliers)', '', 'Humidity (%)')
    sns.boxplot(data=df, y='windspeedKmph', ax=axes[2], color=T3)
    _sa(axes[2], '💨 Wind Speed Distribution (Outliers)', '', 'Wind Speed (km/h)')
    plt.tight_layout()
    fig.suptitle('BOX PLOT ANALYSIS — Outlier Detection Across Climate Features', color=T3, fontsize=10, fontweight='bold', y=1.01)
    _save(fig, 'boxplot')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
#
# Four hypothesis tests are run:
#
#   TEST 1 — Independent T-Test (Two cities)
#     H0: Mean temperature of Delhi == Mean temperature of Bengaluru
#     H1: Mean temperatures are significantly different
#
#   TEST 2 — One-Way ANOVA (All 8 cities)
#     H0: Mean temperature is the same across all 8 Indian cities
#     H1: At least one city has a significantly different mean temperature
#
#   TEST 3 — Mann-Whitney U Test (Non-parametric, two seasons)
#     H0: Temperature distribution in Monsoon == Summer (Spring)
#     H1: The two season distributions differ significantly
#
#   TEST 4 — Chi-Square Test (Season vs City — frequency table)
#     H0: Season distribution is independent of city
#     H1: Season and city are NOT independent (association exists)
#
#   Alpha level: 0.05 for all tests
# ══════════════════════════════════════════════════════════════════════════════

def chart_hypothesis(df):
    ALPHA = 0.05
    results = {}

    # ── TEST 1: Independent T-Test ─────────────────────────────────────────
    delhi     = df[df['city'] == 'Delhi']['tempC'].dropna()
    bengaluru = df[df['city'] == 'Bengaluru']['tempC'].dropna()
    t_stat, t_p = stats.ttest_ind(delhi, bengaluru, equal_var=False)  # Welch's T-test
    results['ttest'] = {
        'name'   : "T-Test: Delhi vs Bengaluru",
        'H0'     : "H₀: Mean temp of Delhi = Mean temp of Bengaluru",
        'H1'     : "H₁: Mean temps are significantly different",
        'stat'   : round(t_stat, 4),
        'p_value': round(t_p, 6),
        'reject' : t_p < ALPHA,
        'color'  : T4 if t_p < ALPHA else T1,
    }

    # ── TEST 2: One-Way ANOVA ──────────────────────────────────────────────
    city_groups = [df[df['city'] == c]['tempC'].dropna() for c in CITIES]
    f_stat, f_p = stats.f_oneway(*city_groups)
    results['anova'] = {
        'name'   : "ANOVA: All 8 Cities",
        'H0'     : "H₀: Mean temp is equal across all 8 cities",
        'H1'     : "H₁: At least one city has different mean temp",
        'stat'   : round(f_stat, 4),
        'p_value': round(f_p, 6),
        'reject' : f_p < ALPHA,
        'color'  : T4 if f_p < ALPHA else T1,
    }

    # ── TEST 3: Mann-Whitney U (Non-parametric) ────────────────────────────
    monsoon = df[df['season'] == 'Monsoon']['tempC'].dropna()
    spring  = df[df['season'] == 'Spring']['tempC'].dropna()
    u_stat, u_p = stats.mannwhitneyu(monsoon, spring, alternative='two-sided')
    results['mannwhitney'] = {
        'name'   : "Mann-Whitney: Monsoon vs Spring",
        'H0'     : "H₀: Temp distribution in Monsoon = Spring",
        'H1'     : "H₁: Distributions are significantly different",
        'stat'   : round(u_stat, 4),
        'p_value': round(u_p, 6),
        'reject' : u_p < ALPHA,
        'color'  : T4 if u_p < ALPHA else T1,
    }

    # ── TEST 4: Chi-Square Test ────────────────────────────────────────────
    contingency = pd.crosstab(df['city'], df['season'])
    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)
    results['chisquare'] = {
        'name'   : "Chi-Square: City vs Season",
        'H0'     : "H₀: Season distribution is independent of city",
        'H1'     : "H₁: Season distribution depends on city",
        'stat'   : round(chi2_stat, 4),
        'p_value': round(chi2_p, 6),
        'reject' : chi2_p < ALPHA,
        'color'  : T4 if chi2_p < ALPHA else T1,
    }

    # ── PLOT ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14)); _sf(fig)
    fig.suptitle(
        '🔬 HYPOTHESIS TESTING — Statistical Significance of Climate Differences (α = 0.05)',
        color=T1, fontsize=13, fontweight='bold', y=0.98
    )

    # Row 1: Distribution plots for T-Test and Mann-Whitney
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    # Row 2: Bar chart of p-values and verdict table
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    # Row 3: City mean temps (ANOVA context) and season boxplot
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    # ── Subplot 1: T-Test KDE — Delhi vs Bengaluru ────────────────────────
    ax1.set_facecolor(CARD)
    delhi_s   = delhi.sample(min(5000, len(delhi)), random_state=42)
    beng_s    = bengaluru.sample(min(5000, len(bengaluru)), random_state=42)
    sns.kdeplot(delhi_s,     ax=ax1, color=T4,  fill=True, alpha=0.4, label=f'Delhi  (μ={delhi.mean():.1f}°C)')
    sns.kdeplot(beng_s,      ax=ax1, color=T3,  fill=True, alpha=0.4, label=f'Bengaluru (μ={bengaluru.mean():.1f}°C)')
    verdict1 = "✅ Reject H₀" if results['ttest']['reject'] else "❌ Fail to Reject H₀"
    ax1.axvline(delhi.mean(),     color=T4, linestyle='--', linewidth=1.5)
    ax1.axvline(bengaluru.mean(), color=T3, linestyle='--', linewidth=1.5)
    _sa(ax1, f"T-Test: Delhi vs Bengaluru\np={results['ttest']['p_value']} → {verdict1}", 'Temp (°C)', 'Density')
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    # ── Subplot 2: Mann-Whitney KDE — Monsoon vs Spring ───────────────────
    ax2.set_facecolor(CARD)
    mon_s = monsoon.sample(min(5000, len(monsoon)), random_state=42)
    spr_s = spring.sample(min(5000, len(spring)),   random_state=42)
    sns.kdeplot(mon_s, ax=ax2, color=T2,  fill=True, alpha=0.4, label=f'Monsoon (μ={monsoon.mean():.1f}°C)')
    sns.kdeplot(spr_s, ax=ax2, color=T5,  fill=True, alpha=0.4, label=f'Spring  (μ={spring.mean():.1f}°C)')
    verdict3 = "✅ Reject H₀" if results['mannwhitney']['reject'] else "❌ Fail to Reject H₀"
    ax2.axvline(monsoon.mean(), color=T2, linestyle='--', linewidth=1.5)
    ax2.axvline(spring.mean(),  color=T5, linestyle='--', linewidth=1.5)
    _sa(ax2, f"Mann-Whitney: Monsoon vs Spring\np={results['mannwhitney']['p_value']} → {verdict3}", 'Temp (°C)', 'Density')
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    # ── Subplot 3: p-value comparison bar chart ────────────────────────────
    ax3.set_facecolor(CARD)
    test_labels = ['T-Test\n(Delhi vs\nBengaluru)', 'ANOVA\n(8 Cities)', 'Mann-Whitney\n(Monsoon vs\nSpring)', 'Chi-Square\n(City vs\nSeason)']
    p_vals  = [results[k]['p_value'] for k in ['ttest','anova','mannwhitney','chisquare']]
    colors  = [results[k]['color']   for k in ['ttest','anova','mannwhitney','chisquare']]
    p_plot  = [min(p, 0.5) for p in p_vals]   # cap for display (very small p → invisible bar)
    bars = ax3.bar(test_labels, p_plot, color=colors, width=0.5, edgecolor=BORDER, linewidth=0.8)
    ax3.axhline(ALPHA, color='white', linewidth=1.8, linestyle='--', label=f'α = {ALPHA}')
    for bar, pv in zip(bars, p_vals):
        label = f'p={pv:.2e}' if pv < 0.001 else f'p={pv:.4f}'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 label, ha='center', color=TEXT, fontsize=7.5, fontweight='bold')
    _sa(ax3, '📊 p-value Comparison Across All Tests\n(bar below dashed line → Reject H₀)', 'Test', 'p-value')
    ax3.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    # ── Subplot 4: Summary verdict table ──────────────────────────────────
    ax4.set_facecolor(CARD)
    ax4.axis('off')
    table_data = []
    for k in ['ttest','anova','mannwhitney','chisquare']:
        r = results[k]
        verdict = "REJECT H₀ ✅" if r['reject'] else "FAIL TO REJECT ❌"
        sig     = "Significant" if r['reject'] else "Not Significant"
        table_data.append([r['name'], f"{r['p_value']:.2e}", sig, verdict])
    col_labels = ['Test', 'p-value', 'Result', 'Verdict']
    tbl = ax4.table(
        cellText=table_data, colLabels=col_labels,
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(CARD2 if row > 0 else DIM)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=TEXT, fontweight='bold' if row == 0 else 'normal')
    ax4.set_title('📋 Hypothesis Test Summary Table', color=TEXT, fontsize=11, fontweight='bold', pad=12)

    # ── Subplot 5: ANOVA context — city mean temps ─────────────────────────
    ax5.set_facecolor(CARD)
    city_means = df.groupby('city')['tempC'].mean().reindex(CITIES)
    city_ci    = df.groupby('city')['tempC'].sem().reindex(CITIES) * 1.96   # 95% CI
    bars5 = ax5.bar(CITIES, city_means, color=CITY_COLORS, width=0.6, edgecolor=BORDER, linewidth=0.7)
    ax5.errorbar(CITIES, city_means, yerr=city_ci, fmt='none', color='white', capsize=4, linewidth=1.5)
    grand_mean = df['tempC'].mean()
    ax5.axhline(grand_mean, color=T4, linewidth=2, linestyle='--', label=f'Grand Mean = {grand_mean:.1f}°C')
    for bar, val in zip(bars5, city_means):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}°C', ha='center', color=TEXT, fontsize=7.5, fontweight='bold')
    verdict2 = "✅ Reject H₀" if results['anova']['reject'] else "❌ Fail to Reject H₀"
    _sa(ax5, f"ANOVA: Mean Temp per City (F={results['anova']['stat']}, {verdict2})", 'City', 'Mean Temp (°C)')
    ax5.tick_params(axis='x', rotation=30, colors=MUTED)
    ax5.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    # ── Subplot 6: Season boxplot (Mann-Whitney context) ──────────────────
    ax6.set_facecolor(CARD)
    season_order = ['Winter','Spring','Monsoon','Autumn']
    season_colors = [T3, T5, T2, T4]
    sns.boxplot(
        data=df[df['season'].isin(season_order)],
        x='season', y='tempC', order=season_order,
        palette=season_colors, ax=ax6, width=0.5
    )
    verdict3b = "✅ Reject H₀" if results['mannwhitney']['reject'] else "❌ Fail to Reject H₀"
    _sa(ax6, f"Season-wise Temp Distribution\n(Mann-Whitney Monsoon vs Spring → {verdict3b})", 'Season', 'Temp (°C)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'hypothesis')

    # Return structured results for views.py to use
    return {k: {
        'name'   : v['name'],
        'H0'     : v['H0'],
        'H1'     : v['H1'],
        'stat'   : float(v['stat']),
        'p_value': float(v['p_value']),
        'reject' : int(bool(v['reject'])),     # 1/0 — always JSON serializable
        'verdict': "Reject H₀ — Statistically Significant" if v['reject']
                   else "Fail to Reject H₀ — Not Statistically Significant",
    } for k, v in results.items()}


def run_charts():
    print("Loading dataset…"); df = load_data(80000)
    print(f"  ✓ {len(df)} rows")
    print("[1/6] Scatter");     chart_scatter(df)
    print("[2/6] Line");        chart_line(df)
    print("[3/6] Bar");         chart_bar(df)
    print("[4/6] Heatmap");     chart_heatmap(df)
    print("[5/6] Boxplot");     chart_boxplot(df)
    print("[6/6] Hypothesis");  hyp = chart_hypothesis(df)
    print("✅ Charts done.")
    return {
        'total_rows'    : 771264,
        'total_cities'  : 8,
        'year_range'    : '2009–2019',
        'mean_temp'     : round(float(df['tempC'].mean()), 2),
        'max_temp'      : round(float(df['maxtempC'].max()), 1),
        'min_temp'      : round(float(df['mintempC'].min()), 1),
        'mean_humidity' : round(float(df['humidity'].mean()), 1),
        'hottest_city'  : df.groupby('city')['tempC'].mean().idxmax(),
        'coolest_city'  : df.groupby('city')['tempC'].mean().idxmin(),
        'hypothesis'    : hyp,  # ← hypothesis results passed to frontend
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
        'algorithm'     : algo,
        'r2'            : r2,
        'rmse'          : rmse,
        'mae'           : mae,
        'importances'   : importances,
        'user_prediction': user_prediction,
        'target'        : target,
        'train_size'    : len(X_train),
        'test_size'     : len(X_test),
        'features_used' : len(features),
        'scaler'        : scaler_t,
        'hyperparams'   : valid,
    }