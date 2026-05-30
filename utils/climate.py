import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import numpy as np


def get_climate_data(lat: float, lon: float, start_year: int = 2010, end_year: int = 2023) -> dict:
    """
    Fetch climate data from NASA POWER API.
    T2M           = Temperature at 2 meters (°C)
    PRECTOTCORR   = Precipitation (mm/day)
    RH2M          = Relative Humidity at 2m (%)
    """
    url    = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        'parameters' : 'T2M,PRECTOTCORR,RH2M',
        'community'  : 'RE',
        'longitude'  : lon,
        'latitude'   : lat,
        'start'      : start_year,
        'end'        : end_year,
        'format'     : 'JSON'
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data     = response.json()
        return data['properties']['parameter']
    except Exception as e:
        print(f"NASA POWER API error: {e}")
        return None


def parse_climate_data(raw_data: dict) -> pd.DataFrame:
    """Parse NASA POWER API response into clean DataFrame."""
    temp_data = raw_data.get('T2M', {})
    rain_data = raw_data.get('PRECTOTCORR', {})
    rh_data   = raw_data.get('RH2M', {})

    records = []
    for key in temp_data:
        # Skip annual averages (month 13)
        if key.endswith('13'):
            continue
        try:
            date = pd.to_datetime(key, format='%Y%m')
            records.append({
                'date'        : date,
                'temperature' : temp_data.get(key, np.nan),
                'rainfall'    : rain_data.get(key, np.nan),
                'humidity'    : rh_data.get(key, np.nan)
            })
        except:
            continue

    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)

    # Replace fill values with NaN
    df.replace(-999.0, np.nan, inplace=True)

    return df


def calculate_anomaly(df: pd.DataFrame, baseline_end: str = '2020-01-01') -> pd.DataFrame:
    """Calculate temperature and rainfall anomaly vs baseline period."""
    baseline = df[df['date'] < baseline_end]
    baseline_temp = baseline['temperature'].mean()
    baseline_rain = baseline['rainfall'].mean()

    df['temp_anomaly'] = df['temperature'] - baseline_temp
    df['rain_anomaly'] = df['rainfall']    - baseline_rain

    return df


def plot_climate_trends(df: pd.DataFrame, location_name: str, disaster_date: str = None) -> io.BytesIO:
    """
    Plot temperature and rainfall trends.
    Returns plot as bytes buffer for Streamlit display.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.patch.set_facecolor('#0f1117')

    for ax in axes:
        ax.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    # Temperature trend
    axes[0].plot(df['date'], df['temperature'],
                 color='#e74c3c', linewidth=1.5, alpha=0.8)
    axes[0].fill_between(df['date'], df['temperature'],
                         alpha=0.2, color='#e74c3c')

    # Add trend line
    z = np.polyfit(range(len(df)), df['temperature'].fillna(method='ffill'), 1)
    p = np.poly1d(z)
    axes[0].plot(df['date'], p(range(len(df))),
                 '--', color='white', linewidth=1, alpha=0.5, label='Trend')

    axes[0].set_title(f'Temperature (°C) — {location_name}')
    axes[0].set_ylabel('°C', color='white')
    axes[0].legend(facecolor='#1a1f2e', labelcolor='white')
    axes[0].grid(True, alpha=0.1)

    # Rainfall trend
    axes[1].bar(df['date'], df['rainfall'],
                color='#3498db', alpha=0.7, width=25)
    axes[1].set_title(f'Rainfall (mm/day) — {location_name}')
    axes[1].set_ylabel('mm/day', color='white')
    axes[1].grid(True, alpha=0.1)

    # Temperature anomaly
    colors_anomaly = ['#e74c3c' if x > 0 else '#3498db'
                      for x in df['temp_anomaly'].fillna(0)]
    axes[2].bar(df['date'], df['temp_anomaly'],
                color=colors_anomaly, alpha=0.8, width=25)
    axes[2].axhline(y=0, color='white', linewidth=0.8, alpha=0.5)
    axes[2].set_title(f'Temperature Anomaly (°C) — vs Baseline')
    axes[2].set_ylabel('°C', color='white')
    axes[2].grid(True, alpha=0.1)

    # Add disaster date line to all plots
    if disaster_date:
        d = pd.Timestamp(disaster_date)
        for ax in axes:
            ax.axvline(d, color='#f39c12', linewidth=2,
                       linestyle='--', label='Disaster Date')
            ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=8)

    # Format x axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(f'Climate Vulnerability Analysis — {location_name}',
                 fontsize=14, color='white', fontweight='bold', y=1.01)
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    buf.seek(0)
    plt.close()

    return buf


def get_climate_summary(df: pd.DataFrame, disaster_date: str = None) -> dict:
    """Generate climate summary statistics."""
    summary = {
        'avg_temp'         : round(df['temperature'].mean(), 2),
        'max_temp'         : round(df['temperature'].max(), 2),
        'avg_rainfall'     : round(df['rainfall'].mean(), 2),
        'avg_humidity'     : round(df['humidity'].mean(), 2),
        'temp_trend'       : None,
        'warming_rate'     : None,
    }

    # Calculate warming rate
    if len(df) > 1:
        z = np.polyfit(range(len(df)),
                       df['temperature'].fillna(method='ffill'), 1)
        summary['warming_rate'] = round(z[0] * 12, 4)  # per year
        summary['temp_trend']   = 'warming' if z[0] > 0 else 'cooling'

    # Pre vs post disaster comparison
    if disaster_date:
        d    = pd.Timestamp(disaster_date)
        pre  = df[df['date'] < d]
        post = df[df['date'] >= d]

        if len(pre) > 0 and len(post) > 0:
            summary['pre_disaster_avg_temp']  = round(pre['temperature'].mean(), 2)
            summary['post_disaster_avg_temp'] = round(post['temperature'].mean(), 2)
            summary['temp_change']            = round(
                summary['post_disaster_avg_temp'] - summary['pre_disaster_avg_temp'], 2
            )

    return summary