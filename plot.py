import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_regimes():
    # 1. Load the results we just saved
    df = pd.read_csv("regime_analysis_results.csv", index_col=0, parse_dates=True)
    
    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df.index, df['Close'], color='black', linewidth=1, label='SPY Price')
    
    # 3. Color the background based on Regime
    # We identify standard "Bull" days. 
    # (Check your CSV to see if Bull is 0 or 1. usually the code sets Bull='Bull')
    
    # Helper to find continuous regions
    y_min, y_max = ax.get_ylim()
    
    # Iterate through the data to paint "Bear" regions Red
    # (We assume the default background is white/greenish, so we highlight the crash)
    bear_dates = df[df['regime'] == 'Bear'].index
    
    if len(bear_dates) > 0:
        for date in bear_dates:
            ax.axvspan(date, date + pd.Timedelta(days=1), color='red', alpha=0.3, lw=0)

    # 4. Styling
    ax.set_title("SPY Market Regimes (HMM Detected)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format Date Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # 5. Save
    plt.tight_layout()
    plt.savefig("regime_chart.png", dpi=300)
    print("Chart saved as 'regime_chart.png'")

if __name__ == "__main__":
    plot_regimes()