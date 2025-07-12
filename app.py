
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import timedelta

st.set_page_config(page_title="Funding Rate APR Tool", layout="wide")
st.title("📈 Funding Rate APR Calculator")

st.markdown("""
Upload funding data from two exchanges and calculate net APR for delta-neutral strategies.
Select which is long/short, and the app handles the rest — including interval normalization.
""")

# Step 1: File Upload
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload first exchange file", type=["csv", "xlsx"], key="file1")
with col2:
    file2 = st.file_uploader("Upload second exchange file", type=["csv", "xlsx"], key="file2")

if file1 and file2:

    def load_file(file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

    df1 = load_file(file1)
    df2 = load_file(file2)

    st.markdown("### 🔧 Select Roles and Settings")
    col1, col2 = st.columns(2)
    with col1:
        role1 = st.selectbox("Role for File 1", ["Long", "Short"], key="r1")
        exch1 = st.selectbox("Exchange for File 1", ["Bybit", "WooX", "Other"], key="e1")
    with col2:
        role2 = st.selectbox("Role for File 2", ["Long", "Short"], key="r2")
        exch2 = st.selectbox("Exchange for File 2", ["Bybit", "WooX", "Other"], key="e2")

    def normalize(df, exchange):
        df = df.copy()
        time_col = [c for c in df.columns if "time" in c.lower()][0]
        rate_col = [c for c in df.columns if "funding" in c.lower()][0]
        df[time_col] = pd.to_datetime(df[time_col])

        # Clean funding rate
        if exchange == "Bybit":
            df[rate_col] = df[rate_col].astype(float) * 100
            interval = 8
        elif exchange == "WooX":
            df[rate_col] = df[rate_col].astype(str).str.replace('%','').astype(float)
            interval = int(df.get('funding_interval', '4h').astype(str).str.replace('h','').iloc[0])
        else:
            # Try to infer from time gaps
            df = df.sort_values(by=time_col)
            interval = int(df[time_col].diff().median() / timedelta(hours=1))
            df[rate_col] = df[rate_col].astype(str).str.replace('%','').astype(float)

        return df[[time_col, rate_col]].rename(columns={time_col: 'time', rate_col: 'funding_rate'}), interval

    df_long, df_short = (df1, df2) if role1 == "Long" else (df2, df1)
    exch_long, exch_short = (exch1, exch2) if role1 == "Long" else (exch2, exch1)

    df_long, interval_long = normalize(df_long, exch_long)
    df_short, interval_short = normalize(df_short, exch_short)

    # Merge
    merged = pd.merge(df_long, df_short, on='time', suffixes=('_long', '_short'))

    # APR calculation per funding rule
    def compute_apr(row):
        r_long = -row['funding_rate_long'] if row['funding_rate_long'] > 0 else abs(row['funding_rate_long'])
        r_short = row['funding_rate_short'] if row['funding_rate_short'] > 0 else -abs(row['funding_rate_short'])
        apr_long = r_long * (24 / interval_long) * 365
        apr_short = r_short * (24 / interval_short) * 365
        return apr_long + apr_short

    merged['net_apr'] = merged.apply(compute_apr, axis=1)

    # Daily chart
    merged['date'] = merged['time'].dt.date
    daily_apr = merged.groupby('date')['net_apr'].mean()

    st.markdown("### 📊 Net APR Over Time (Daily Average)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(daily_apr.index, daily_apr.values, marker='o')
    ax.axhline(daily_apr.mean(), linestyle='--', color='red', label='30d Avg')
    ax.set_ylabel("APR (%)")
    ax.set_xlabel("Date")
    ax.set_title("Daily Net APR")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Summary APR
    st.markdown("### 📈 Average APR Summary")
    latest = merged['time'].max()
    for d in [30, 14, 7, 3, 1]:
        avg = merged[merged['time'] >= latest - pd.Timedelta(days=d)]['net_apr'].mean()
        st.write(f"**Last {d}d**: {avg:.2f}% APR")
