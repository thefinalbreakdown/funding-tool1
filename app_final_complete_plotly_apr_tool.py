
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

st.set_page_config(page_title="Funding Rate APR Tool", layout="wide")
st.title("📈 Funding Rate APR Calculator")

st.markdown("""
Upload funding data from two exchanges and calculate net APR for delta-neutral strategies.
Supports Binance, KuCoin, WooX, Bybit, and manual uploads.
""")

st.subheader("📁 Upload Funding Rate Files")
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

    st.subheader("🔧 Select Exchange Roles & Settings")
    col1, col2 = st.columns(2)
    with col1:
        role1 = st.selectbox("Role for File 1", ["Long", "Short"], key="r1")
        exch1 = st.selectbox("Exchange for File 1", ["Bybit", "WooX", "Binance", "KuCoin", "Manual"], key="e1")
    with col2:
        role2 = st.selectbox("Role for File 2", ["Long", "Short"], key="r2")
        exch2 = st.selectbox("Exchange for File 2", ["Bybit", "WooX", "Binance", "KuCoin", "Manual"], key="e2")

    if role1 == role2:
        st.error("❌ You must select one exchange as Long and one as Short.")
        st.stop()

    def normalize(df, exchange, label):
        df = df.copy()
        if exchange == "Manual":
            time_col = [c for c in df.columns if "time" in c.lower()][0]
            rate_col = [c for c in df.columns if "funding" in c.lower()][0]
            df[time_col] = pd.to_datetime(df[time_col])
            df[rate_col] = df[rate_col].astype(float)
            df = df.rename(columns={time_col: "time", rate_col: "funding_rate"})
            interval = st.number_input(f"Manual interval (hours) for {label}", min_value=1, value=8, step=1, key=label)
        elif exchange == "Bybit":
            df['time'] = pd.to_datetime(df[df.columns[0]])
            df['funding_rate'] = df[df.columns[2]].astype(float) * 100
            interval = 8
            interval = st.number_input(f"Detected interval: {interval}h — override? ({label})", min_value=1, value=interval, step=1, key=label)
        elif exchange == "WooX":
            df['time'] = pd.to_datetime(df[df.columns[0]])
            df['funding_rate'] = df[df.columns[2]].astype(str).str.replace('%','').astype(float)
            interval = 4
            interval = st.number_input(f"Detected interval: {interval}h — override? ({label})", min_value=1, value=interval, step=1, key=label)
        elif exchange == "Binance":
            df['time'] = pd.to_datetime(df['Time'])
            df['funding_rate'] = df['Funding Rate'].astype(str).str.replace('%','').astype(float)
            interval = 8
            interval = st.number_input(f"Detected interval: {interval}h — override? ({label})", min_value=1, value=interval, step=1, key=label)
        elif exchange == "KuCoin":
            df['time'] = pd.to_datetime(df['Time'].astype(str).str.strip())
            df['funding_rate'] = df['Funding Rate'].astype(str).str.replace('%','').str.replace('\r\n','').astype(float)
            interval = 8
            interval = st.number_input(f"Detected interval: {interval}h — override? ({label})", min_value=1, value=interval, step=1, key=label)
        else:
            st.error("Unknown exchange format.")
            st.stop()
        return df[['time', 'funding_rate']], interval

    df_long, df_short = (df1, df2) if role1 == "Long" else (df2, df1)
    exch_long, exch_short = (exch1, exch2) if role1 == "Long" else (exch2, exch1)
    df_long, interval_long = normalize(df_long, exch_long, 'File 1')
    df_short, interval_short = normalize(df_short, exch_short, 'File 2')

    merged = pd.merge(df_long, df_short, on='time', suffixes=('_long', '_short'))

    def compute_apr(row):
        r_long = -row['funding_rate_long'] if row['funding_rate_long'] > 0 else abs(row['funding_rate_long'])
        r_short = row['funding_rate_short'] if row['funding_rate_short'] > 0 else -abs(row['funding_rate_short'])
        apr_long = r_long * (24 / interval_long) * 365
        apr_short = r_short * (24 / interval_short) * 365
        return apr_long, apr_short, apr_long + apr_short

    aprs = merged.apply(compute_apr, axis=1, result_type='expand')
    merged[['apr_long', 'apr_short', 'net_apr']] = aprs
    merged['date'] = merged['time'].dt.date

    st.subheader("⏱ Filter Period")
    days_filter = st.selectbox("Select period to display", [30, 14, 7, 3, 1], format_func=lambda x: f"Last {x} days")
    latest = merged['time'].max()
    filtered = merged[merged['time'] >= latest - pd.Timedelta(days=days_filter)]

    st.subheader("📊 Daily Net APR")
    daily_net = filtered.groupby('date')['net_apr'].mean().reset_index()
    fig1 = px.line(daily_net, x="date", y="net_apr", title="Daily Net APR", markers=True)
    fig1.add_hline(y=daily_net['net_apr'].mean(), line_dash="dash", annotation_text="Average")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Daily APR by Exchange")
    daily_apr = filtered.groupby('date')[['apr_long', 'apr_short']].mean().reset_index()
    fig2 = px.line(daily_apr, x='date', y=['apr_long', 'apr_short'], markers=True)
    fig2.update_layout(title="Daily APR: Long vs Short", yaxis_title="APR (%)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Raw Funding Rate Comparison")
    daily_funding = filtered.groupby('date')[['funding_rate_long', 'funding_rate_short']].mean().reset_index()
    fig3 = px.line(daily_funding, x='date', y=['funding_rate_long', 'funding_rate_short'], markers=True)
    fig3.update_layout(title="Daily Avg Funding Rate (%)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📉 All Raw Funding Rate Events")
    fig4 = px.line(filtered, x='time', y=['funding_rate_long', 'funding_rate_short'], title="All Raw Funding Rates", markers=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📈 All APR Events")
    fig5 = px.line(filtered, x='time', y=['apr_long', 'apr_short', 'net_apr'], title="All APR Events", markers=False)
    fig5.update_traces(hovertemplate="Time: %{x}<br>APR: %{y:.2f}%")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("💰 ROI Calculator")
    start_amt = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
    min_date, max_date = merged['date'].min(), merged['date'].max()
    col1, col2 = st.columns(2)
    with col1:
        roi_start = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        roi_end = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    roi_data = merged[(merged['date'] >= roi_start) & (merged['date'] <= roi_end)]
    daily_apr = roi_data.groupby('date')['net_apr'].mean()
    equity = [start_amt]
    for apr in daily_apr:
        daily_yield = (apr / 100) / 365
        equity.append(equity[-1] * (1 + daily_yield))

    equity = equity[1:]
    roi_df = pd.DataFrame({"Date": daily_apr.index, "Equity": equity})
    fig6 = px.line(roi_df, x='Date', y='Equity', title="ROI Curve", markers=True)
    st.plotly_chart(fig6, use_container_width=True)

    final_val = equity[-1] if equity else start_amt
    st.markdown(f"**Final Portfolio Value:** ${final_val:.2f}")
    st.markdown(f"**Net ROI:** {((final_val / start_amt - 1) * 100):.2f}%")
