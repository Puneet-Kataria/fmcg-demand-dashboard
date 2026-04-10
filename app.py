from openai import OpenAI
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")
import threading
import requests
import time

def keep_alive():
    while True:
        time.sleep(300)  # ping every 5 minutes
        try:
            requests.get("https://fmcg-dashboard-puneet.streamlit.app/")
        except:
            pass

threading.Thread(target=keep_alive, daemon=True).start()
# ------------------ CONFIG ------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide", page_title="FMCG Demand Dashboard")
st.title("📊 FMCG Demand Dashboard")

# ------------------ LOAD DATA (CACHED) ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fmcg_regional_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

data = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.header("Filters")
st.sidebar.caption(
    "📌 Data Source: Google Trends search interest index for India (2021–2026). "
    "'Demand Index' represents relative search volume (0–100), not absolute sales figures."
)

region = st.sidebar.selectbox("Select Region", sorted(data['Region'].unique()))
category = st.sidebar.selectbox("Select Category", sorted(data['Category'].unique()))

# ------------------ FILTER DATA ------------------
filtered = data[(data['Region'] == region) & (data['Category'] == category)].copy()
filtered = filtered.sort_values('date')

if filtered.empty:
    st.error("No data available for this combination. Please try another region or category.")
    st.stop()

# ------------------ HEADER ------------------
st.markdown(f"### 📍 {category} in {region}")

# ------------------ MAIN TREND ------------------
st.subheader("📈 Demand Trend")

fig = px.line(
    filtered,
    x='date',
    y='Demand Index',
    labels={'date': 'Date', 'Demand Index': 'Demand Index (Google Trends)'}
)
fig.update_traces(line_color='#00b4d8')
st.plotly_chart(fig, use_container_width=True)

# ------------------ KPIs ------------------
col1, col2, col3 = st.columns(3)

latest = round(filtered['Demand Index'].iloc[-1], 2)
avg = round(filtered['Demand Index'].mean(), 2)
peak_row = filtered.loc[filtered['Demand Index'].idxmax()]
peak_val = round(peak_row['Demand Index'], 2)
peak_date = peak_row['date'].strftime('%b %Y')

col1.metric("Latest Demand", latest)
col2.metric("Average Demand", avg)
col3.metric("Peak Demand", f"{peak_val} ({peak_date})")

st.markdown("---")

# ------------------ FORECAST (SARIMA) ------------------
st.subheader("🔮 Demand Forecast (Next 6 Months)")

df_model = filtered.set_index('date')[['Demand Index']].copy()

# Split: last 6 months for validation, rest for training
train = df_model.iloc[:-6]
test = df_model.iloc[-6:]

try:
    # SARIMA(1,1,1)(1,1,0,12) - captures trend + monthly seasonality
    model = SARIMAX(
        train['Demand Index'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)

    # In-sample validation MAPE
    test_pred = model_fit.forecast(steps=6)
    test_actual = test['Demand Index'].values
    test_pred_values = test_pred.values

    # Only compute MAPE if no zeros in actuals
    non_zero = test_actual != 0
    if non_zero.sum() > 0:
        mape = mean_absolute_percentage_error(
            test_actual[non_zero],
            test_pred_values[non_zero]
        ) * 100
        mape_str = f"{round(mape, 1)}%"
    else:
        mape_str = "N/A"

    # Now refit on full data and forecast next 6 months
    model_full = SARIMAX(
        df_model['Demand Index'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_full_fit = model_full.fit(disp=False)
    forecast = model_full_fit.forecast(steps=6).clip(lower=0)

    # Show model accuracy badge
    st.caption(f"🤖 Model: SARIMA(1,1,1)(1,1,0,12) &nbsp;|&nbsp; Forecast MAPE (6-month holdout): **{mape_str}**")
    st.info("ℹ️ High MAPE is expected for search trend data due to irregular spikes. "
        "SARIMA captures seasonality but cannot predict viral or event-driven demand surges.")

    # Forecast chart
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df_model.index,
        y=df_model['Demand Index'],
        name="Actual",
        line=dict(color='#00b4d8')
    ))

    fig2.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        name="Forecast",
        line=dict(color='#f77f00', dash='dot'),
        mode='lines+markers'
    ))

    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand Index",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Forecast table — formatted dates
    forecast_df = pd.DataFrame({
        'Month': forecast.index.strftime('%b %Y'),
        'Predicted Demand Index': forecast.values.round(2)
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    forecast_direction = forecast.iloc[-1] - forecast.iloc[0]
    forecast_avg = forecast.mean()
    current_avg = df_model['Demand Index'].tail(6).mean()
    forecast_change = ((forecast_avg - current_avg) / current_avg * 100) if current_avg != 0 else 0
    threshold = df_model['Demand Index'].std() * 0.10
    # PRIMARY: Use forecast_change vs current as the main signal
        # SECONDARY: Use forecast_direction as tiebreaker only when change is small
        if abs(forecast_change) > 10:
            # Large change vs current — use this as primary signal
            if forecast_change > 0:
                trend_emoji = "📈"
                trend_text = "rising"
            else:
                trend_emoji = "📉"
                trend_text = "declining"
        else:
            # Small change vs current — use direction within forecast period
            if forecast_direction > threshold:
                trend_emoji = "📈"
                trend_text = "rising"
            elif forecast_direction < -threshold:
                trend_emoji = "📉"
                trend_text = "declining"
            else:
                trend_emoji = "➡️"
                trend_text = "stable"

        st.info(
            f"{trend_emoji} **Forecast Summary:** Demand for **{category}** in **{region}** "
            f"is expected to be **{trend_text}** over the next 6 months. "
            f"Forecasted average demand index: **{round(forecast_avg, 2)}** vs "
            f"current 6-month average of **{round(current_avg, 2)}** "
            f"({'▲' if forecast_change > 0 else '▼'} {abs(round(forecast_change, 1))}% change)."
        )

        # Fix 2 — Data quality warning for zero-heavy categories
        zero_pct = (df_model['Demand Index'] == 0).mean() * 100
        if zero_pct > 30:
            st.warning(
                f"⚠️ Data Quality Notice: {round(zero_pct, 1)}% of historical values "
                f"for **{category}** in **{region}** are zero. This is likely due to low "
                f"search volume relative to other categories. Forecast and interpretation "
                f"should be treated with caution."
            )

except Exception as e:
    st.warning(f"Forecasting failed for this combination: {e}")

st.markdown("---")

# ------------------ INSIGHTS ------------------
st.subheader("📊 Key Insights (Selected Category)")

recent = df_model['Demand Index'].tail(6)
previous = df_model['Demand Index'].iloc[-12:-6]

recent_avg = recent.mean()
previous_avg = previous.mean()

change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg != 0 else 0
volatility = df_model['Demand Index'].std()

# FIX: Trend and volatility use SAME metric — avoid contradicting signals
# Show trend
if change > 5:
    st.success(f"📈 Demand increased by {round(change, 2)}% in the last 6 months vs previous 6 months.")
elif change < -5:
    st.error(f"📉 Demand decreased by {round(abs(change), 2)}% in the last 6 months vs previous 6 months.")
else:
    st.info("⚖️ Demand is stable — less than 5% change in the last 6 months.")

# Show volatility separately and clearly
if volatility > 10:
    st.warning(
        f"⚠️ High volatility detected (std dev: {round(volatility, 2)}). "
        "Demand fluctuates significantly — this is separate from the short-term trend above."
    )
else:
    st.success(
        f"✅ Low volatility (std dev: {round(volatility, 2)}). "
        "Demand is relatively consistent over the full period."
    )

st.write(f"📅 Peak demand observed in **{peak_date}** with index value **{peak_val}**")

st.markdown("---")

# ------------------ AI INSIGHTS ------------------
st.subheader("🤖 AI Insights (Advanced Analysis)")

summary = f"""
Category: {category}
Region: {region}
Recent Avg Demand (last 6 months): {round(recent_avg, 2)}
Previous Avg Demand (prior 6 months): {round(previous_avg, 2)}
Change: {round(change, 2)}%
Volatility (std dev over full period): {round(volatility, 2)}
Peak Demand: {peak_val} in {peak_date}
Note: Demand Index is based on Google Trends relative search interest (0-100 scale) for India.
"""

if st.button("Generate AI Insights"):
    with st.spinner("Analyzing data with AI..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior FMCG business analyst specializing in the Indian market."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze the following FMCG demand data and provide:

1. Trend Insight — what the numbers suggest about consumer interest
2. Risk or Concern — key risks based on volatility and trend
3. Business Recommendation — specific to Indian FMCG context
4. One Actionable Strategy — concrete, implementable next step

Data:
{summary}
"""
                    }
                ]
            )

            result = response.choices[0].message.content

            st.success("Analysis complete!")
            st.markdown("### 📈 Key Insights")
            st.write(result)

            st.download_button(
                label="📥 Download Insights",
                data=result,
                file_name=f"ai_insights_{region}_{category}.txt"
            )

        except Exception as e:
            st.error(f"AI analysis failed: {e}. Please check your API key or try again.")

st.markdown("---")

# ------------------ COMPARISON ------------------
st.subheader("📊 Multi-Category Comparison (Optional)")

multi_category = st.multiselect(
    "Select categories to compare",
    options=sorted(data['Category'].unique()),
    help="Compare demand trends across multiple categories in the selected region."
)

if multi_category:
    compare_df = data[
        (data['Region'] == region) &
        (data['Category'].isin(multi_category))
    ]

    # Distinct color palette — avoid same-shade blues
    color_palette = ['#00b4d8', '#f77f00', '#06d6a0', '#e63946', '#9b5de5']
    color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(multi_category)}

    fig3 = px.line(
        compare_df,
        x='date',
        y='Demand Index',
        color='Category',
        color_discrete_map=color_map,
        labels={'date': 'Date', 'Demand Index': 'Demand Index (Google Trends)'}
    )

    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ------------------ RAW DATA ------------------
with st.expander("📁 View Raw Data"):
    st.caption("Showing filtered data for selected Region and Category.")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
