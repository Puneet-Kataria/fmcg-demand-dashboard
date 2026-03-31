from openai import OpenAI
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# ------------------ CONFIG ------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("📊 FMCG Demand Dashboard")

# ------------------ LOAD DATA ------------------
data = pd.read_csv("fmcg_regional_data.csv")
data['date'] = pd.to_datetime(data['date'])

# ------------------ SIDEBAR ------------------
st.sidebar.header("Filters")

region = st.sidebar.selectbox("Select Region", data['Region'].unique())
category = st.sidebar.selectbox("Select Category", data['Category'].unique())

# ------------------ FILTER DATA ------------------
filtered = data[(data['Region'] == region) & (data['Category'] == category)]

# ------------------ HEADER ------------------
st.markdown(f"### 📍 {category} in {region}")

# ------------------ MAIN TREND ------------------
st.subheader("📈 Demand Trend")

fig = px.line(filtered, x='date', y='Demand Index')
st.plotly_chart(fig, use_container_width=True)

# ------------------ KPIs ------------------
col1, col2 = st.columns(2)

col1.metric("Latest Demand", round(filtered['Demand Index'].iloc[-1], 2))
col2.metric("Average Demand", round(filtered['Demand Index'].mean(), 2))

st.markdown("---")

# ------------------ FORECAST ------------------
st.subheader("🔮 Demand Forecast (Next 6 Months)")

df = filtered.sort_values('date').copy()
df.set_index('date', inplace=True)

model = ARIMA(df['Demand Index'], order=(1, 0, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=6)

# Forecast graph
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index,
    y=df['Demand Index'],
    name="Actual"
))

fig2.add_trace(go.Scatter(
    x=forecast.index,
    y=forecast,
    name="Forecast"
))

st.plotly_chart(fig2, use_container_width=True)

# Forecast table
st.dataframe(forecast)

st.markdown("---")

# ------------------ INSIGHTS ------------------
st.subheader("📊 Key Insights (Selected Category)")

recent = df['Demand Index'].tail(6)
previous = df['Demand Index'].iloc[-12:-6]

recent_avg = recent.mean()
previous_avg = previous.mean()

change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg != 0 else 0

# Trend insight
if change > 5:
    st.success(f"Demand increased by {round(change,2)}% in last 6 months 📈")
elif change < -5:
    st.error(f"Demand decreased by {round(abs(change),2)}% in last 6 months 📉")
else:
    st.info("Demand is stable ⚖️")

# Volatility insight
volatility = df['Demand Index'].std()

if volatility > 10:
    st.warning("Demand is highly volatile ⚠️")
else:
    st.success("Demand is relatively stable ✅")

# Peak demand
peak = df.loc[df['Demand Index'].idxmax()]
st.write(f"📅 Peak demand observed in {peak.name.strftime('%b %Y')} with value {round(peak['Demand Index'],2)}")

st.markdown("---")

# ------------------ AI INSIGHTS ------------------
st.subheader("🤖 AI Insights (Advanced Analysis)")

summary = f"""
Category: {category}
Region: {region}
Recent Avg Demand: {round(recent_avg,2)}
Previous Avg Demand: {round(previous_avg,2)}
Change: {round(change,2)}%
Volatility: {round(volatility,2)}
"""

if st.button("Generate AI Insights"):
    with st.spinner("Analyzing data with AI..."):

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
You are a senior FMCG business analyst.

Analyze the following data and provide:

1. Trend insight  
2. Risk or concern  
3. Business recommendation  
4. One actionable strategy  

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
            file_name="ai_insights.txt"
        )

st.markdown("---")

# ------------------ COMPARISON ------------------
st.subheader("📊 Multi-Category Comparison (Optional)")

multi_category = st.multiselect(
    "Select categories to compare",
    data['Category'].unique()
)

if multi_category:
    compare_df = data[
        (data['Region'] == region) &
        (data['Category'].isin(multi_category))
    ]

    fig3 = px.line(
        compare_df,
        x='date',
        y='Demand Index',
        color='Category'
    )

    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ------------------ RAW DATA ------------------
with st.expander("📁 View Raw Data"):
    st.dataframe(filtered)
