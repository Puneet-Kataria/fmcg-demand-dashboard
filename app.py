import sys
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 FMCG Demand Dashboard")

# Load data
data = pd.read_csv("fmcg_regional_data.csv")

# Convert date
data['date'] = pd.to_datetime(data['date']).dt.date

# Sidebar filters
region = st.sidebar.selectbox("Select Region", data['Region'].unique())
category = st.sidebar.selectbox("Select Category", data['Category'].unique())

# Filter data
filtered = data[(data['Region'] == region) & (data['Category'] == category)]

# Plot
fig = px.line(filtered, x='date', y='Demand Index', title=f"{category} in {region}")

st.plotly_chart(fig)

# Show data
st.write(filtered.head())

# Forecast
from statsmodels.tsa.arima.model import ARIMA

df = filtered.sort_values('date')
df.set_index('date', inplace=True)

model = ARIMA(df['Demand Index'], order=(1,0,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=6)

st.subheader("📈 Forecast (Next 6 Months)")
st.write(forecast)

#forecast graph
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Demand Index'],
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=forecast.index,
    y=forecast,
    name="Forecast"
))

st.plotly_chart(fig)

#KPIs
col1, col2 = st.columns(2)

col1.metric("Latest Demand", round(df['Demand Index'].iloc[-1], 2))
col2.metric("Avg Demand", round(df['Demand Index'].mean(), 2))

#Multi-category comparison
multi_category = st.multiselect("Compare Categories", data['Category'].unique())
if multi_category:
    compare_df = data[
        (data['Region'] == region) &
        (data['Category'].isin(multi_category))
    ]

    fig2 = px.line(compare_df, x='date', y='Demand Index', color='Category',
                   title="Category Comparison")

    st.plotly_chart(fig2)
 
# insights
recent = filtered['Demand Index'].tail(6)
previous = filtered['Demand Index'].iloc[-12:-6]

recent_avg = recent.mean()
previous_avg = previous.mean()

if previous_avg != 0:
    change = ((recent_avg - previous_avg) / previous_avg) * 100
else:
    change = 0

st.subheader("📊 Insights")

# Trend insight
if change > 5:
    st.success(f"Demand increased by {round(change,2)}% in last 6 months 📈")
elif change < -5:
    st.error(f"Demand decreased by {round(abs(change),2)}% in last 6 months 📉")
else:
    st.info("Demand is stable ⚖️")

# Volatility insight
volatility = filtered['Demand Index'].std()

if volatility > 10:
    st.warning("Demand is highly volatile ⚠️")
else:
    st.success("Demand is relatively stable ✅")

# Peak month insight
peak = filtered.loc[filtered['Demand Index'].idxmax()]

st.write(f"📅 Peak demand observed in {peak['date'].strftime('%b %Y')} with value {round(peak['Demand Index'],2)}")
