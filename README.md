# 📊 FMCG Demand Dashboard

A live FMCG demand analytics dashboard built with Python and Streamlit.

## What it does
- Tracks regional FMCG demand trends across 4 Indian markets (North, South, East, West)
- Forecasts demand 6 months ahead using SARIMA time-series modelling
- Validates forecast accuracy using MAPE (holdout validation)
- Generates AI-powered business insights using GPT-4o-mini
- Flags data quality issues transparently

## Built with
Python, Streamlit, Statsmodels (SARIMA), Plotly, OpenAI API, scikit-learn

## Data Source
Google Trends search interest index for India (2021–2026).
Demand Index represents relative search volume (0–100), not absolute sales figures.

## Live App
https://fmcg-dashboard-puneet.streamlit.app/
