import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import time
import re

st.title("Gapminder Dashboard")
st.write("BIPM Project - Unlocking Lifetimes: Visualizing Progress in Longevity and Poverty Eradication")

def parse_number(value):
    """
    Convert formatted numbers like '3.28M', '407k', '2650' to actual numbers
    """
    if pd.isna(value) or value == '':
        return np.nan
    
    # If it's already a number, return it
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string and clean
    value = str(value).strip()
    
    # Handle different suffixes
    if value.endswith('k'):
        return float(value[:-1]) * 1000
    elif value.endswith('M'):
        return float(value[:-1]) * 1000000
    elif value.endswith('B'):
        return float(value[:-1]) * 1000000000
    else:
        # Try to convert directly
        try:
            return float(value)
        except:
            return np.nan

@st.cache_data
def load_and_process_data():
    # Load files (paths work both locally and in Docker)
    import os
    
    # Check if running in Docker (where files are directly in /app/data)
    if os.path.exists("data/pop.csv"):
        data_path = "data/"
    else:
        data_path = "app/data/"
    
    population_df = pd.read_csv(f"{data_path}pop.csv")
    life_expectancy_df = pd.read_csv(f"{data_path}lex.csv")
    gni_df = pd.read_csv(f"{data_path}ny_gnp.csv")
    
    st.write("**Data loading info:**")
    st.write(f"Population: {population_df.shape[0]} countries, {population_df.shape[1]-1} years")
    st.write(f"Life Expectancy: {life_expectancy_df.shape[0]} countries, {life_expectancy_df.shape[1]-1} years")
    st.write(f"GNI: {gni_df.shape[0]} countries, {gni_df.shape[1]-1} years")
    
    # Forward fill missing values
    population_df = population_df.ffill(axis=1)
    life_expectancy_df = life_expectancy_df.ffill(axis=1)
    gni_df = gni_df.ffill(axis=1)
    
    # Convert to tidy format
    pop_tidy = population_df.melt(id_vars=["country"], var_name="year", value_name="population")
    life_tidy = life_expectancy_df.melt(id_vars=["country"], var_name="year", value_name="life_expectancy")
    gni_tidy = gni_df.melt(id_vars=["country"], var_name="year", value_name="gni_per_capita")
    
    # Convert year to numeric
    pop_tidy["year"] = pd.to_numeric(pop_tidy["year"], errors='coerce')
    life_tidy["year"] = pd.to_numeric(life_tidy["year"], errors='coerce')
    gni_tidy["year"] = pd.to_numeric(gni_tidy["year"], errors='coerce')
    
    # THE KEY FIX: Parse the formatted numbers
    st.write(" **Parsing formatted numbers...**")
    
    # Parse population numbers (3.28M, 407k, etc.)
    pop_tidy["population"] = pop_tidy["population"].apply(parse_number)
    
    # Life expectancy should already be numeric, but let's be safe
    life_tidy["life_expectancy"] = pd.to_numeric(life_tidy["life_expectancy"], errors='coerce')
    
    # Parse GNI numbers (25.3k, 89.5k, etc.)
    gni_tidy["gni_per_capita"] = gni_tidy["gni_per_capita"].apply(parse_number)
    
    # Check how much data we have after parsing
    st.write(f"After parsing - Population records with valid data: {pop_tidy['population'].notna().sum():,}")
    st.write(f"After parsing - Life expectancy records with valid data: {life_tidy['life_expectancy'].notna().sum():,}")
    st.write(f"After parsing - GNI records with valid data: {gni_tidy['gni_per_capita'].notna().sum():,}")
    
    # Filter to overlapping years (1990-2023 based on GNI data)
    min_year = 1990
    max_year = 2023
    
    pop_filtered = pop_tidy[(pop_tidy["year"] >= min_year) & (pop_tidy["year"] <= max_year)]
    life_filtered = life_tidy[(life_tidy["year"] >= min_year) & (life_tidy["year"] <= max_year)]
    gni_filtered = gni_tidy[(gni_tidy["year"] >= min_year) & (gni_tidy["year"] <= max_year)]
    
    # Merge the datasets
    merged1 = pd.merge(pop_filtered, life_filtered, on=["country", "year"], how="inner")
    final_df = pd.merge(merged1, gni_filtered, on=["country", "year"], how="inner")
    
    st.write(f"After merging: {final_df.shape[0]:,} records from {final_df['country'].nunique()} countries")
    
    # Remove rows with missing essential data
    final_df = final_df.dropna(subset=["country", "year", "population", "life_expectancy", "gni_per_capita"])
    
    # Remove rows with invalid data (zeros or negatives)
    final_df = final_df[
        (final_df["population"] > 0) & 
        (final_df["life_expectancy"] > 0) & 
        (final_df["gni_per_capita"] > 0)
    ]
    
    # Convert year back to int
    final_df["year"] = final_df["year"].astype(int)
    
    st.write(f"**Final dataset:** {len(final_df):,} records, {final_df['country'].nunique()} countries, years {final_df['year'].min()}-{final_df['year'].max()}")
    
    return final_df

# Load the data
data = load_and_process_data()

if len(data) == 0:
    st.error("No data available after processing!")
    st.stop()

# Get available years and countries
years = sorted(data["year"].unique())
countries = sorted(data["country"].unique())

# Sidebar controls
st.sidebar.header(" Controls")

# Year slider
selected_year = st.sidebar.slider(
    "Select Year",
    min_value=min(years),
    max_value=max(years),
    value=max(years),
    step=1
)

# Country selector with smart defaults
major_countries = [
    "China", "India", "United States", "Indonesia", "Pakistan", 
    "Brazil", "Nigeria", "Bangladesh", "Russia", "Mexico",
    "Japan", "Philippines", "Ethiopia", "Vietnam", "Egypt",
    "Germany", "Iran", "Turkey", "Thailand", "United Kingdom"
]

available_major_countries = [c for c in major_countries if c in countries]
if not available_major_countries:
    available_major_countries = countries[:10]

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=available_major_countries[:8],
    help="Select countries to display on the chart"
)

# Show dataset info
with st.sidebar.expander("Dataset Info"):
    st.write(f"**Countries:** {len(countries)}")
    st.write(f"**Years:** {min(years)}-{max(years)}")
    st.write(f"**Records:** {len(data):,}")

# Main visualization
if selected_countries:
    filtered_data = data[
        (data["year"] == selected_year) & 
        (data["country"].isin(selected_countries))
    ]
    
    if not filtered_data.empty:
        # Calculate max values for consistent scaling
        max_gni = data["gni_per_capita"].max()
        
        # Create the bubble chart
        fig = px.scatter(
            filtered_data,
            x="gni_per_capita",
            y="life_expectancy",
            size="population",
            color="country",
            hover_name="country",
            hover_data={
                "gni_per_capita": ":$,.0f",
                "life_expectancy": ":.1f years",
                "population": ":,.0f people",
                "year": False
            },
            log_x=True,
            size_max=60,
            title=f" Life Expectancy vs Income Per Person - {selected_year}",
            labels={
                "gni_per_capita": "Income per person (GDP per capita, PPP$ inflation-adjusted)",
                "life_expectancy": "Life expectancy (years)"
            }
        )
        
        # Set consistent axis ranges
        fig.update_xaxes(
            range=[np.log10(300), np.log10(max_gni * 1.1)],
            title="Income per person (GDP per capita, PPP$ inflation-adjusted) →"
        )
        
        fig.update_yaxes(
            range=[30, 90],
            title="← Life expectancy (years)"
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"Showing **{len(filtered_data)} countries** for **{selected_year}**")
        
        # Show data table
        if st.checkbox(" Show data table"):
            display_data = filtered_data[["country", "population", "life_expectancy", "gni_per_capita"]].copy()
            display_data["population"] = display_data["population"].apply(lambda x: f"{x:,.0f}")
            display_data["life_expectancy"] = display_data["life_expectancy"].apply(lambda x: f"{x:.1f}")
            display_data["gni_per_capita"] = display_data["gni_per_capita"].apply(lambda x: f"${x:,.0f}")
            display_data.columns = ["Country", "Population", "Life Expectancy", "Income per Person"]
            st.dataframe(display_data, use_container_width=True)
    else:
        st.warning("⚠️ No data available for selected countries and year")
else:
    st.info(" Please select countries from the sidebar!")

# Animation
st.sidebar.markdown("---")
st.sidebar.subheader(" Animation")

if selected_countries:
    animation_speed = st.sidebar.slider("⚡ Speed (seconds per year)", 0.1, 2.0, 0.5, 0.1)
    
    if st.sidebar.button("▶️ Play Animation", type="primary"):
        chart_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        max_gni = data["gni_per_capita"].max()
        
        for i, year in enumerate(years):
            progress_bar.progress((i + 1) / len(years))
            
            year_data = data[
                (data["year"] == year) & 
                (data["country"].isin(selected_countries))
            ]
            
            if not year_data.empty:
                fig = px.scatter(
                    year_data,
                    x="gni_per_capita",
                    y="life_expectancy",
                    size="population",
                    color="country",
                    hover_name="country",
                    log_x=True,
                    size_max=60,
                    title=f" Life Expectancy vs Income Per Person - {year}"
                )
                
                fig.update_xaxes(range=[np.log10(300), np.log10(max_gni * 1.1)])
                fig.update_yaxes(range=[30, 90])
                fig.update_layout(height=600)
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            time.sleep(animation_speed)
        
        progress_bar.empty()
        st.success("Animation complete!")

# Show all countries
with st.expander(" All Available Countries"):
    cols = st.columns(4)
    for i, country in enumerate(countries):
        cols[i % 4].write(f"• {country}")

st.markdown("---")
st.markdown("""
- **Bubble size** = Population
- **X-axis (right)** = Higher income per person  
- **Y-axis (up)** = Longer life expectancy
- **Animation** shows progress over time from 1990-2023
""")