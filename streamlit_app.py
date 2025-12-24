"""
Streamlit Frontend for Food Desert Interactive Map
Displays US Census Tracts with food desert predictions
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Food Desert Map | US Census Tracts",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #e63946 0%, #f77f00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        color: #a8dadc;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
        border: 1px solid rgba(168, 218, 220, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(230, 57, 70, 0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e63946 0%, #f77f00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #a8dadc;
        font-size: 0.9rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1.5rem;
        color: #f1faee;
    }
    
    .legend-dot {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        margin-right: 8px;
    }
    
    .sidebar .stSelectbox label {
        color: #a8dadc !important;
        font-weight: 500;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
        border-right: 1px solid rgba(168, 218, 220, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #e63946 0%, #f77f00 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(230, 57, 70, 0.4);
    }
    
    .info-box {
        background: rgba(168, 218, 220, 0.1);
        border-left: 4px solid #e63946;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .info-box p, .info-box li {
        color: #ffffff;
    }
    
    .info-box strong {
        color: #ffffff;
    }
    
    /* Style expander headers white */
    .streamlit-expanderHeader {
        color: #ffffff !important;
    }
    
    div[data-testid="stExpander"] summary span {
        color: #ffffff !important;
    }
    
    /* Sidebar filter headers - black text */
    section[data-testid="stSidebar"] h3 {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Feature columns for the model
FEATURE_COLS = [
    'Urban', 'PovertyRate', 'MedianFamilyIncome', 'lapophalf', 'lapophalfshare',
    'lalowihalf', 'lalowihalfshare', 'lakidshalfshare', 'laseniorshalf',
    'laseniorshalfshare', 'lawhitehalfshare', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare'
]


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data():
    """Load and prepare census tract data"""
    try:
        # Load CSV data
        df = pd.read_csv(r"Data/FoodAccessResearchAtlasData2019.xlsx - Food Access Research Atlas - for model.csvFoodAccessResearchAtlasData.csv")
        
        # Load shapefile (using 2019 shapefile to match 2019 CSV data)
        gdf = gpd.read_file(r"Data/cb_2019_us_tract_500k.zip")
        
        # Convert IDs to string and pad with leading zeros to match GEOID format (11 characters)
        # CSV IDs are stored as integers which strips leading zeros (e.g., Alabama 01001020100 -> 1001020100)
        df['ID'] = df['ID'].astype(str).str.zfill(11)
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(11)
        
        # Remove duplicate IDs from CSV (keep first occurrence)
        df = df.drop_duplicates(subset='ID', keep='first')
        
        # Merge data - use inner join to only keep tracts that have matching data
        gdf = gdf.merge(df, left_on="GEOID", right_on="ID", how="inner")
        
        # OPTIMIZATION: Pre-simplify geometries for faster rendering
        # More aggressive simplification (0.005) significantly reduces polygon complexity
        gdf['geometry'] = gdf['geometry'].simplify(0.005, preserve_topology=True)
        
        # OPTIMIZATION: Pre-calculate centroids for faster map centering
        gdf['centroid_lat'] = gdf.geometry.centroid.y
        gdf['centroid_lon'] = gdf.geometry.centroid.x
        
        return gdf, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data
def make_predictions(_gdf, _model):
    """Make food desert predictions for all tracts"""
    gdf = _gdf.copy()
    
    # Initialize columns
    gdf['is_food_desert'] = 0
    gdf['probability'] = 0.0
    
    # Find valid rows for prediction
    valid_mask = gdf[FEATURE_COLS].notna().all(axis=1)
    
    if valid_mask.sum() > 0 and _model is not None:
        X = gdf.loc[valid_mask, FEATURE_COLS].fillna(0)
        probs = _model.predict_proba(X)[:, 1]
        gdf.loc[valid_mask, 'probability'] = probs
        gdf.loc[valid_mask, 'is_food_desert'] = (probs >= 0.37).astype(int)
    
    return gdf


@st.cache_data
def prepare_geojson(_gdf, state_filter, county_filter):
    """Cache the GeoJSON preparation for faster repeat renders"""
    # Filter data
    if state_filter and state_filter != "All States":
        gdf_filtered = _gdf[_gdf['State'] == state_filter].copy()
        if county_filter and county_filter != "All Counties":
            gdf_filtered = gdf_filtered[gdf_filtered['County'] == county_filter].copy()
    else:
        gdf_filtered = _gdf.copy()
    
    # Only include essential columns to reduce size
    essential_cols = ['geometry', 'GEOID', 'State', 'County', 'probability', 
                      'is_food_desert', 'PovertyRate', 'MedianFamilyIncome', 
                      'Pop2010', 'Urban', 'centroid_lat', 'centroid_lon']
    available_cols = [c for c in essential_cols if c in gdf_filtered.columns]
    gdf_minimal = gdf_filtered[available_cols].copy()
    
    # Convert to GeoJSON string once
    geojson_str = gdf_minimal.to_json()
    
    return gdf_filtered, geojson_str


def create_map(gdf_filtered, zoom_level=6, geojson_str=None):
    """Create interactive choropleth map using Plotly - OPTIMIZED"""
    
    if len(gdf_filtered) == 0:
        return None
    
    # Use pre-simplified geometries (done at load time)
    gdf_plot = gdf_filtered.copy()
    
    # OPTIMIZATION: Build hover text using vectorized operations (much faster than apply)
    food_desert_label = np.where(gdf_plot['is_food_desert'] == 1, '🔴 Yes', '⚪ No')
    urban_label = np.where(gdf_plot['Urban'] == 1, 'Yes', 'No')
    
    gdf_plot['hover_text'] = (
        '<b>Census Tract:</b> ' + gdf_plot['GEOID'].astype(str) + '<br>' +
        '<b>State:</b> ' + gdf_plot['State'].fillna('N/A').astype(str) + '<br>' +
        '<b>County:</b> ' + gdf_plot['County'].fillna('N/A').astype(str) + '<br>' +
        '<b>────────────</b><br>' +
        '<b>Food Desert:</b> ' + food_desert_label + '<br>' +
        '<b>Probability:</b> ' + (gdf_plot['probability'] * 100).round(1).astype(str) + '%<br>' +
        '<b>────────────</b><br>' +
        '<b>Poverty Rate:</b> ' + gdf_plot['PovertyRate'].round(1).astype(str) + '%<br>' +
        '<b>Median Income:</b> $' + gdf_plot['MedianFamilyIncome'].fillna(0).astype(int).astype(str) + '<br>' +
        '<b>Population:</b> ' + gdf_plot['Pop2010'].fillna(0).astype(int).astype(str) + '<br>' +
        '<b>Urban:</b> ' + urban_label
    )
    
    # OPTIMIZATION: Use pre-calculated centroids
    if 'centroid_lat' in gdf_plot.columns:
        center_lat = gdf_plot['centroid_lat'].mean()
        center_lon = gdf_plot['centroid_lon'].mean()
    else:
        center_lat = gdf_plot.geometry.centroid.y.mean()
        center_lon = gdf_plot.geometry.centroid.x.mean()
    
    # Custom white-to-red color scale
    white_to_red = [
        [0.0, '#ffffff'],
        [0.3, '#ffcccc'],
        [0.5, '#ff8080'],
        [0.7, '#e63946'],
        [1.0, '#b31b2c']
    ]
    
    # OPTIMIZATION: Only include essential columns in GeoJSON to reduce size
    essential_cols = ['geometry', 'GEOID', 'probability', 'hover_text']
    gdf_minimal = gdf_plot[essential_cols].copy()
    
    # Use pre-prepared GeoJSON if available, otherwise generate
    if geojson_str:
        geojson_data = json.loads(geojson_str)
    else:
        geojson_data = json.loads(gdf_minimal.to_json())
    
    # Create the map using Plotly with probability for continuous color scale
    fig = px.choropleth_mapbox(
        gdf_minimal,
        geojson=geojson_data,
        locations=gdf_minimal.index,
        color='probability',
        color_continuous_scale=white_to_red,
        range_color=[0, 1],
        mapbox_style="carto-darkmatter",
        zoom=zoom_level,
        center={"lat": center_lat, "lon": center_lon},
        opacity=0.75,
        hover_name='GEOID',
        custom_data=['hover_text'],
        labels={'probability': 'Food Desert Probability'}
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        marker_line_width=0.3,
        marker_line_color='rgba(255,255,255,0.2)'
    )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        mapbox=dict(
            accesstoken=None,
            bearing=0,
            pitch=0
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text="Food Desert<br>Probability",
                side="right",
                font=dict(color='#f1faee')
            ),
            tickformat=".0%",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            len=0.6,
            thickness=15,
            bgcolor='rgba(0,0,0,0.5)',
            tickfont=dict(color='#f1faee')
        )
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🗺️ US Food Desert Map</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive visualization of food access across US Census Tracts</p>', unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("Loading data..."):
        model = load_model()
        gdf, df = load_data()
    
    if gdf is None or df is None:
        st.error("Failed to load data. Please check that the data files exist in the Data folder.")
        return
    
    # Make predictions
    with st.spinner("Making predictions..."):
        gdf = make_predictions(gdf, model)
    
    # Sidebar filters
    st.sidebar.markdown("### Filter Options")
    
    # Get unique states
    states = sorted(gdf['State'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox(
        "Select State",
        options=["All States"] + states,
        index=0
    )
    
    # Filter by state
    if selected_state != "All States":
        gdf_filtered = gdf[gdf['State'] == selected_state].copy()
        
        # Get counties for selected state
        counties = sorted(gdf_filtered['County'].dropna().unique().tolist())
        selected_county = st.sidebar.selectbox(
            "Select County",
            options=["All Counties"] + counties,
            index=0
        )
        
        if selected_county != "All Counties":
            gdf_filtered = gdf_filtered[gdf_filtered['County'] == selected_county].copy()
            zoom_level = 10
        else:
            zoom_level = 7
    else:
        # Show all census tracts for the entire US
        st.sidebar.info(f"Showing {len(gdf):,} census tracts")
        gdf_filtered = gdf.copy()
        zoom_level = 4
    
    # Additional filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Additional Filters")
    
    show_only_deserts = st.sidebar.checkbox("Show only Food Deserts", value=False)
    if show_only_deserts:
        gdf_filtered = gdf_filtered[gdf_filtered['is_food_desert'] == 1].copy()
    
    urban_filter = st.sidebar.radio(
        "Area Type",
        options=["All", "Urban Only", "Rural Only"],
        index=0
    )
    
    if urban_filter == "Urban Only":
        gdf_filtered = gdf_filtered[gdf_filtered['Urban'] == 1].copy()
    elif urban_filter == "Rural Only":
        gdf_filtered = gdf_filtered[gdf_filtered['Urban'] == 0].copy()
    
    # Poverty rate filter - only apply if there's valid poverty data
    if 'PovertyRate' in gdf_filtered.columns and len(gdf_filtered) > 0:
        # Get min/max from non-null values only
        poverty_data = gdf_filtered['PovertyRate'].dropna()
        if len(poverty_data) > 0:
            min_poverty = float(poverty_data.min())
            max_poverty = float(poverty_data.max())
            
            if min_poverty < max_poverty:
                poverty_range = st.sidebar.slider(
                    "Poverty Rate Range (%)",
                    min_value=min_poverty,
                    max_value=max_poverty,
                    value=(min_poverty, max_poverty)
                )
                # Only filter if user changed the range from default
                if poverty_range[0] > min_poverty or poverty_range[1] < max_poverty:
                    gdf_filtered = gdf_filtered[
                        (gdf_filtered['PovertyRate'] >= poverty_range[0]) & 
                        (gdf_filtered['PovertyRate'] <= poverty_range[1])
                    ].copy()
    
    # Statistics cards
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_tracts = len(gdf_filtered)
    food_desert_count = gdf_filtered['is_food_desert'].sum() if len(gdf_filtered) > 0 else 0
    percentage = (food_desert_count / total_tracts * 100) if total_tracts > 0 else 0
    avg_poverty = gdf_filtered['PovertyRate'].mean() if 'PovertyRate' in gdf_filtered.columns and len(gdf_filtered) > 0 else 0
    avg_income = gdf_filtered['MedianFamilyIncome'].mean() if 'MedianFamilyIncome' in gdf_filtered.columns and len(gdf_filtered) > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total_tracts:,}</div>
            <div class="stat-label">Total Census Tracts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{int(food_desert_count):,}</div>
            <div class="stat-label">Food Desert Tracts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{percentage:.1f}%</div>
            <div class="stat-label">Food Desert Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_poverty:.1f}%</div>
            <div class="stat-label">Avg Poverty Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Legend
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <span style="color: #f1faee; font-weight: 500; margin-right: 1rem;">Food Desert Risk:</span>
        <span class="legend-item">
            <span class="legend-dot" style="background: #ffffff; border: 1px solid #ccc;"></span>
            Low (0%)
        </span>
        <span class="legend-item">
            <span class="legend-dot" style="background: #ffb3b3;"></span>
            Medium
        </span>
        <span class="legend-item">
            <span class="legend-dot" style="background: #e63946;"></span>
            High (100%)
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Map
    if len(gdf_filtered) > 0:
        with st.spinner("Rendering map..."):
            fig = create_map(gdf_filtered, zoom_level)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("No data available for the selected filters.")
    else:
        st.warning("No census tracts found for the selected filters.")
    
    # Data table
    st.markdown("---")
    st.markdown('<h3 style="color: #ffffff;">Census Tract Data</h3>', unsafe_allow_html=True)
    
    with st.expander("View Data Table", expanded=False):
        display_cols = ['GEOID', 'State', 'County', 'is_food_desert', 'probability', 
                       'PovertyRate', 'MedianFamilyIncome', 'Pop2010', 'Urban',
                       'TractLOWI', 'TractSNAP', 'lapophalf']
        
        available_cols = [col for col in display_cols if col in gdf_filtered.columns]
        
        display_df = gdf_filtered[available_cols].copy()
        display_df = display_df.rename(columns={
            'GEOID': 'Census Tract ID',
            'is_food_desert': 'Food Desert',
            'probability': 'Probability',
            'PovertyRate': 'Poverty Rate (%)',
            'MedianFamilyIncome': 'Median Income ($)',
            'Pop2010': 'Population',
            'TractLOWI': 'Low Income Pop',
            'TractSNAP': 'SNAP Recipients',
            'lapophalf': 'Low Access Pop'
        })
        
        st.dataframe(
            display_df.head(500),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv,
            file_name="food_desert_data.csv",
            mime="text/csv"
        )
    
    # Info section
    st.markdown("---")
    st.markdown('<h3 style="color: #ffffff;">About This Map</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>What is a Food Desert?</strong></p>
        <p>A food desert is defined as a census tract where a significant portion of the population has 
        low access to supermarkets or large grocery stores. This map uses the USDA Food Access Research Atlas 
        data and a Random Forest classification model to identify food desert tracts.</p>
        <p><strong>Data Sources:</strong></p>
        <ul>
            <li>USDA Food Access Research Atlas (2019)</li>
            <li>US Census Bureau TIGER/Line Shapefiles</li>
        </ul>
        <p><strong>Key Metrics Shown:</strong></p>
        <ul>
            <li><strong>Poverty Rate:</strong> Percentage of population below poverty line</li>
            <li><strong>Median Income:</strong> Median family income in the tract</li>
            <li><strong>Low Access Population:</strong> People living more than 0.5 miles from a supermarket</li>
            <li><strong>SNAP Recipients:</strong> Number of SNAP/food stamp recipients in the tract</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #ffffff; padding: 1rem;">
        <p>Built with Streamlit • Data from USDA Food Access Research Atlas</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

