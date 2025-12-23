"""
Flask API Backend for Food Desert Map Application
Serves census tract data with food desert predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import geopandas as gpd
import pickle
import json
import logging
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

# Feature columns used by the model
FEATURE_COLS = [
    'Urban', 'PovertyRate', 'MedianFamilyIncome', 'lapophalf', 'lapophalfshare',
    'lalowihalf', 'lalowihalfshare', 'lakidshalfshare', 'laseniorshalf',
    'laseniorshalfshare', 'lawhitehalfshare', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare'
]

# Load Model
model = None
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found. Please ensure model.pkl exists.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Load and prepare data
gdf = None
df = None

def load_data():
    global gdf, df
    try:
        # Load CSV data
        df = pd.read_csv(r"Data\FoodAccessResearchAtlasData.csv")
        logger.info(f"CSV data loaded. Shape: {df.shape}")
        
        # Load shapefile
        # Load shapefile (using 2019 shapefile to match 2019 CSV data)
        gdf = gpd.read_file(r"Data\cb_2019_us_tract_500k.zip")
        logger.info(f"Shapefile loaded. Shape: {gdf.shape}")
        
        # Merge data
        if "ID" in df.columns and "GEOID" in gdf.columns:
            # Convert ID to string and pad with leading zeros to match GEOID format (11 characters)
            # CSV IDs are stored as integers which strips leading zeros (e.g., Alabama 01001020100 -> 1001020100)
            df['ID'] = df['ID'].astype(str).str.zfill(11)
            gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(11)
            
            # Remove duplicate IDs from CSV (keep first occurrence)
            df = df.drop_duplicates(subset='ID', keep='first')
            
            # Use inner join to only keep tracts with matching data
            gdf = gdf.merge(df, left_on="GEOID", right_on="ID", how="inner")
            logger.info(f"Data merged successfully. Final shape: {gdf.shape}")
        else:
            logger.warning("ID column mismatch - check column names")
            
        # Make predictions for all tracts
        if model is not None and gdf is not None:
            # Fill missing values for prediction
            valid_mask = gdf[FEATURE_COLS].notna().all(axis=1)
            gdf['is_food_desert'] = 0
            gdf['probability'] = 0.0
            
            if valid_mask.sum() > 0:
                X = gdf.loc[valid_mask, FEATURE_COLS].fillna(0)
                probs = model.predict_proba(X)[:, 1]
                gdf.loc[valid_mask, 'probability'] = probs
                gdf.loc[valid_mask, 'is_food_desert'] = (probs >= 0.37).astype(int)
                logger.info(f"Predictions completed. Food deserts: {gdf['is_food_desert'].sum()}")
                
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

# Load data on startup
load_data()


@app.route("/")
def index():
    return jsonify({
        "message": "Food Desert API",
        "endpoints": ["/api/states", "/api/tracts", "/api/tract/<id>", "/health"]
    })


@app.route("/api/states")
def get_states():
    """Get list of unique states"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    states = sorted(df['State'].dropna().unique().tolist())
    return jsonify({"states": states})


@app.route("/api/counties/<state>")
def get_counties(state):
    """Get list of counties for a state"""
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    counties = sorted(df[df['State'] == state]['County'].dropna().unique().tolist())
    return jsonify({"counties": counties})


@app.route("/api/tracts")
def get_tracts():
    """Get census tract data for mapping - supports filtering by state/county"""
    if gdf is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    state = request.args.get('state', None)
    county = request.args.get('county', None)
    
    filtered_gdf = gdf.copy()
    
    if state:
        filtered_gdf = filtered_gdf[filtered_gdf['State'] == state]
    
    if county:
        filtered_gdf = filtered_gdf[filtered_gdf['County'] == county]
    
    if len(filtered_gdf) == 0:
        return jsonify({"error": "No data found for the specified filters"}), 404
    
    # Simplify geometry for faster loading
    filtered_gdf = filtered_gdf.copy()
    filtered_gdf['geometry'] = filtered_gdf['geometry'].simplify(0.001)
    
    # Convert to GeoJSON
    geojson = json.loads(filtered_gdf.to_json())
    
    # Add statistics
    stats = {
        "total_tracts": len(filtered_gdf),
        "food_desert_tracts": int(filtered_gdf['is_food_desert'].sum()),
        "avg_poverty_rate": float(filtered_gdf['PovertyRate'].mean()) if 'PovertyRate' in filtered_gdf.columns else 0,
        "avg_median_income": float(filtered_gdf['MedianFamilyIncome'].mean()) if 'MedianFamilyIncome' in filtered_gdf.columns else 0
    }
    
    return jsonify({
        "geojson": geojson,
        "stats": stats
    })


@app.route("/api/tract/<tract_id>")
def get_tract_details(tract_id):
    """Get detailed info for a specific tract"""
    if gdf is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    tract = gdf[gdf['GEOID'] == tract_id]
    
    if len(tract) == 0:
        return jsonify({"error": "Tract not found"}), 404
    
    tract = tract.iloc[0]
    
    return jsonify({
        "id": tract_id,
        "state": tract.get('State', 'N/A'),
        "county": tract.get('County', 'N/A'),
        "is_food_desert": int(tract.get('is_food_desert', 0)),
        "probability": float(tract.get('probability', 0)),
        "poverty_rate": float(tract.get('PovertyRate', 0)),
        "median_income": float(tract.get('MedianFamilyIncome', 0)),
        "population": int(tract.get('Pop2010', 0)),
        "urban": int(tract.get('Urban', 0)),
        "low_access_pop": int(tract.get('lapophalf', 0)),
        "snap_recipients": int(tract.get('TractSNAP', 0)) if pd.notna(tract.get('TractSNAP')) else 0
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "data_loaded": gdf is not None,
        "data_shape": list(gdf.shape) if gdf is not None else None
    })


if __name__ == "__main__":
    logger.info("Starting Food Desert API...")
    app.run(debug=True, host="0.0.0.0", port=5000)

