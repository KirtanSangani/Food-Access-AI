from flask import Flask, request, jsonify, render_template
import pandas as pd 
import geopandas as gpd
import pickle
import folium
import os
from shapely.geometry import Point
import logging
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

#Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log')
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs("static", exist_ok=True)
os.makedirs("logs",exist_ok="True")

#Load Model
try:
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found. Please ensure model.pkl exists.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

#Load Data
try:
    df = pd.read_csv("Food-Access-AI\Data\FoodAccessResearchAtlasData.csv")
    gdf = pd.read_file("Food-Access-AI\Data\cb_2024_us_tract_500k.zip")

    if "ID" not in df.columns:
        logger.error("ID column not found in CSV data")
    else:
        gdf = gdf.merge(df, left_on="GEOID", right_on="ID", how="left")
    
    logger.info(f"Data loaded successfully. Shape: {gdf.shape}")
except FileNotFoundError as e:
    logging.error(f"Data not found: {e}")
    gdf = None
except Exception as e:
    logging.error(f"Error loading data: {e}")
    gdf = None

@app.route("/")
def index():
    #Main Page
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():
    #Make predictions for food desert
    try:
        if model is None:
            return jsonify({"error:": "Model not loaded. Please check server configuration."}), 500
        
        if gdf is None:
            return jsonify({"error": "Geographic data not loaded"}), 500
        
        data = request.json
        if not data:
            return jsonify({"error": "No data provided."}), 400
        
        state = data.get("state", "").strip()
        county = data.get("county", "").strip()

        if not state or not county:
            return jsonify({"error": "Both state and county are required"}), 400
        
        logger.info(f"Processing request for {county}, {state}")

        
