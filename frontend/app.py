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
        logging.FileHandler('logs/app.log'), 
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
    df = pd.read_csv(r"C:\Users\kirta\OneDrive\Desktop\Projects\Food-Access-AI\Data\FoodAccessResearchAtlasData.csv")
    gdf = gpd.read_file(r"Food-Access-AI\Data\cb_2024_us_tract_500k.zip")

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

        sub_gdf = None

        state_mask = gdf.astype(str).apply(lambda x: x.str.contains(state, case=False,na=False)).any(axis=1)
        county_mask = gdf.astype(str).apply(lambda x: x.str.contains(county.replace(" County", ""), case=False, na=False)).any(axis=1)
        sub_gdf = gdf[state_mask & county_mask]

        if sub_gdf.empty:
            return jsonify({"error": f"No data found for {county}, {state}. Please check spelling and ensure you are using the full state name"}), 404
        
        logger.info(f"Found {len(sub_gdf)} census tracts for {county}, {state}")

        #Check if required columns exist
        missing_cols = [col for col in feature_cols if col not in sub_gdf.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required data columns:  {missing_cols[:3]}{'...' if len(missing_cols) > 3 else ''}"}), 500
        
        sub_gdf_clean = sub_gdf[feature_cols].fillna(0)

        #Make prediction

        try:
            predictions = model.predict_proba(sub_gdf_clean)
            sub_gdf = sub_gdf.copy()
            sub_gdf['predictability'] = predictions[:, 1]
            sub_gdf['is_food_desert'] = (sub_gdf['predictability'] >= 0.37).astype(int)

            logger.info(f"Predictions completed. Found {sub_gdf['is_food_desert'].sum()} food desert tracts")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({"error": f"Prediction model failed. Please try again."})

        #Create Map
        try:
            center = sub_gdf.geometry.unary_union.centroid
            m = folium.Map(
                location = [center.y, center.x],
                zoom_start=10,
                tiles='OpenStreetMap'
            )

            folium.Choropleth(
                geo_data=sub_gdf,
                data=sub_gdf,
                columns=["ID", "is_food_desert"],
                key_on="feature.properties,ID",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Food Desert Status (1=Yes, 0=No)"
            ).add_to(m)

            for idx, row in sub_gdf.iterrrows():
                if row['is_food_desert'] == 1:
                    centroid = row.geometry.centroid
                    popup_text = f"""
                    <div style= "font-family: Arial; width: 200px;">
                        <b>Food Desert Area</b><br>
                        <hr style="margin: 5px 0;">
                        <b>Probability:</b> {row['predictability']:.1%}<br>
                        <b>Poverty rate:</b> {row.get('PovertyRate', 'N/A'):.1%}<br>
                        <b?>Median Income:</b> ${row.get('MedianFamilyIncome', 0):, .0f}<br>
                        <b>Urban:</b> {'Yes' if row.get('Urban', 0) else 'No'}
                    </div>
                    """
                    folium.CircleMarker(
                        location = [centroid.y,centroid.x],
                        radius = 8,
                        popup = folium.Popup(popup_text, max_width=250),
                        color = 'darkred',
                        fill = True,
                        fillColor='red',
                        fillOpacity = 0.8
                    ).add_to(m)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                map_path = f"static/map_{timestamp}.html"
                m.save(map_path)

                total_tracts = len(sub_gdf)
                food_desert_tracts = sub_gdf['is_food_desert'].sum()
                percentage = round((food_desert_tracts/ total_tracts) * 100, 1) if total_tracts > 0 else 0

                logging.info(f"Map generated successfully: {map_path}")

                return jsonify ({"map_url": f"/map?file=map_{timestamp}.html", "stats": {
                    "total_tracts": total_tracts,
                    "food_desert_tracts": int(food_desert_tracts),
                    "percentage": percentage
                }})
            
        except Exception as e:
            logger.error(f"Map generation failed: {e}")
            return jsonify({"error": f"Map generation failed. Please try again."}), 500
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Internal server error. Please try again later."}), 500

@app.route("/map")
def map():
    filename = request.args.get('file', 'map.html')
    map_path = f"static/{filename}"

    if os.path.exists(map_path):
        return send_file(map_path)
    else:
        return jsonify({"error": "Map not found. Please generate a map first."}), 404
    
@app.route("/health")
def health():
    return jsonify ({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "data_shape": gdf.shape if gdf is not None else None,
        "version": "1.0.0"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonifty({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    logger.info("Starting Food Desert Predictor...")
    app.run(debug = True, host = "0.0.0.0", port = 5000)

