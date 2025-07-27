from flask import Flask, request, jsonify, render_template
import pandas as pd 
import geopandas as gpd
import pickle
import folium
import os
from shapely.geometry import Point
import logging
from flask_cors import CORS

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
logger = logger.getLogger(__name__)


@app.route('/', method =["GET"])
def index(): 
    return render.template("index.html")

def load_model():
    with open(r"Food-Access-AI\backend\notebooks\model.pkl","rb") as f:
        return pickle.load(f)

df_model = pd.read_csv(r"Food-Access-AI\backend\Data\FoodAccessResearchAtlasData.csv")

def load_geodata():
    gdf = gpd.read_file(r"Food-Access-AI\backend\Data\cb_2024_us_tract_500k.zip")
    gdf["ID"] = df_model["ID"]
    return gdf

model = load_model()

gdf = load_geodata()
county_gdf = pd.read_csv(r'Food-Access-AI\backend\Data\national_county.txt', header = None, dtype=str, names = ["STATE", "STATEFP", "COUNTYFP", "COUNTY_NAME","CLASSFP"])

gdf = gdf.merge(county_gdf[["STATEFP","COUNTYFP","COUNTY_NAME"]], on=["STATEFP","COUNTYFP"],how="left")

@app.route('/predict', methods = ["POST"])
def predict():
    data = request.json
    state = data.get("state")
    county = data.get("county")

    # Subset to user’s county
    sub_gdf = gdf[(gdf["STATE_NAME"] == state) & (gdf["COUNTY_NAME"] == county)]

    # Make prediction
    feature_cols = ['Urban','PovertyRate','MedianFamilyIncome','lapophalf','lapophalfshare',
                    'lalowihalf','lalowihalfshare','lakidshalfshare','laseniorshalf',
                    'laseniorshalfshare','lawhitehalfshare','lahunvhalfshare','lasnaphalf','lasnaphalfshare']
    
    sub_gdf['predictability'] = model.predict_proba(sub_gdf[feature_cols])[:,1]
    sub_gdf['is_food_desert'] = (sub_gdf["predictability"] >= 0.37).astype(int)
     
     #Build Map
    m = folium.Map(location = map_center,zoom_start=10)
    folium.Choropleth(
        geo_data=new_gdf,
        data=new_gdf,
        columns=["id","is_food_desert"],
        key_on="feature.properties.ID",
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Food Desert: 1 = Yes, 0 = No"
    ).add_to(m)

    map_path = "static/map.html"
    m.save(map_path)

    return jsonify({"map_url": f"/map"})

@app.route("/map")
def map():
    return send_file("static/map.html")

if __name__ == "__main__":
    app.run(debug=True)