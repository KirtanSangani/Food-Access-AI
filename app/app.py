import pandas as pd
import streamlit as st
import joblib
import folium
import geopandas as gpd
import pickle
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

#Load Model and feature dataset
@st.cache_resource
def load_model():
    with open(r"C:\Users\kirta\OneDrive\Desktop\Projects\Food-Access-AI\notebooks\model.pkl","rb") as f:
        return pickle.load(f)

df_model = pd.read_csv(r"C:\Users\kirta\OneDrive\Desktop\Projects\Food-Access-AI\Data\FoodAccessResearchAtlasData.csv")

@st.cache_data
def load_geodata():
    gdf = gpd.read_file(r"C:\Users\kirta\OneDrive\Desktop\Projects\Food-Access-AI\Data\cb_2024_us_tract_500k.zip")
    gdf["ID"] = df_model["ID"]
    return gdf

model = load_model()

print(type(model))
print(dir(model))

gdf = load_geodata()
county_gdf = pd.read_csv(r'C:\Users\kirta\OneDrive\Desktop\Projects\Food-Access-AI\Data\national_county.txt', header = None, dtype=str, names = ["STATE", "STATEFP", "COUNTYFP", "COUNTY_NAME","CLASSFP"])

gdf = gdf.merge(county_gdf[["STATEFP","COUNTYFP","COUNTY_NAME"]], on=["STATEFP","COUNTYFP"],how="left")

#Establishing session_state
if "show_map" not in st.session_state:
    st.session_state["show_map"] = False
if "selected_state" not in st.session_state:
    st.session_state["selected_state"] = None
if "selected_county" not in st.session_state:
    st.session_state["selected_county"] = None

st.title("Food Desert Predictor")
st. markdown("Select a state and county to find out if a region is a food desert")

#State and County Selector
state_list = sorted(gdf["STATE_NAME"].unique())
state = st.selectbox("Select a State",state_list)

county_list = sorted(gdf[gdf["STATE_NAME"] == state]["COUNTY_NAME"].unique())
county = st.selectbox("Select a County",county_list)

if st.button("Create Map"):
        st.session_state["selected_state"] = state
        st.session_state["selected_county"] = county
        st.session_state["show_map"] = True

if st.session_state["show_map"]:
        state = st.session_state["selected_state"]
        county = st.session_state["selected_county"]

        #use model to create predictions
        feature_cols = ['Urban','PovertyRate','MedianFamilyIncome','lapophalf','lapophalfshare','lalowihalf','lalowihalfshare','lakidshalfshare','laseniorshalf','laseniorshalfshare','lawhitehalfshare','lahunvhalfshare','lasnaphalf','lasnaphalfshare']
        gdf['predictability'] = model.predict_proba(model[feature_cols])[:,1]
        gdf['is_food_desert'] = (gdf["predictability"] >= 0.37).astype(int)

        #Find center of the map
        new_gdf = gdf[(gdf["STATE_NAME"] == state) & (gdf["COUNTY_NAME"] == county)]
        center = new_gdf.geometry.unary_union.centroid
        map_center = [center.y,center.x]

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

        st_data = st_folium(m, width = 800, height = 600)

        




