# Food-Access-AI: Predictive Food Analytics Platform

# Inspiration
When trying to decide what my first big, individual, personal project would be, I was stuck trying to find a median between doing something personal for me, but also something viable through the technologies I knew how to use. What brought me towards this project was actually a day volunteering at the food bank. Throughout the time I was volunteering, I saw the amount of food necessary to help out people in need. It brought me to the idea of food shortages around the US, and why do they occur? Are food shortages focused within certain areas? What are the biggest reasons why food shortages like this occur? What if there were an application that allowed people to see which areas are having food shortages? 

After some research, I learned about "food deserts" and the data necessary to determine whether an area is considered a food desert. This research is what brought me to create Food-Access-AI.

# Project Overview
Food-Access-AI is an end-to-end data science and Machine Learning pipeline designed to identify and predict food shortages and insecurity across the United States. By analyzing a dataset containing over 70,000 census records, the application identifies underserved community indicators that can help policymakers and government officials visualize areas with food shortages, allowing for policy changes to be created and food shortages to be decreased. 

The platform provides census-level analysis for all 3,142 US counties, utilizing geographical visualizations to highlight regions that have been identified as "food deserts."

# Key Features
* Machine Learning Model: Utilized a RandomForest Classification Model to determine food desert indicators based on socio-economic and geographic census data.
  * Implemented caching and algorithmic optimizations to allow for the most optimal model performance possible.
* User Interface: Integrated a Folium visualization to create a dynamic map, allowing users to explore the entirety of the United States to analyze food desert patterns throughout the country, but also allow for analysis at a county-level.
* Backend: Implemented FlaskAPI that serves for model predictions and Pickle for a fast, scalable interface.
* Hyperparameter Tuning: Used scikit-learn's GridSearchCV to ensure model accuracy and robustness of the Machine Learning Model.

# Tech Stack
* Languages: Python (Data Science & Backend), CSS (Styling)
* Machine Learning: Scikit-learn (RandomForest, GridSearchCV)
* Web Framework: Flask
* Data Manipulation/Analysis: Pandas, NumPy
* Data Visualization: Folium, Matplotlib, Seaborn
* Deployment: Pickle, Streamlit

# Data & Preprocessing
The model is trained on a dataset of 70,000+ census records. The pipeline includes:
* Data Preprocessing: Handling null values, feature scaling, and encoding categorical variables.
* Feature Engineering: Identifying key factors of food insecurity through Random Forest Feature Importance.
* Model Training: Training and tuning a Random Forest Classifier to handle demographic data.
* Geospatial Analysis: Mapping the classification results to geographic coordinates to create the map visualization.

# Installation and Setup
Prerequisites: 
* Python: 3.8+
* pip

Setup:
### 1. Clone the repository
```bash
git clone https://github.com/KirtanSangani/Food-Access-AI.git
cd Food-Access-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask application
```bash
python app.py
```

4. Access the platform: Navigate to http://127.0.0.1:5000 in your web browser

# Impact & Social Significance
Food-Access-AI transforms  census data into a  tool for social change, providing an accurate roadmap for addressing systemic food shortages. By identifying high-priority censuses through real-time assessment, the platform allows policymakers to move from reactive aid to proactive intervention—enabling the strategic placement of economic grants, mobile markets, and public transit routes. This data-driven approach ensures that limited resources are directed toward communities with the most necessary socio-economic need, effectively bridging the gap between macro-economic reporting and the "last mile" of local food access.

Looking ahead, the project’s roadmap focuses on evolving from predictive identification to prescriptive solutions. Future updates will integrate geospatial mapping to pinpoint specific neighborhoods in need. Because the underlying architecture is region-agnostic, the platform is positioned to scale beyond the U.S., offering a modular framework that can be adapted to combat famine and food insecurity across the Global South using international datasets from the UN and World Bank.

# Developer
Developed By: Kirtan Sangani

Affiliation: University of North Carolina - Chapel Hill
