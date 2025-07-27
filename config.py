import os

class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get("FLASK_DEBUG") or True

    #Data File Paths
    MODEL_PATH = 'model.pkl'
    DATA_CSV_PATH = 'Food-Access-AI\Data\FoodAccessResearchAtlasData.csv'
    DATA_SHAPEFILE_PATH = 'Food-Access-AI\Data\cb_2024_us_tract_500k.zip'

    #Prediction Threshold
    FOOD_DESERT_THRESHOLD = 0.37

    #Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/app.log'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}