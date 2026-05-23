import joblib
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import get_cached_prediction, set_cached_prediction

model = joblib.load(settings.MODEL_PATH)

def car_price_predict(data: dict):
    cache_key = " ".join([str(val) for val in data.values()])
    cached = get_cached_prediction(cache_key)
    if cached:
        return cached
    
    input_value = pd.DataFrame([data])
    prediction = model.predict(input_value)[0]
    set_cached_prediction(cache_key, prediction)
    return prediction