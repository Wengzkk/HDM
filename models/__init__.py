from models.HDM import *
from loguru import logger

def get_model_from_args(**kwargs)->HDMModel:
    model = HDMModel(**kwargs)
    return model