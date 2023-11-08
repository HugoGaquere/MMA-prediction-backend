from fastapi import FastAPI
from ufc.predictor import FightResultPredictor
from ufc.stats import get_fighter_data, get_fighters_name

app = FastAPI()
Predictor = FightResultPredictor()


@app.get("/api/v1/predict")
def read_predict(fighter_id_1: int, fighter_id_2: int):
    result = Predictor.predict_winner(fighter_id_1, fighter_id_2)
    return result


@app.get("/api/v1/fighters_name")
def read_fighters_name() -> list[dict]:
    return get_fighters_name()


@app.get("/api/v1/fighter_data/{fighter_id}")
def read_fighter_data(fighter_id: int):
    return get_fighter_data(fighter_id)


@app.get("/api/v1/model_accuracy")
def read_model_accuracy(fighter_id: int):
    return {"accuracy": 0.78}
