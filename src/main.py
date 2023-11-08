from fastapi import FastAPI
from ufc.predictor import FightResultPredictor
from ufc import stats

app = FastAPI()
_FightResultPredictor = FightResultPredictor()


@app.get("/")
def read_root() -> str:
    return "Hello"

@app.get("/predict")
def read_predict(fighter_id_1: int, fighter_id_2: int):
    data_1 = stats.get_fighter_data(fighter_id_1)
    data_2 = stats.get_fighter_data(fighter_id_2)
    result = _FightResultPredictor.predict_winner(data_1, data_2)
    return result


@app.get("/fighters_name")
def read_fighters_name() -> list[dict]:
    return stats.get_fighters_name()


@app.get("/fighter_stats/{fighter_id}")
def read_fighter_stats(fighter_id: int):
    return stats.get_fighter_data(fighter_id)
