from fastapi import FastAPI
from predictor import predictor

app = FastAPI()
FightResultPredictor = predictor.FightResultPredictor()
UFCStats = predictor.UFCStats()


@app.get("/")
def read_root() -> dict:
    return FightResultPredictor.predict_winner("fighter 1", "fighter 2")


@app.get("/fighters_name")
def read_fighters_name() -> list:
    return UFCStats.get_fighters_name()


@app.get("/fighter_stats")
def read_fighter_stats() -> dict:
    fighter_name = UFCStats.get_fighters_name()[0]
    return UFCStats.get_fighter_data(fighter_name)
