import pandas as pd
from predictor.fighters import UFCStats
from xgboost import XGBClassifier


class FightResultPredictor:
    def __init__(self) -> None:
        self.model = self._load_model("predictor/data/models/model.json")

    def _load_model(self, model_path: str) -> XGBClassifier:
        xgb = XGBClassifier()
        xgb.load_model(model_path)
        return xgb

    def predict_winner(self, fighter_1: str, fighter_2: str) -> dict[str, float]:
        return {fighter_1: 0.4, fighter_2: 0.6}
