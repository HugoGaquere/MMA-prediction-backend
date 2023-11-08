from xgboost import XGBClassifier
from ufc.stats import get_fighter_data, get_fighter_name

class FightResultPredictor:
    def __init__(self) -> None:
        self.model = self._load_model("ufc/data/models/model.json")

    def _load_model(self, model_path: str) -> XGBClassifier:
        xgb = XGBClassifier()
        xgb.load_model(model_path)
        return xgb

    def predict_winner(self, fighter_id_1: int, fighter_id_2: int):
        data = [get_fighter_data(fighter_id_1) + get_fighter_data(fighter_id_2)]
        preds = self.model.predict_proba(data).tolist()

        return {
            f'{get_fighter_name(fighter_id_1)}': preds[0][1],
            f'{get_fighter_name(fighter_id_2)}': preds[0][0]
        }
    