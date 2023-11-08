from xgboost import XGBClassifier

class FightResultPredictor:
    def __init__(self) -> None:
        self.model = self._load_model("ufc/data/models/model.json")

    def _load_model(self, model_path: str) -> XGBClassifier:
        xgb = XGBClassifier()
        xgb.load_model(model_path)
        return xgb

    def predict_winner(self, fighter_data_1: list, fighter_data_2: list):
        """
        Probabilty that the fighter 1 win over fighter 2
        """
        data = [fighter_data_1 + fighter_data_2]
        preds = self.model.predict_proba(data).tolist()
        return {'loose': preds[0][0], 'win': preds[0][1], }
