import pandas as pd

import os

path = os.getcwd()


print("===============================")
print(path)
print("===============================")


class UFCStats:
    def __init__(self) -> None:
        self._fighters_name = pd.read_csv("predictor/data/stats/fighters_name.csv")
        self._fighter_stats = pd.read_csv("predictor/data/stats/fighters_name.csv")
        self._ufc_stats = pd.read_csv("predictor/data/stats/ufc_stats.csv")

    def get_fighters_name(self) -> list[str]:
        return self._ufc_stats["fighter"].unique().tolist()

    def get_fighter_data(self, fighter_name: str) -> dict:
        fights = UFCStats._get_fights(self._ufc_stats, fighter_name)
        extra_features = UFCStats._compute_extra_features(fights)
        fights = UFCStats._clean_columns(fights)
        stats = fights.median().to_frame().transpose()
        stats = pd.concat([extra_features, fights], axis=1)
        return stats.to_dict('records')[0]

    @staticmethod
    def _get_fights(df: pd.DataFrame, fighter_name: str):
        all_fights = df[df["fighter"] == fighter_name]
        return all_fights

    @staticmethod
    def _clean_columns(data):
        cleaned_data = data.drop(
            columns=[
                "fighter",
                "Unnamed: 0",
                "event",
                "location",
                "attendance",
                "time",
                "scheduled_rounds",
                "weight_class",
                "round",
                "last_round",
                "id",
                "result",
                "winner",
                "fight_date",
            ]
        )
        return cleaned_data
    
    @staticmethod
    def _compute_extra_features(df: pd.DataFrame) -> pd.DataFrame:
        # compute nb fights
        nb_fights = len(df['id'].unique())
        extra_features = {
            'nb_fights': [nb_fights],
            'L': 0,
            'W': 0,
            'D': 0,
            'NC': 0   
        }
        #compute nb wins / loses
        grouped = df.groupby('id')
        fights = [group.iloc[0] for _, group in grouped]
        fights = pd.DataFrame(fights)
        wins_loses = fights['winner'].value_counts()
        extra_features.update(wins_loses)
        return pd.DataFrame(extra_features)
