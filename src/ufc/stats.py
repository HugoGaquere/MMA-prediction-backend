import pandas as pd

fighters_name = pd.read_csv("ufc/data/stats/fighters_name.csv")
fighter_stats = pd.read_csv("ufc/data/stats/fighters_name.csv")
ufc_stats = pd.read_csv("ufc/data/stats/ufc_stats.csv")


def get_fighters_name() -> list[dict]:
    return fighters_name.to_dict("records")


def get_fighter_name(fighter_id) -> str:
    return fighters_name[fighters_name["id"] == fighter_id]["fighter"].values[0]


def get_fighter_data(fighter_id: int) -> list:
    all_data = _get_all_data(fighter_id)
    extra_features = _compute_extra_features(all_data)
    all_data = _clean_columns(all_data)
    stats = all_data.median().to_frame().transpose()
    stats = pd.concat([extra_features, stats], axis=1)
    return stats.values[0].tolist()


def _get_all_data(fighter_id: int):
    fighter_name = get_fighter_name(fighter_id)
    all_fights = ufc_stats[ufc_stats["fighter"] == fighter_name]
    return all_fights


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


def _compute_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    # compute nb fights
    nb_fights = len(df["id"].unique())
    extra_features = {"nb_fights": [nb_fights], "L": 0, "W": 0, "D": 0, "NC": 0}
    # compute nb wins / loses
    grouped = df.groupby("id")
    fights = [group.iloc[0] for _, group in grouped]
    fights = pd.DataFrame(fights)
    wins_loses = fights["winner"].value_counts()
    extra_features.update(wins_loses)
    return pd.DataFrame(extra_features)
