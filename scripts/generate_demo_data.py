#!/usr/bin/env python3
"""Generate demo datasets for the EDA Agent.

Creates realistic synthetic versions of popular Kaggle datasets:
  - Titanic (survival classification)
  - Iris (flower measurements)
  - Dota 2 Pro Matches (game statistics)
  - Restaurant Data (intentionally messy for cleaning showcase)
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
random.seed(42)


# ---------------------------------------------------------------------------
# 1. Titanic
# ---------------------------------------------------------------------------
def generate_titanic(n: int = 891) -> pd.DataFrame:
    """Realistic Titanic-style survival dataset."""
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = np.full(n, np.nan)
    for i in range(n):
        if rng.random() < 0.80:  # 20% missing
            if pclass[i] == 1:
                age[i] = rng.normal(38, 12)
            elif pclass[i] == 2:
                age[i] = rng.normal(30, 14)
            else:
                age[i] = rng.normal(25, 12)
    age = np.clip(age, 0.5, 80)

    sibsp = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.68, 0.23, 0.05, 0.02, 0.015, 0.005])
    parch = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.76, 0.13, 0.06, 0.02, 0.02, 0.01])

    fare = np.zeros(n)
    for i in range(n):
        base = {1: 80, 2: 20, 3: 10}[pclass[i]]
        fare[i] = max(0, rng.normal(base, base * 0.5))

    embarked_vals = []
    for i in range(n):
        if rng.random() < 0.995:
            embarked_vals.append(rng.choice(["S", "C", "Q"], p=[0.72, 0.19, 0.09]))
        else:
            embarked_vals.append(None)

    # Survival (correlated with sex, pclass, age)
    survived = np.zeros(n, dtype=int)
    for i in range(n):
        prob = 0.38
        if sex[i] == "female":
            prob += 0.35
        if pclass[i] == 1:
            prob += 0.15
        elif pclass[i] == 3:
            prob -= 0.12
        if not np.isnan(age[i]) and age[i] < 15:
            prob += 0.15
        survived[i] = 1 if rng.random() < np.clip(prob, 0.05, 0.95) else 0

    first_names_m = ["James", "John", "Robert", "William", "Thomas", "Charles", "George",
                     "Henry", "Edward", "Joseph", "Samuel", "David", "Patrick", "Michael"]
    first_names_f = ["Mary", "Anna", "Elizabeth", "Margaret", "Alice", "Helen", "Lucy",
                     "Emma", "Sarah", "Catherine", "Florence", "Agnes", "Ellen", "Dorothy"]
    last_names = ["Smith", "Johnson", "Brown", "Williams", "Jones", "Davis", "Miller",
                  "Wilson", "Moore", "Taylor", "Anderson", "White", "Harris", "Martin",
                  "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez",
                  "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "King", "Wright"]
    titles = {"male": ["Mr.", "Master", "Rev.", "Dr."], "female": ["Mrs.", "Miss", "Ms."]}

    names = []
    for i in range(n):
        fn = rng.choice(first_names_m if sex[i] == "male" else first_names_f)
        ln = rng.choice(last_names)
        title = rng.choice(titles[sex[i]])
        names.append(f"{ln}, {title} {fn}")

    cabin = [None] * n
    for i in range(n):
        if rng.random() < (0.6 if pclass[i] == 1 else 0.15 if pclass[i] == 2 else 0.05):
            deck = rng.choice(list("ABCDEF"))
            cabin[i] = f"{deck}{rng.integers(1, 150)}"

    ticket = [f"{''.join(rng.choice(list(string.ascii_uppercase), size=rng.integers(0, 4)))}"
              f"{'/' if rng.random() < 0.1 else ' ' if rng.random() < 0.3 else ''}"
              f"{rng.integers(1000, 999999)}" for _ in range(n)]

    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex,
        "Age": np.where(np.isnan(age), np.nan, np.round(age, 0)),
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": np.round(fare, 2),
        "Cabin": cabin,
        "Embarked": embarked_vals,
    })


# ---------------------------------------------------------------------------
# 2. Iris
# ---------------------------------------------------------------------------
def generate_iris() -> pd.DataFrame:
    """Classic Iris dataset — 150 samples, 4 features, 3 species."""
    species_params = {
        "setosa":     {"sl": (5.0, 0.35), "sw": (3.4, 0.38), "pl": (1.5, 0.17), "pw": (0.2, 0.10)},
        "versicolor": {"sl": (5.9, 0.52), "sw": (2.8, 0.31), "pl": (4.3, 0.47), "pw": (1.3, 0.20)},
        "virginica":  {"sl": (6.6, 0.64), "sw": (3.0, 0.32), "pl": (5.6, 0.55), "pw": (2.0, 0.27)},
    }
    rows = []
    for species, params in species_params.items():
        for _ in range(50):
            rows.append({
                "sepal_length": max(4.0, round(rng.normal(*params["sl"]), 1)),
                "sepal_width":  max(2.0, round(rng.normal(*params["sw"]), 1)),
                "petal_length": max(1.0, round(rng.normal(*params["pl"]), 1)),
                "petal_width":  max(0.1, round(rng.normal(*params["pw"]), 1)),
                "species": species,
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Dota 2 Pro Matches
# ---------------------------------------------------------------------------
def generate_dota2(n: int = 500) -> pd.DataFrame:
    """Synthetic Dota 2 professional match data."""
    heroes = [
        "Anti-Mage", "Axe", "Bane", "Bloodseeker", "Crystal Maiden",
        "Drow Ranger", "Earthshaker", "Juggernaut", "Mirana", "Morphling",
        "Shadow Fiend", "Phantom Lancer", "Puck", "Pudge", "Razor",
        "Sand King", "Storm Spirit", "Sven", "Tiny", "Vengeful Spirit",
        "Windranger", "Zeus", "Kunkka", "Lina", "Lion",
        "Shadow Shaman", "Slardar", "Tidehunter", "Witch Doctor", "Lich",
        "Riki", "Enigma", "Tinker", "Sniper", "Necrophos",
        "Warlock", "Beastmaster", "Queen of Pain", "Venomancer", "Faceless Void",
    ]
    teams = [
        "Team Secret", "OG", "Team Liquid", "Evil Geniuses", "PSG.LGD",
        "Virtus.pro", "Fnatic", "Natus Vincere", "Alliance", "Team Spirit",
        "Tundra Esports", "Gaimin Gladiators", "BetBoom Team", "9Pandas", "nouns",
    ]
    tournaments = [
        "The International 2024", "ESL One Birmingham", "DreamLeague Season 22",
        "Riyadh Masters 2024", "BetBoom Dacha", "PGL Wallachia",
    ]
    regions = ["EU", "NA", "CN", "SEA", "SA", "CIS"]

    rows = []
    for i in range(n):
        duration_min = max(15, rng.normal(38, 10))
        radiant_team = rng.choice(teams)
        dire_team = rng.choice([t for t in teams if t != radiant_team])
        radiant_win = bool(rng.random() < 0.52)

        radiant_score = max(5, int(rng.normal(30, 12)))
        dire_score = max(5, int(rng.normal(30, 12)))
        if radiant_win:
            radiant_score = max(radiant_score, dire_score + rng.integers(0, 10))
        else:
            dire_score = max(dire_score, radiant_score + rng.integers(0, 10))

        radiant_picks = list(rng.choice(heroes, size=5, replace=False))
        dire_picks = list(rng.choice([h for h in heroes if h not in radiant_picks], size=5, replace=False))

        rows.append({
            "match_id": 7_800_000_000 + i,
            "tournament": rng.choice(tournaments),
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
            "radiant_team": radiant_team,
            "dire_team": dire_team,
            "radiant_win": radiant_win,
            "duration_minutes": round(duration_min, 1),
            "radiant_score": radiant_score,
            "dire_score": dire_score,
            "first_blood_time_sec": max(10, int(rng.normal(120, 80))),
            "radiant_towers_destroyed": rng.integers(0, 12),
            "dire_towers_destroyed": rng.integers(0, 12),
            "radiant_pick_1": radiant_picks[0],
            "radiant_pick_2": radiant_picks[1],
            "radiant_pick_3": radiant_picks[2],
            "radiant_pick_4": radiant_picks[3],
            "radiant_pick_5": radiant_picks[4],
            "dire_pick_1": dire_picks[0],
            "dire_pick_2": dire_picks[1],
            "dire_pick_3": dire_picks[2],
            "dire_pick_4": dire_picks[3],
            "dire_pick_5": dire_picks[4],
            "radiant_gold_advantage_10min": int(rng.normal(0, 3000)),
            "radiant_xp_advantage_10min": int(rng.normal(0, 4000)),
            "region": rng.choice(regions),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Restaurant Data (intentionally messy)
# ---------------------------------------------------------------------------
def generate_restaurant(n: int = 500) -> pd.DataFrame:
    """Messy restaurant dataset — showcases data cleaning capabilities."""
    names = [
        "The Golden Fork", "Mama Rosa's Kitchen", "Sushi Palace", "Burger Barn",
        "Spice Route", "Le Petit Bistro", "Dragon Wok", "Taco Fiesta",
        "Olive Garden Express", "Pizza Planet", "Café Mocha", "The Grill House",
        "Noodle Bar", "Steakhouse 55", "Green Leaf Vegan", "Seafood Shack",
        "Curry House", "Ramen Republic", "BBQ Smoke House", "The Pancake Spot",
        "Pasta La Vista", "Wings & Things", "Pho Real", "Kebab King",
        "The Dumpling Den", "Falafel Factory", "Lobster Lounge", "Waffle World",
    ]
    cuisines_correct = [
        "Italian", "Mexican", "Japanese", "Chinese", "Indian",
        "French", "American", "Thai", "Mediterranean", "Korean",
    ]
    # Intentional misspellings & inconsistencies for data cleaning
    cuisines_messy = cuisines_correct + [
        "italian", "ITALIAN", "Itallian", "mexcian", "Japaneese",
        "chinese", "INDIAN", "french ", " American", "thai food",
    ]
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "San Francisco", "Seattle", "Miami", "Boston", "Denver",
    ]
    price_ranges = ["$", "$$", "$$$", "$$$$"]

    rows = []
    for i in range(n):
        name = rng.choice(names)
        # Sometimes duplicate name with slight variation
        if rng.random() < 0.08:
            name = name + " "  # trailing space
        if rng.random() < 0.05:
            name = name.upper()

        cuisine = rng.choice(cuisines_messy)
        city = rng.choice(cities)
        if rng.random() < 0.03:
            city = city.lower()  # inconsistent casing

        rating = round(rng.normal(3.8, 0.7), 1)
        rating = np.clip(rating, 1.0, 5.0)
        if rng.random() < 0.05:
            rating = np.nan  # some missing ratings

        reviews = max(0, int(rng.exponential(200)))
        if rng.random() < 0.03:
            reviews = -1  # invalid value

        price = rng.choice(price_ranges, p=[0.20, 0.40, 0.30, 0.10])

        avg_meal_cost = {
            "$": rng.normal(12, 3),
            "$$": rng.normal(25, 5),
            "$$$": rng.normal(50, 10),
            "$$$$": rng.normal(90, 20),
        }[price]
        avg_meal_cost = round(max(5, avg_meal_cost), 2)
        if rng.random() < 0.04:
            avg_meal_cost = np.nan

        delivery = rng.choice([True, False, "Yes", "No", "yes", None], p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05])
        outdoor_seating = rng.choice([True, False, "Y", "N", None], p=[0.25, 0.35, 0.15, 0.15, 0.1])

        phone = f"({rng.integers(200,999)}) {rng.integers(200,999)}-{rng.integers(1000,9999)}"
        if rng.random() < 0.08:
            phone = "N/A"
        if rng.random() < 0.05:
            phone = ""

        rows.append({
            "restaurant_name": name,
            "cuisine_type": cuisine,
            "city": city,
            "rating": rating,
            "number_of_reviews": reviews,
            "price_range": price,
            "avg_meal_cost": avg_meal_cost,
            "has_delivery": delivery,
            "has_outdoor_seating": outdoor_seating,
            "phone": phone,
            "date_opened": (
                pd.Timestamp("2005-01-01") + pd.Timedelta(days=int(rng.integers(0, 7000)))
            ).strftime("%Y-%m-%d") if rng.random() < 0.9 else (
                # Inconsistent date formats
                (pd.Timestamp("2005-01-01") + pd.Timedelta(days=int(rng.integers(0, 7000)))).strftime("%m/%d/%Y")
            ),
        })

    df = pd.DataFrame(rows)
    # Add some full duplicate rows
    n_dupes = int(n * 0.03)
    dupes = df.sample(n=n_dupes, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating demo datasets…\n")

    datasets = {
        "titanic.csv": generate_titanic,
        "iris.csv": generate_iris,
        "dota2_matches.csv": generate_dota2,
        "restaurant_data.csv": generate_restaurant,
    }

    for filename, generator in datasets.items():
        df = generator()
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"  [OK] {filename}: {df.shape[0]} rows x {df.shape[1]} columns -> {path}")

    print(f"\nAll datasets saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
