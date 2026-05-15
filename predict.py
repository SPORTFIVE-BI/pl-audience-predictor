"""
predict.py — INFERENCE INTERFACE
================================
This is the file your web platform calls to make predictions.

USAGE
-----
from predict import predict_audience

# Predict for a single fixture (short horizon — all info known)
result = predict_audience(
    home_team="Manchester United",
    away_team="Liverpool",
    season="2025/26",
    match_date="2025-12-14",
    kick_off_uk="16:30",
    n_concurrent=0,
)
# Returns a dict: {region: {p10, p50, p90}} for all 8 regions

# Predict for a fixture before slot is set (long horizon)
result = predict_audience(
    home_team="Manchester United",
    away_team="Liverpool",
    season="2025/26",
    match_date="2025-12-14",
    # kick_off_uk and n_concurrent omitted → model predicts with masked features
)

# Predict for a list of fixtures (returns a DataFrame)
results = predict_basket(fixtures_list)
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = Path(__file__).parent / "model_v3_clean.pkl"
LEAGUE_TABLES_PATH = Path(__file__).parent / "league_tables.xlsx"  # for prior-season lookups

with open(MODEL_PATH, "rb") as f:
    _A = pickle.load(f)
_model_p10 = _A["model_p10"]
_model_p50 = _A["model_p50"]
_model_p90 = _A["model_p90"]
_encoders = _A["encoders"]
_feature_list = _A["feature_list"]
_categorical = _A["categorical_features"]

REGIONS = ["uk","europe_excl_uk","asia_pacific","latin_america",
           "mena","north_america","sub_saharan_africa","global_audience"]

# Team tiers (must match training)
TEAM_TIERS = {
    "Liverpool": 1, "Manchester United": 1, "Manchester City": 1,
    "Arsenal": 1, "Tottenham Hotspur": 1, "Chelsea": 1,
    "Newcastle United": 2, "Aston Villa": 2, "West Ham United": 2,
    "Everton": 2, "Leeds United": 2, "Brighton & Hove Albion": 2,
    "Leicester City": 2,
}
def get_tier(team): return TEAM_TIERS.get(team, 3)

def norm_pair(a, b): return tuple(sorted([a, b]))

DERBIES = {
    norm_pair("Manchester United", "Liverpool"): 1,
    norm_pair("Arsenal", "Tottenham Hotspur"): 1,
    norm_pair("Newcastle United", "Sunderland"): 1,
    norm_pair("Manchester City", "Manchester United"): 2,
    norm_pair("Liverpool", "Everton"): 2,
    norm_pair("Aston Villa", "Birmingham City"): 2,
    norm_pair("Millwall", "West Ham United"): 2,
    norm_pair("Norwich City", "Ipswich Town"): 2,
    norm_pair("Brighton & Hove Albion", "Crystal Palace"): 3,
    norm_pair("Leeds United", "Manchester United"): 3,
    norm_pair("Tottenham Hotspur", "Chelsea"): 3,
}

REGION_TZ_HOURS = {
    "uk": 0, "europe_excl_uk": +1, "asia_pacific": +7,
    "latin_america": -4, "mena": +3, "north_america": -5,
    "sub_saharan_africa": +2, "global_audience": 0,
}

# Load prior-season tables once at import time
def _load_prior_tables():
    """Build a lookup of (season, team) -> (position, points, gd)."""
    SEASON_ORDER = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]
    tables = {}
    # Manually-entered 2020/21
    p2020 = [
        (1,"Manchester City",86,51),(2,"Manchester United",74,29),(3,"Liverpool",69,26),
        (4,"Chelsea",67,22),(5,"Leicester City",66,18),(6,"West Ham United",65,15),
        (7,"Tottenham Hotspur",62,23),(8,"Arsenal",61,1),(9,"Leeds United",59,8),
        (10,"Everton",59,1),(11,"Aston Villa",55,9),(12,"Newcastle United",45,-16),
        (13,"Wolverhampton Wanderers",45,-9),(14,"Crystal Palace",44,-25),
        (15,"Southampton",43,-21),(16,"Brighton & Hove Albion",41,-6),
        (17,"Burnley",39,-22),(18,"Fulham",28,-26),
        (19,"West Bromwich Albion",26,-41),(20,"Sheffield United",23,-43),
    ]
    for pos, team, pts, gd in p2020:
        tables[("2020/21", team)] = (pos, pts, gd)
    # Load from Excel
    if LEAGUE_TABLES_PATH.exists():
        xl = pd.ExcelFile(LEAGUE_TABLES_PATH)
        sheet_map = {"2021-22":"2021/22","2022-23":"2022/23","2023-24":"2023/24","2024-25":"2024/25"}
        for sheet, season in sheet_map.items():
            t = pd.read_excel(xl, sheet_name=sheet)
            for _, r in t.iterrows():
                team = str(r["Team"]).replace("(C)","").replace("(R)","").strip()
                team = team.split(" (")[0].strip()
                pts = int(str(r["Pts"]).split("[")[0])
                gd = int(str(r["GD"]).replace("−","-"))
                tables[(season, team)] = (int(r["Pos"]), pts, gd)
    return tables

_PRIOR_TABLES = _load_prior_tables()

def _get_prior(season, team):
    """Look up a team's prior-season stats; returns (pos, pts, gd, newly_promoted)."""
    SEASON_ORDER = ["2020/21","2021/22","2022/23","2023/24","2024/25","2025/26"]
    if season not in SEASON_ORDER or SEASON_ORDER.index(season) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    prior = SEASON_ORDER[SEASON_ORDER.index(season) - 1]
    if (prior, team) in _PRIOR_TABLES:
        pos, pts, gd = _PRIOR_TABLES[(prior, team)]
        return (pos, pts, gd, 0)
    return (21, 35, -10, 1)  # newly promoted

def _build_feature_row(home_team, away_team, season, match_date, region,
                       kick_off_uk=None, n_concurrent=None):
    """Build one row of features for one (match, region)."""
    md = pd.Timestamp(match_date)
    day_of_week = md.day_name() if match_date else None
    month = md.month
    day = md.day

    # Calendar flags
    is_opening_weekend = 1 if (month == 8 and day <= 20) else 0
    is_final_day = 1 if (month == 5 and day >= 20) else 0
    is_festive_period = 1 if ((month == 12 and day >= 23) or (month == 1 and day <= 2)) else 0
    is_boxing_day = 1 if (month == 12 and day == 26) else 0
    def _post_int(d):
        m, day = d.month, d.day
        if m == 9 and 8 <= day <= 15: return 1
        if m == 10 and day <= 8: return 1
        if m == 11 and 18 <= day <= 25: return 1
        if m == 3 and 22 <= day <= 29: return 1
        return 0
    is_post_international = _post_int(md)
    gameweek = max(1, min(38, (md - pd.Timestamp(f"{md.year if month >= 7 else md.year-1}-08-01")).days // 7))

    # Slot-related (maskable)
    if kick_off_uk is not None:
        h, m = kick_off_uk.split(":")
        ko_min = int(h) * 60 + int(m)
        # Slot group
        if ko_min in range(11*60, 13*60+1): slot_group = "12:30_sat_early"
        elif ko_min in range(13*60+30, 14*60+30): slot_group = "14:00_sun_afternoon"
        elif ko_min in range(14*60+30, 16*60+1): slot_group = "15:00_sat_blackout"
        elif ko_min in range(16*60+15, 16*60+45): slot_group = "16:30_sun_marquee"
        elif ko_min in range(17*60, 18*60+1): slot_group = "17:30"
        elif ko_min in range(18*60+15, 18*60+50): slot_group = "18:30"
        elif ko_min in range(19*60, 20*60): slot_group = "19:30_midweek"
        elif ko_min in range(20*60, 20*60+30): slot_group = "20:00_evening"
        else: slot_group = f"other_{kick_off_uk}"
        local_kick_off_hour = ((ko_min / 60) + REGION_TZ_HOURS[region]) % 24
        is_uk_blackout = 1 if (day_of_week == "Saturday" and 14*60+45 <= ko_min <= 17*60+15 and region == "uk") else 0
    else:
        ko_min = np.nan
        slot_group = None
        local_kick_off_hour = np.nan
        is_uk_blackout = 0

    # Team tiers
    home_tier = get_tier(home_team)
    away_tier = get_tier(away_team)
    pair_min_tier = min(home_tier, away_tier)

    # Prior-season strength
    h_pos, h_pts, h_gd, h_promoted = _get_prior(season, home_team)
    a_pos, a_pts, a_gd, a_promoted = _get_prior(season, away_team)
    pair_strength_points = (h_pts or 0) + (a_pts or 0)
    pair_strength_min_pos = min(h_pos, a_pos) if not pd.isna(h_pos) else 21
    both_top6 = 1 if (h_pos <= 6 and a_pos <= 6) else 0
    both_bottom6 = 1 if (h_pos >= 15 and a_pos >= 15) else 0

    # Derby
    derby_tier = DERBIES.get(norm_pair(home_team, away_team), 0)

    return {
        "season": season, "region": region,
        "home_team": home_team, "away_team": away_team,
        "day_of_week": day_of_week if day_of_week else None,
        "month": month, "gameweek": gameweek,
        "is_opening_weekend": is_opening_weekend,
        "is_final_day": is_final_day,
        "is_festive_period": is_festive_period,
        "is_boxing_day": is_boxing_day,
        "is_post_international": is_post_international,
        "kick_off_uk_minutes": ko_min,
        "slot_group": slot_group,
        "local_kick_off_hour": local_kick_off_hour,
        "n_concurrent": n_concurrent if n_concurrent is not None else np.nan,
        "is_uk_blackout": is_uk_blackout,
        "home_tier": home_tier, "away_tier": away_tier,
        "pair_min_tier": pair_min_tier,
        "home_prior_pos": h_pos, "away_prior_pos": a_pos,
        "home_prior_pts": h_pts, "away_prior_pts": a_pts,
        "home_prior_gd": h_gd, "away_prior_gd": a_gd,
        "home_newly_promoted": h_promoted, "away_newly_promoted": a_promoted,
        "pair_strength_points": pair_strength_points,
        "pair_strength_min_pos": pair_strength_min_pos,
        "both_top6": both_top6,
        "both_bottom6_or_promoted": both_bottom6,
        "derby_tier": derby_tier,
    }

def _encode_row(row_dict):
    """Encode categorical features using the training-time encoders."""
    out = row_dict.copy()
    for col in _categorical:
        v = out.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[col] = -1
        else:
            try:
                out[col] = int(_encoders[col].transform([[v]])[0][0])
            except Exception:
                out[col] = -1
    return out

def predict_audience(home_team, away_team, season, match_date,
                     kick_off_uk=None, n_concurrent=None):
    """Predict audience across all regions for a single fixture.
    
    Returns dict: {region: {'p10': lower, 'p50': point estimate, 'p90': upper}}
    
    Args:
        home_team, away_team: team names (must match training data spelling)
        season: e.g. "2025/26"
        match_date: ISO date string, e.g. "2025-12-14"
        kick_off_uk: optional, e.g. "16:30" — if None, long-horizon prediction
        n_concurrent: optional int — if None, long-horizon prediction
    """
    results = {}
    for region in REGIONS:
        row = _build_feature_row(home_team, away_team, season, match_date,
                                  region, kick_off_uk, n_concurrent)
        encoded = _encode_row(row)
        X = pd.DataFrame([{k: encoded.get(k) for k in _feature_list}])
        # Reorder columns to match training
        X = X[_feature_list]
        p10 = float(np.exp(_model_p10.predict(X)[0]))
        p50 = float(np.exp(_model_p50.predict(X)[0]))
        p90 = float(np.exp(_model_p90.predict(X)[0]))
        # Fix quantile crossing — ensure p10 <= p50 <= p90
        p10, p50, p90 = sorted([p10, p50, p90])
        results[region] = {"p10": p10, "p50": p50, "p90": p90}
    return results

def predict_basket(fixtures):
    """Predict for a list of fixtures.
    
    Args:
        fixtures: list of dicts, each with keys: home_team, away_team, season,
                  match_date, optionally kick_off_uk and n_concurrent
    
    Returns:
        DataFrame with one row per (fixture, region) and p10/p50/p90 columns,
        plus a 'package_total' summary row per region.
    """
    rows = []
    for f in fixtures:
        preds = predict_audience(
            home_team=f["home_team"], away_team=f["away_team"],
            season=f["season"], match_date=f["match_date"],
            kick_off_uk=f.get("kick_off_uk"),
            n_concurrent=f.get("n_concurrent"),
        )
        for region, vals in preds.items():
            rows.append({
                "home_team": f["home_team"], "away_team": f["away_team"],
                "season": f["season"], "match_date": f["match_date"],
                "region": region,
                "p10": vals["p10"], "p50": vals["p50"], "p90": vals["p90"],
            })
    df = pd.DataFrame(rows)
    # Add package totals
    totals = df.groupby("region")[["p10","p50","p90"]].sum().reset_index()
    totals["home_team"] = "PACKAGE TOTAL"
    totals["away_team"] = f"{len(fixtures)} fixtures"
    totals["season"] = ""
    totals["match_date"] = ""
    df = pd.concat([df, totals[df.columns]], ignore_index=True)
    return df

# ============================================================
# QUICK SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("Testing inference...\n")

    # Test 1: short horizon (Sunday marquee fixture, no concurrent matches)
    print("=== Test 1: Liverpool vs Man Utd, 2025-12-14 Sun 16:30, no concurrent ===")
    r = predict_audience("Liverpool", "Manchester United", "2025/26", "2025-12-14",
                          kick_off_uk="16:30", n_concurrent=0)
    for region, p in r.items():
        print(f"  {region:<22} P10={p['p10']/1e6:>5.2f}M  P50={p['p50']/1e6:>5.2f}M  P90={p['p90']/1e6:>5.2f}M")

    # Test 2: long horizon (same fixture, slot unknown)
    print("\n=== Test 2: Same fixture, long-horizon (slot unknown) ===")
    r = predict_audience("Liverpool", "Manchester United", "2025/26", "2025-12-14")
    for region, p in r.items():
        print(f"  {region:<22} P10={p['p10']/1e6:>5.2f}M  P50={p['p50']/1e6:>5.2f}M  P90={p['p90']/1e6:>5.2f}M")

    # Test 3: a smaller fixture in the Saturday 3pm blackout
    print("\n=== Test 3: Brentford vs Bournemouth, Sat 15:00 (5 concurrent matches) ===")
    r = predict_audience("Brentford", "AFC Bournemouth", "2025/26", "2025-10-25",
                          kick_off_uk="15:00", n_concurrent=5)
    for region, p in r.items():
        print(f"  {region:<22} P10={p['p10']/1e6:>5.2f}M  P50={p['p50']/1e6:>5.2f}M  P90={p['p90']/1e6:>5.2f}M")

    # Test 4: basket of three fixtures
    print("\n=== Test 4: Basket of 3 fixtures ===")
    fixtures = [
        {"home_team":"Manchester City","away_team":"Arsenal","season":"2025/26",
         "match_date":"2025-11-08","kick_off_uk":"16:30","n_concurrent":0},
        {"home_team":"Liverpool","away_team":"Chelsea","season":"2025/26",
         "match_date":"2025-11-09","kick_off_uk":"17:30","n_concurrent":1},
        {"home_team":"Brighton & Hove Albion","away_team":"Crystal Palace","season":"2025/26",
         "match_date":"2025-11-08","kick_off_uk":"15:00","n_concurrent":4},
    ]
    df = predict_basket(fixtures)
    print(df[df["region"] == "global_audience"][["home_team","away_team","p10","p50","p90"]].to_string(index=False))
