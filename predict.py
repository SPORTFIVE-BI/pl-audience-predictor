"""
predict.py — INFERENCE INTERFACE (model_v5)
===========================================

Loads model_v5.pkl (trained on five complete seasons, 2021/22-2025/26).

IMPORTANT CHANGE vs the previous version
----------------------------------------
The earlier predict.py never computed the four "stature" features that were
added in v3, so it passed blanks for them — including stature_rank_in_day,
the model's third most important input. Measured cost: about 10 percentage
points of accuracy. This version computes them.

Those features describe how a fixture compares with the OTHER matches played
on the same day. At inference we don't always know the full slate, so:

  - predict_basket() groups the fixtures you supply by date and computes the
    features from that group. Give it a whole matchday and they'll be exact.
  - predict_audience() for a single fixture approximates: it assumes the
    n_concurrent other matches are of league-average stature.
    Pass `other_fixtures_today` to make it exact.

USAGE
-----
from predict import predict_audience, predict_basket, basket_range

result = predict_audience(
    home_team="Liverpool", away_team="Manchester United",
    season="2026/27", match_date="2026-12-12",
    kick_off_uk="16:30", n_concurrent=0,
)
# -> {region: {"p10":..., "p50":..., "p90":...}}

# Exact stature features when you know the rest of the day's card:
result = predict_audience(
    ..., other_fixtures_today=[("Arsenal","Chelsea"), ("Everton","Burnley")]
)
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = Path(__file__).parent / "model_v5.pkl"
LEAGUE_TABLES_PATH = Path(__file__).parent / "league_tables.xlsx"

with open(MODEL_PATH, "rb") as f:
    _A = pickle.load(f)

_model_p10 = _A["model_p10"]
_model_p50 = _A["model_p50"]
_model_p90 = _A["model_p90"]
_encoders = _A["encoders"]
_feature_list = _A["feature_list"]
_categorical = _A["categorical_features"]
_team_avg = _A["team_avg_for_stature"]          # team -> mean global audience
_median_team_avg = _A["median_team_avg"]
BASKET_SPREAD = {int(k): (float(v[0]), float(v[1]))
                 for k, v in _A.get("basket_spread", {}).items()}
_GW_CAL = {int(k): float(v) for k, v in _A.get("gameweek_calibration", {}).items()}

REGIONS = ["uk", "europe_excl_uk", "asia_pacific", "latin_america",
           "mena", "north_america", "sub_saharan_africa", "global_audience"]

REGION_TZ_HOURS = {
    "uk": 0, "europe_excl_uk": +1, "asia_pacific": +7, "latin_america": -4,
    "mena": +3, "north_america": -5, "sub_saharan_africa": +2, "global_audience": 0,
}

TEAM_TIERS = {
    "Liverpool": 1, "Manchester United": 1, "Manchester City": 1,
    "Arsenal": 1, "Tottenham Hotspur": 1, "Chelsea": 1,
    "Newcastle United": 2, "Aston Villa": 2, "West Ham United": 2,
    "Everton": 2, "Leeds United": 2, "Brighton & Hove Albion": 2,
    "Leicester City": 2,
}
def _tier(team): return TEAM_TIERS.get(team, 3)

def _pair(a, b): return tuple(sorted([a, b]))

DERBIES = {
    _pair("Manchester United", "Liverpool"): 1,
    _pair("Arsenal", "Tottenham Hotspur"): 1,
    _pair("Newcastle United", "Sunderland"): 1,
    _pair("Manchester City", "Manchester United"): 2,
    _pair("Liverpool", "Everton"): 2,
    _pair("Aston Villa", "Birmingham City"): 2,
    _pair("Millwall", "West Ham United"): 2,
    _pair("Norwich City", "Ipswich Town"): 2,
    _pair("Brighton & Hove Albion", "Crystal Palace"): 3,
    _pair("Leeds United", "Manchester United"): 3,
    _pair("Tottenham Hotspur", "Chelsea"): 3,
}

SEASON_ORDER = ["2020/21", "2021/22", "2022/23", "2023/24",
                "2024/25", "2025/26", "2026/27", "2027/28"]


# ------------------------------------------------------------------
# Prior-season league tables
# ------------------------------------------------------------------
def _load_prior_tables():
    tables = {}
    for pos, team, pts, gd in [
        (1,"Manchester City",86,51),(2,"Manchester United",74,29),(3,"Liverpool",69,26),
        (4,"Chelsea",67,22),(5,"Leicester City",66,18),(6,"West Ham United",65,15),
        (7,"Tottenham Hotspur",62,23),(8,"Arsenal",61,1),(9,"Leeds United",59,8),
        (10,"Everton",59,1),(11,"Aston Villa",55,9),(12,"Newcastle United",45,-16),
        (13,"Wolverhampton Wanderers",45,-9),(14,"Crystal Palace",44,-25),
        (15,"Southampton",43,-21),(16,"Brighton & Hove Albion",41,-6),
        (17,"Burnley",39,-22),(18,"Fulham",28,-26),
        (19,"West Bromwich Albion",26,-41),(20,"Sheffield United",23,-43)]:
        tables[("2020/21", team)] = (pos, pts, gd)

    if LEAGUE_TABLES_PATH.exists():
        xl = pd.ExcelFile(LEAGUE_TABLES_PATH)
        sheets = {"2021-22":"2021/22","2022-23":"2022/23",
                  "2023-24":"2023/24","2024-25":"2024/25"}
        for sheet, season in sheets.items():
            if sheet not in xl.sheet_names:
                continue
            t = pd.read_excel(xl, sheet_name=sheet)
            for _, r in t.iterrows():
                team = str(r["Team"]).split("(")[0].strip()
                pts = int(str(r["Pts"]).split("[")[0])
                gd = int(str(r["GD"]).replace("−", "-"))
                tables[(season, team)] = (int(r["Pos"]), pts, gd)
    return tables

_PRIOR_TABLES = _load_prior_tables()


def _get_prior(season, team):
    """(position, points, goal difference, newly_promoted_flag) for the season before."""
    if season not in SEASON_ORDER or SEASON_ORDER.index(season) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    prior = SEASON_ORDER[SEASON_ORDER.index(season) - 1]
    if (prior, team) in _PRIOR_TABLES:
        pos, pts, gd = _PRIOR_TABLES[(prior, team)]
        return (pos, pts, gd, 0)
    return (21, 35, -10, 1)          # newly promoted / unknown


# ------------------------------------------------------------------
# Stature features
# ------------------------------------------------------------------
def _pair_stature(home, away):
    """Proxy for how big a fixture is: sum of each club's historic mean audience."""
    return (_team_avg.get(home, _median_team_avg) +
            _team_avg.get(away, _median_team_avg))


def _stature_features(home, away, n_concurrent=None, other_fixtures_today=None):
    """
    Returns the four day-level stature features.

    other_fixtures_today : list of (home, away) for the rest of the day's card.
                           Exact when supplied.
    n_concurrent         : used to approximate the slate when the card is unknown —
                           assumes the other matches are of league-average stature.
    """
    mine = _pair_stature(home, away)

    if other_fixtures_today:
        others = [_pair_stature(h, a) for h, a in other_fixtures_today]
    elif n_concurrent and n_concurrent > 0:
        # Approximation: assume typical opposition on the same slate.
        others = [2 * _median_team_avg] * int(n_concurrent)
    else:
        others = []

    if not others:
        return {"max_competing_stature": 0.0,
                "stature_ratio_to_day": 1.0,
                "stature_rank_in_day": 1,
                "slate_concentration_index": 0.0}

    day = others + [mine]
    day_mean = float(np.mean(day))
    return {
        "max_competing_stature": float(max(others)),
        "stature_ratio_to_day": float(mine / day_mean) if day_mean > 0 else 1.0,
        "stature_rank_in_day": int(sum(1 for v in day if v >= mine)),
        "slate_concentration_index": float(np.std(day, ddof=1)) if len(day) > 1 else 0.0,
    }


# ------------------------------------------------------------------
# Feature row
# ------------------------------------------------------------------
def _slot_group(kick_off_uk):
    t = str(kick_off_uk)[:5]
    if t in ["11:00","11:30","12:00","12:30","13:00"]: return "12:30_sat_early"
    if t in ["13:30","14:00","14:15","14:05"]: return "14:00_sun_afternoon"
    if t in ["14:30","15:00","15:30","15:45","16:00","15:15"]: return "15:00_sat_blackout"
    if t in ["16:15","16:30"]: return "16:30_sun_marquee"
    if t in ["17:15","17:30","18:00"]: return "17:30"
    if t in ["18:30","18:45"]: return "18:30"
    if t in ["19:00","19:15","19:30","19:45","19:40"]: return "19:30_midweek"
    if t in ["20:00","20:15"]: return "20:00_evening"
    return f"other_{t}"


def _build_feature_row(home_team, away_team, season, match_date, region,
                        kick_off_uk=None, n_concurrent=None,
                        other_fixtures_today=None):
    md = pd.Timestamp(match_date)
    day_of_week = md.day_name()
    month, day = md.month, md.day

    # Gameweek: matched against a calibration table built from five seasons.
    # A naive days//7 overshoots by ~2 gameweeks because the season spans about
    # 40 weeks with international breaks.
    season_start_year = md.year if month >= 7 else md.year - 1
    days_since_aug1 = (md - pd.Timestamp(f"{season_start_year}-08-01")).days
    if _GW_CAL:
        gameweek = int(min(_GW_CAL, key=lambda g: abs(_GW_CAL[g] - days_since_aug1)))
    else:
        gameweek = int(np.clip(days_since_aug1 // 7, 1, 38))

    is_opening_weekend = int(gameweek == 1)
    is_final_day = int(gameweek == 38)
    is_festive_period = int((month == 12 and day >= 23) or (month == 1 and day <= 2))
    is_boxing_day = int(month == 12 and day == 26)
    is_post_international = int(
        (month == 9 and 8 <= day <= 15) or (month == 10 and day <= 8) or
        (month == 11 and 18 <= day <= 25) or (month == 3 and 22 <= day <= 29))

    if kick_off_uk is not None:
        h, m = str(kick_off_uk)[:5].split(":")
        ko_min = int(h) * 60 + int(m)
        slot = _slot_group(kick_off_uk)
        local_hour = ((ko_min / 60) + REGION_TZ_HOURS[region]) % 24
        is_uk_blackout = int(day_of_week == "Saturday" and
                              14*60 + 45 <= ko_min <= 17*60 + 15 and region == "uk")
    else:
        ko_min, slot, local_hour, is_uk_blackout = np.nan, None, np.nan, 0

    h_tier, a_tier = _tier(home_team), _tier(away_team)
    h_pos, h_pts, h_gd, h_promo = _get_prior(season, home_team)
    a_pos, a_pts, a_gd, a_promo = _get_prior(season, away_team)

    stature = _stature_features(home_team, away_team, n_concurrent, other_fixtures_today)

    row = {
        "season": season, "region": region,
        "home_team": home_team, "away_team": away_team,
        "day_of_week": day_of_week,
        "month": month, "gameweek": gameweek,
        "is_opening_weekend": is_opening_weekend,
        "is_final_day": is_final_day,
        "is_festive_period": is_festive_period,
        "is_boxing_day": is_boxing_day,
        "is_post_international": is_post_international,
        "kick_off_uk_minutes": ko_min,
        "slot_group": slot,
        "local_kick_off_hour": local_hour,
        "n_concurrent": n_concurrent if n_concurrent is not None else np.nan,
        "is_uk_blackout": is_uk_blackout,
        "home_tier": h_tier, "away_tier": a_tier,
        "pair_min_tier": min(h_tier, a_tier),
        "home_prior_pos": h_pos, "away_prior_pos": a_pos,
        "home_prior_pts": h_pts, "away_prior_pts": a_pts,
        "home_prior_gd": h_gd, "away_prior_gd": a_gd,
        "home_newly_promoted": h_promo, "away_newly_promoted": a_promo,
        "pair_strength_points": (h_pts or 0) + (a_pts or 0),
        "pair_strength_min_pos": min(h_pos, a_pos) if not pd.isna(h_pos) else 21,
        "both_top6": int(h_pos <= 6 and a_pos <= 6) if not pd.isna(h_pos) else 0,
        "both_bottom6_or_promoted": int(h_pos >= 15 and a_pos >= 15) if not pd.isna(h_pos) else 0,
        "derby_tier": DERBIES.get(_pair(home_team, away_team), 0),
    }
    row.update(stature)
    return row


def _encode(row):
    out = dict(row)
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


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def predict_audience(home_team, away_team, season, match_date,
                     kick_off_uk=None, n_concurrent=None,
                     other_fixtures_today=None):
    """Predict audience across all regions for one fixture.

    Returns {region: {"p10":…, "p50":…, "p90":…}}.
    Omit kick_off_uk / n_concurrent for a long-horizon estimate (wider range).
    """
    results = {}
    for region in REGIONS:
        row = _build_feature_row(home_team, away_team, season, match_date, region,
                                  kick_off_uk, n_concurrent, other_fixtures_today)
        X = pd.DataFrame([_encode(row)])[_feature_list]
        p10 = float(np.exp(_model_p10.predict(X)[0]))
        p50 = float(np.exp(_model_p50.predict(X)[0]))
        p90 = float(np.exp(_model_p90.predict(X)[0]))
        p10, p50, p90 = sorted([p10, p50, p90])       # guard against quantile crossing
        results[region] = {"p10": p10, "p50": p50, "p90": p90}
    return results


def basket_range(p50_sum, n):
    """Realistic P10/P90 for a basket total of n fixtures.

    Uses spreads measured empirically from out-of-sample backtests. Never sum
    individual P10s/P90s — that assumes every fixture misses in the same
    direction at once and gives ranges three to four times too wide.
    """
    if not BASKET_SPREAD:
        return p50_sum * 0.85, p50_sum * 1.15
    sizes = sorted(BASKET_SPREAD)
    if n <= sizes[0]:
        lo, hi = BASKET_SPREAD[sizes[0]]
    elif n >= sizes[-1]:
        lo, hi = BASKET_SPREAD[sizes[-1]]
    else:
        for i in range(len(sizes) - 1):
            if sizes[i] <= n <= sizes[i + 1]:
                t = (n - sizes[i]) / (sizes[i + 1] - sizes[i])
                la, ha = BASKET_SPREAD[sizes[i]]
                lb, hb = BASKET_SPREAD[sizes[i + 1]]
                lo, hi = la + t * (lb - la), ha + t * (hb - ha)
                break
    return p50_sum * lo, p50_sum * hi


def predict_basket(fixtures):
    """Predict for a list of fixture dicts.

    Each dict needs home_team, away_team, season, match_date; optionally
    kick_off_uk and n_concurrent.

    Fixtures sharing a match_date are treated as the same day's card, so the
    stature features are computed exactly rather than approximated.

    Returns a DataFrame with one row per (fixture, region), plus PACKAGE TOTAL
    rows whose p10/p90 use the empirical basket spread.
    """
    fixtures = list(fixtures)

    by_date = {}
    for f in fixtures:
        by_date.setdefault(str(f["match_date"]), []).append(f)

    rows = []
    for f in fixtures:
        same_day = by_date.get(str(f["match_date"]), [])
        others = [(o["home_team"], o["away_team"]) for o in same_day if o is not f]
        preds = predict_audience(
            home_team=f["home_team"], away_team=f["away_team"],
            season=f["season"], match_date=f["match_date"],
            kick_off_uk=f.get("kick_off_uk"),
            n_concurrent=f.get("n_concurrent"),
            other_fixtures_today=others if others else None,
        )
        for region, v in preds.items():
            rows.append({"home_team": f["home_team"], "away_team": f["away_team"],
                         "season": f["season"], "match_date": f["match_date"],
                         "region": region, **v})

    df = pd.DataFrame(rows)
    n = len(fixtures)
    totals = df.groupby("region")["p50"].sum().reset_index()
    lo_hi = totals["p50"].apply(lambda s: basket_range(s, n))
    totals["p10"] = [x[0] for x in lo_hi]
    totals["p90"] = [x[1] for x in lo_hi]
    totals["home_team"] = "PACKAGE TOTAL"
    totals["away_team"] = f"{n} fixtures"
    totals["season"] = ""
    totals["match_date"] = ""
    return pd.concat([df, totals[df.columns]], ignore_index=True)


if __name__ == "__main__":
    print("Model:", _A.get("trained_on", "unknown"))
    print()
    r = predict_audience("Liverpool", "Manchester United", "2026/27",
                          "2026-12-12", kick_off_uk="16:30", n_concurrent=0)
    print("Liverpool v Manchester United, Sun 16:30, no concurrent matches")
    for k, v in r.items():
        print(f"  {k:<20} {v['p50']/1e6:>6.2f}M  ({v['p10']/1e6:.2f}–{v['p90']/1e6:.2f}M)")

    print()
    r2 = predict_audience("Brentford", "AFC Bournemouth", "2026/27",
                           "2026-10-24", kick_off_uk="15:00", n_concurrent=5)
    print("Brentford v Bournemouth, Sat 15:00, 5 concurrent matches")
    g = r2["global_audience"]
    print(f"  Global {g['p50']/1e6:.2f}M ({g['p10']/1e6:.2f}–{g['p90']/1e6:.2f}M)")

    print()
    basket = [
        {"home_team":"Manchester City","away_team":"Arsenal","season":"2026/27",
         "match_date":"2026-11-07","kick_off_uk":"16:30","n_concurrent":0},
        {"home_team":"Liverpool","away_team":"Chelsea","season":"2026/27",
         "match_date":"2026-11-08","kick_off_uk":"17:30","n_concurrent":1},
        {"home_team":"Brighton & Hove Albion","away_team":"Crystal Palace","season":"2026/27",
         "match_date":"2026-11-07","kick_off_uk":"15:00","n_concurrent":4},
    ]
    b = predict_basket(basket)
    t = b[(b["home_team"]=="PACKAGE TOTAL") & (b["region"]=="global_audience")].iloc[0]
    print(f"Basket of 3 — Global total {t['p50']/1e6:.2f}M "
          f"({t['p10']/1e6:.2f}–{t['p90']/1e6:.2f}M)")
