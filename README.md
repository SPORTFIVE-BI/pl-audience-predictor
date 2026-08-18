# Premier League Audience Predictor — deployment files

Five files. Replace all five in your GitHub repo, commit, and Streamlit Cloud redeploys automatically.

| File | Change since last deploy |
|---|---|
| `model_v5.pkl` | **New.** Trained on five complete seasons (2021/22–2025/26, 1,900 matches) |
| `predict.py` | **Rewritten.** Fixes a bug that was costing ~10 accuracy points — see below |
| `app_v2.py` | Basket ranges now read from the model file instead of being hardcoded |
| `league_tables.xlsx` | Unchanged |
| `requirements.txt` | Unchanged (scikit-learn pinned to 1.8.0) |

**Delete `model_v3_clean.pkl` from the repo** once this is deployed — nothing references it.

## Bugs fixed in this release

An audit compared every feature as computed at inference against the value the model was trained on, using 380 real fixtures. Four mismatches turned up.

| Bug | What was wrong | Effect |
|---|---|---|
| Stature features never computed | All four passed as blanks, including `stature_rank_in_day` — the model's 3rd most important input | Large |
| `team_avg` averaged across regions | Made `max_competing_stature` and `slate_concentration_index` 4x too small | Small |
| Gameweek from `days // 7` | Overshot by ~2 gameweeks; the season spans ~40 weeks with breaks | Negligible |
| `is_opening_weekend` / `is_final_day` | Derived from the wrong gameweek | Negligible |

Nothing crashed, because the model tolerates missing values — which is why this went unnoticed.

Measured on 2024/25 with a model that had never seen that season, holding the model constant and varying only the inference code:

| Inference code | Global wMAPE |
|---|---|
| As deployed | 35.6% |
| + gameweek fixed | 36.0% |
| + stature features, wrong scale | 27.3% |
| **All correct (ships now)** | **26.5%** |

**A 9 point improvement**, essentially all of it from computing the stature features at all. Every feature now reproduces exactly except gameweek, which is right 74% of the time and within one gameweek 91% of the time — the residual can't be removed without the published fixture list, and its influence is negligible.

Where the gain lands:

| Slot | Before | After |
|---|---|---|
| Midweek 19:30 | 51.5% | 29.4% |
| Saturday 15:00 | 58.8% | 40.7% |
| Sunday 14:00 | 37.7% | 30.0% |
| Sunday 16:30 | 14.6% | 11.5% |
| Monday/Friday 20:00 | 16.0% | 16.3% |

The slots that improve most are the ones where knowing a fixture's standing among the day's other matches matters — exactly what the stature features encode. Package totals also went from over-predicting by 6-9% to within 4%.

## How the stature features work

They describe how a fixture compares with the other matches on the same day — whether it's the marquee pick or one of several competing for attention.

- **`predict_basket()`** groups fixtures by date and computes them from the fixtures you supply. Upload a full matchday and they're exact.
- **`predict_audience()`** for a single fixture approximates: it assumes the `n_concurrent` other matches are of league-average stature. Good enough for most use, and you can pass `other_fixtures_today=[("Arsenal","Chelsea"), ...]` to make it exact.

## Basket ranges

`BASKET_SPREAD` now lives inside the model file and is regenerated on every retrain, so the app's uncertainty ranges stay in step with the model automatically.

Current calibration, measured out-of-sample:

| Package size | 80% range |
|---|---|
| 5 fixtures | −25% to +29% |
| 10 fixtures | −18% to +20% |
| 20 fixtures | −14% to +12% |
| 38 fixtures | −10% to +8% |

## Accuracy

Leave-one-season-out cross-validation — each season predicted by a model trained on the other four:

| Held-out season | Global wMAPE |
|---|---|
| 2021/22 | 25.1% |
| 2022/23 | 31.4% |
| 2023/24 | 26.7% |
| 2024/25 | 26.5% |
| 2025/26 | 30.1% |
| **Mean** | **27.9%** |

Expect roughly 28–30% on 2026/27. Excluding Saturday 3pm fixtures it's about 22%.

For a single fixture the median error is 25%: half of predictions land within a quarter of the actual figure. Package totals are far tighter — that's the point of the basket ranges above.

## Deploying

1. Download all five files
2. In your GitHub repo, delete the old `app_v2.py`, `predict.py`, `model_v3_clean.pkl`
3. Upload the five files, commit
4. Streamlit Cloud rebuilds in a minute or two

To test locally first:

```
pip install -r requirements.txt
streamlit run app_v2.py
```

Running `python predict.py` on its own prints a few sample predictions — a quick way to confirm the model loads before touching the app.

## Next retrain

When 2026/27 finishes, rerun `build_iteration2_features.py` then `train_v5.py` (both in `iteration_2_full_season/`) with the new season's file added. Check the scikit-learn version at that point matches the pin in `requirements.txt`, or update both together.
