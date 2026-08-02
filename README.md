# VCT 2026 Champs Prediction Toolkit

A classical-statistics toolkit for predicting VALORANT Champions Tour 2026 match outcomes — map picks, bans, and win probabilities — built entirely on historical match data collected from [vlr.gg](https://vlr.gg). No machine learning; every prediction is derived from interpretable statistics (frequency tables, win rates, round differentials, and the log5 formula) (for now).

## Overview

This project merges VCT 2026 match data across all four regions (**Americas**, **EMEA**, **Pacific**, **China**) and multiple events (**Kickoff**, **Stage 1**, **Masters**) into a single dataset, then uses it to answer three questions for any upcoming matchup:

1. **Which maps will each team likely pick and ban?** (based on historical veto tendencies)
2. **What does the full veto sequence look like?** (simulated in true order — Bo3 or Bo5, patch-aware map pool)
3. **Who is more likely to win, and by what probability?** (log5 formula blended with head-to-head record)

## Project Structure

```
.
├── merge_vct_data.ipynb     # Merges all regional/event .xlsx files into one dataset
├── predict_match.ipynb      # Single-matchup prediction (picks, bans, win probability)
├── vct_analysis.ipynb       # Reusable toolkit: modular functions + analyze_matchup() wrapper
├── vct_analysis.py          # Same toolkit as an importable Python module
└── data/                    # (not included) raw .xlsx match files, one per region/event
```

## Data Source & Schema

Match data is manually collected from vlr.gg for each region/event as a `.xlsx` file. Every file shares the same 52-column schema:

| Category | Columns |
|---|---|
| Match metadata | `ID`, `Region`, `Tournament`, `Patch`, `Team 1`, `Team 2`, `Phase`, `Map Numbers` |
| Veto sequence | `Team1Ban1`, `Team2Ban1`, `Team1Pick1`, `Team2Pick1`, `Team1Ban2`, `Team2Ban2`, `Team1Pick2`, `Team2Pick2`, `Decider` |
| Per-map results (×5) | `Map N`, `Map N Winner`, `MNR1`, `MNR2`, `MNHalf`, `MNT1`, `MNT2` |

After merging, two additional columns are computed:

- **`MatchWinner`** — whichever team won more maps
- **`Team1MapsWon`** / **`Team2MapsWon`** — map win counts per team

## Setup

```bash
pip install pandas numpy matplotlib openpyxl
```

1. Collect your regional/event `.xlsx` files into one folder.
2. Run **`merge_vct_data.ipynb`** to produce a single merged dataset (`VCT2026-Merged.xlsx`) with `MatchWinner` added.
3. Use **`vct_analysis.ipynb`** (or `vct_analysis.py`) to run studies on any two-team matchup.

> **Kaggle users:** upload your data as a Kaggle Dataset, attach it via "+ Add Input," and set the `MERGED_FILE` path in the config cell to `/kaggle/input/<your-dataset-name>/<filename>.xlsx` — find the exact path with:
> ```python
> import os
> for dirname, _, filenames in os.walk('/kaggle/input'):
>     for filename in filenames:
>         print(os.path.join(dirname, filename))
> ```

## Usage

```python
from vct_analysis import load_data, analyze_matchup

df = load_data("VCT2026-Merged.xlsx")
results = analyze_matchup(df, "Eternal Fire", "Fnatic")
```

This single call prints a full report and shows the visualization:

- Veto tendencies (pick rate / ban rate per map) for both teams
- Per-map win rate and **average round differential** for both teams
- Overall win rate and head-to-head record
- Predicted win probability (log5 formula, blended with head-to-head)
- A simulated veto sequence, patch-aware and format-aware (see below)
- A 2×2 chart: pick rate and round differential by map, per team

### Individual functions

Each piece is also available standalone:

| Function | Purpose |
|---|---|
| `veto_tendencies(df, team)` | Pick/ban rate per map for a team |
| `map_win_rates(df, team)` | Win rate + avg round diff per map for a team |
| `overall_win_rate(df, team)` | Team's overall match win rate |
| `head_to_head(df, team_a, team_b)` | Direct record between two teams |
| `log5(win_rate_a, win_rate_b)` | Bill James' win-probability formula |
| `win_probability(df, team_a, team_b)` | Full blended win probability |
| `get_map_pool_for_patch(df, patch)` | Active map pool for a given patch, derived from the data itself |
| `simulate_veto(veto_a, veto_b, team_a, team_b, map_pool, match_format)` | Full veto sequence simulation |
| `plot_matchup(...)` | 2×2 pick-rate / round-diff chart |

### Patch-aware map pool

Since the dataset spans multiple patches with different active map pools, the map pool for a simulated veto is derived **from the data itself** for a given patch, rather than hardcoded:

```python
map_pool = get_map_pool_for_patch(df, "12.05-12.08")
```

### Format-aware veto simulation

Veto order differs by match format:

- **Bo3**: `Ban, Ban, Pick, Pick, Ban, Ban, Decider` (2 bans + 1 pick per team)
- **Bo5**: `Ban, Ban, Pick, Pick, Pick, Pick, Decider` (1 ban + 2 picks per team)

```python
veto_sequence = simulate_veto(veto_a, veto_b, team_a, team_b, map_pool, match_format="bo3")
```

The simulator resolves each ban/pick using whichever team's historical tendency is highest **among maps still remaining** (so a map already banned can never be picked). If the map pool doesn't divide evenly into the expected veto length, it flags `⚠ UNRESOLVED` rather than guessing — a signal to double-check the patch grouping.

## Methodology

All predictions use classical statistics — no ML models:

- **Pick/ban prediction**: raw historical frequency of each team picking/banning each map, sequenced through the real veto order so no map can be double-claimed.
- **Win probability**: the [log5 formula](https://en.wikipedia.org/wiki/Log5) (Bill James, sabermetrics) converts each team's standalone win rate into a head-to-head probability, then blends in the actual head-to-head record — weighted more heavily as more meetings accumulate (capped at 50% weight to avoid overfitting to small samples).

## Known Limitations

- **Team name inconsistencies across source files** — e.g. `"PACIFIC ESPORTS"` vs `"PCIFIC ESPORTS"` (a scraping typo present in only some event files) are treated as two different teams, silently splitting that team's history. Worth normalizing team names during the merge step if this affects a team you're studying.
- **Patch labels aren't always clean** — some are single version numbers (`12`), others are ranges (`12.05-12.08`). These are matched by exact string, so a patch grouping that's too coarse or too fine will affect the derived map pool.
- **Small sample sizes** — teams/maps with only 1–2 historical data points produce noisy win rates and pick/ban rates; treat predictions involving thin data with appropriate skepticism (`maps_played` is included in output specifically to flag this).
- **Coin-toss dependency** — which team gets first veto in a real match isn't knowable in advance; `simulate_veto` requires you to specify an order, so consider running it both ways.

## Roadmap / Ideas

- Bracket-level tournament simulation for full Champs 2026


