#!/usr/bin/env python3
"""
behavioural_analysis_pipeline.py

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
This pipeline runs the full behavioural analysis across all usable Atari games
in game_config.json, producing the three descriptor classes developed in my
MSc thesis and extended here to scale across the full Atari-57 benchmark.

The descriptor classes follow the original per-game thesis notebooks, but the
numbers are not directly comparable with them: this version corrects the data
first (actions, lives, rewards and game boundaries, see "Source preparation")
and adjusts two measures for data size and game length (balanced RAM
clustering, windowed n-gram diversity). Run with --also-unadjusted to get the
original versions of those two measures next to the adjusted ones.
The pipeline is config-driven: one script handles all games, with per-game
variation handled through game_config.json.

-------------------------------------------------------------------------------
Descriptor classes
-------------------------------------------------------------------------------
For each game and each available gameplay source (Human, PPO, DQN):

  1. Performance
       Episode return, reward density per 1000 raw frames, first-reward
       latency, life losses, and how each game ended (natural_terminal,
       game_over_lives_left, truncated_time_limit, incomplete_log_end).
       Unfinished games are left out of the performance summary.
       Only computed for games with status = verified or partial in the config.

  2. Action structure
       Action distribution, entropy (bits and normalised 0-1), switching rate,
       mean run length, FIRE rate, trigram and four-gram diversity, top motifs.
       n-gram diversity is computed in non-overlapping windows of 256
       decisions (--ngram-window) and averaged per game, because the measure
       falls with sequence length; games shorter than one window are left out
       of it. Motifs are reported as counts and as shares.
       Computed for all games regardless of score-byte status.

  3. RAM-state visitation
       Exact RAM-state uniqueness and Jaccard overlap, PCA projection
       (figure only), MiniBatchKMeans clustering (k=15, sensitivity k=10, 20),
       Jensen-Shannon and total-variation distances between sources.
       Balanced fitting: byte selection, scaling and k-means are fitted on the
       same number of rows per source (the smallest source, max 5,000; every
       game of a source equally likely to contribute), then all rows are
       assigned. Repeated 10 times with different samples; distances are
       reported as mean, SD and range. A within-source baseline (alternate
       games of one source, half A vs half B) shows how large a distance is
       between games of the same source. Needs >= 2 games per source.
       Note: exact-state uniqueness (unique share of rows) also depends on
       how many rows a source has.
       Computed for all games regardless of score-byte status.

  --also-unadjusted adds the original methods (one clustering fit on all rows
  pooled; n-gram diversity over whole games), labelled UNADJUSTED, so their
  effect can be seen and results compared with earlier notebooks. They are
  biased by data size and game length; report the adjusted results.

-------------------------------------------------------------------------------
Which games run, and to what depth
-------------------------------------------------------------------------------
  verified     -> full analysis (performance + action structure + RAM)
  partial      -> full analysis, performance flagged as low-confidence
  lives_seeded -> action structure + RAM only
  lives_only   -> action structure + RAM only

A game missing data files is skipped with a logged message rather than
crashing, so the pipeline can be run incrementally as new sessions come in.

-------------------------------------------------------------------------------
FIRE handling (auto-derived, overridable per game)
-------------------------------------------------------------------------------
FIRE is not a meaningful movement decision in most games -- it serves the ball
in Breakout, is absent in Freeway and Asterix, and is a directional modifier
in Pong. The pipeline derives a FIRE handling rule per game automatically:

  none      no FIRE or FIRE-combo actions in the minimal set
            (e.g. freeway, skiing, asterix) -- no adjustment needed
  exclude   FIRE present but no directional FIRE-combos
            (e.g. breakout) -- FIRE windows dropped before descriptors
  collapse  directional FIRE-combos present
            (e.g. pong, space_invaders, seaquest) -- RIGHTFIRE mapped to
            RIGHT, LEFTFIRE to LEFT etc., preserving the movement component

To override for a specific game, add a "fire_handling" field in game_config.json:
    "seaquest": { ..., "fire_handling": "keep" }

Valid values: "none", "exclude", "collapse", "keep".

For games that have FIRE, a separate "fire_rate" descriptor (share of
decisions that include FIRE, measured before the rule is applied) is always
reported, so FIRE use stays visible whichever rule is chosen. In Breakout,
"collapse" simply maps FIRE to NOOP: the agent presses FIRE while the ball is
already in play, where it has no effect.
The rule applied to each game is logged to outputs/_summary/fire_handling.csv.

Current config overrides for the six validation games: breakout = collapse
(FIRE only serves the ball, so FIRE counts as NOOP), space_invaders and
seaquest = keep (shooting is an in-play decision). freeway, asterix and skiing
have no FIRE action (auto: none).

-------------------------------------------------------------------------------
Data layout
-------------------------------------------------------------------------------
  aligned/<game>/*.csv             aligned human session CSV (265 cols)
  agent_test_data/<game>/ppo*.csv  PPO agent log (265 cols)
  agent_test_data/<game>/dqn*.csv  DQN agent log (265 cols)

All CSVs must follow the 265-column schema:
  run_ts, episode, step, action, reward, done, episode_return,
  lives_pre, lives_post, ram_pre_0..127, ram_post_0..127

Human sessions are the aligned output of run_pipeline.py (default frameskip 4,
one row per decision) -- the SAME cadence as the agent logs. They are pooled
as-is; this pipeline does not re-window them.

All files per source are loaded (not only the first). ram_post_* columns are
used when present.

-------------------------------------------------------------------------------
Source preparation (before any descriptor is computed)
-------------------------------------------------------------------------------
Each source is brought to common units; every step is logged per game in
outputs/_summary/run_summary.csv.

  1. Actions -> ALE ids. Logged actions are usually indices into the minimal
     action set; they are mapped through minimal_action_set. Auto-detected;
     override per game with "action_encoding": "index" | "ale"
     (or {"Human": "index", "PPO": "index", ...}).
  2. Lives come from RAM for every source when the config has a lives
     counter ("lives_count" {byte, mask?, offset?} or "lives" mode "byte").
     Games without a counter (Freeway, Skiing) get no lives (life losses NaN).
     Game over always counts the last life, for every source.
  3. Per-life episodes (SB3 EpisodicLifeEnv) and logger-cap splits are merged
     back into full games. Rows whose ram_post already shows the next game
     (env auto-reset) are flagged. Games cut off by the env time limit
     (108,000 frames) are marked truncated_time_limit.
  4. Rewards are recomputed from RAM for every source, so all use one
     definition: score(ram_post) - score(ram_pre) from "score_bytes"
     (transcribed from ALE's game source; optional "mask", "score_wrap" for
     scores that roll over, "reward_clip_0_1" for Freeway as in ALE).
     Auto-reset rows keep the logged reward. Replaced only if RAM and logged
     rewards agree in sign on >= 95% of rows, or if the logged rewards are
     all zero (nothing to check against).
     "terminal_penalty" {byte, format, scale, name}: for games that add a lump
     penalty on the final step (Skiing: 5 s per gate not passed, RAM byte
     107 = gates still to pass), the auto-reset row is rebuilt as a typical
     step + scale * counter, and the counter at the end is reported per game
     (Skiing: missed_gates). "score_is_time": true splits the score into
     run_time_s and penalty_s and blanks reward density and first-reward
     latency, which are meaningless when every step is rewarded.
  5. Leading pre-game rows are dropped for games whose lives config has
     require_nonzero_first.

Optional second pass: with "report_without_stuck": true in a game's config,
descriptors are computed again with "stuck stretches" removed -- the rows
after the last reward or life loss in games cut off by the environment's time
limit. Results go to outputs/<game>/without_stuck/; the master table has a
"variant" column ("all" / "without_stuck").

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
  python behavioural_analysis_pipeline.py
  python behavioural_analysis_pipeline.py --games breakout seaquest freeway
  python behavioural_analysis_pipeline.py --skip-ram
  python behavioural_analysis_pipeline.py --no-figures
  python behavioural_analysis_pipeline.py --save-aligned
  python behavioural_analysis_pipeline.py --also-unadjusted
  python behavioural_analysis_pipeline.py --ngram-window 128
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# sklearn is only needed for the RAM section; import lazily so --skip-ram works
# even in an environment without sklearn.
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


# ============================================================================
# Constants
# ============================================================================
ACTION_NAMES = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
    14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE",
}
FIRE_ACTION = 1
FIRE_COMBO_ACTIONS = {10, 11, 12, 13, 14, 15, 16, 17}

# Maps each FIRE-combination action to its movement component for the collapse rule.
COLLAPSE_TO_MOVEMENT = {
    0: 0, 1: 0,           # NOOP, FIRE           -> NOOP
    2: 2, 10: 2,          # UP, UPFIRE           -> UP
    3: 3, 11: 3,          # RIGHT, RIGHTFIRE     -> RIGHT
    4: 4, 12: 4,          # LEFT, LEFTFIRE       -> LEFT
    5: 5, 13: 5,          # DOWN, DOWNFIRE       -> DOWN
    6: 6, 14: 6,          # UPRIGHT, UPRIGHTFIRE -> UPRIGHT
    7: 7, 15: 7,          # UPLEFT, UPLEFTFIRE   -> UPLEFT
    8: 8, 16: 8,          # DOWNRIGHT, ...       -> DOWNRIGHT
    9: 9, 17: 9,          # DOWNLEFT, ...        -> DOWNLEFT
}

SOURCE_ORDER = ["Human", "PPO", "DQN"]
AGENT_RAW_FRAMES_PER_ROW = 4
HUMAN_RAW_FRAMES_PER_ROW = 4
CAP_THRESHOLD_LOGGED_ROWS = 9990   # episodes capped at ~10,000 logged steps

BASE_COLS = ["run_ts", "episode", "step", "action", "reward", "done",
             "episode_return", "lives_pre", "lives_post"]
RAM_COLS = [f"ram_pre_{i}" for i in range(128)]
RAM_POST_COLS = [f"ram_post_{i}" for i in range(128)]
REQUIRED_COLS = BASE_COLS + RAM_COLS
# ram_post is optional for the descriptors, but needed to recompute rewards
# from RAM (row reward = score(ram_post) - score(ram_pre)).
LOAD_COLS = REQUIRED_COLS + RAM_POST_COLS

# Reward recomputation from RAM is accepted only if its sign agrees with the
# logged reward on at least this share of rows (excluding game-reset rows).
REWARD_SIGN_AGREEMENT_MIN = 0.95

# Environment time limit (raw frames). Gymnasium's ALE v5 default and the cut-off
# seen in the SB3 Breakout logs is 108,000 frames (27,000 decisions at frameskip 4).
# A game this long that did not end in game over was cut off by the limit.
ENV_MAX_FRAMES = 108_000
ENV_LIMIT_TOLERANCE = 0.99

# RAM clustering / PCA parameters
NEAR_CONST_STD_THRESHOLD = 1.0
K_MAIN = 15
K_SENSITIVITY = [10, 20]
PCA_SAMPLE_PER_SOURCE = 5000
RANDOM_STATE = 0
KMEANS_N_INIT = 10
# Balanced fitting: the clustering is fitted on the same number of rows from
# each source (at most this many), with every game of a source equally likely
# to contribute, then all rows are assigned. Repeated with different samples.
RAM_FIT_CAP_PER_SOURCE = 5000
RAM_N_REPEATS = 10

# n-gram diversity is computed in non-overlapping windows of this many
# decisions (it falls with sequence length, so whole games of different
# lengths are not comparable). Games shorter than one window are left out.
NGRAM_WINDOW = 256

PERFORMANCE_STATUSES = {"verified", "partial"}
LOW_CONFIDENCE_STATUSES = {"partial"}


# ============================================================================
# FIRE handling
# ============================================================================
def derive_fire_handling(minimal_action_set):
    """Derive FIRE rule from the minimal action set: none / exclude / collapse."""
    aset = set(int(a) for a in minimal_action_set)
    has_fire = FIRE_ACTION in aset
    has_combo = len(aset & FIRE_COMBO_ACTIONS) > 0
    if has_combo:
        return "collapse"
    if has_fire:
        return "exclude"
    return "none"


def resolve_fire_handling(game_cfg):
    """Return (rule, source) -- config override wins over auto-derived."""
    override = game_cfg.get("fire_handling")
    if override in {"none", "exclude", "collapse", "keep"}:
        return override, "config-override"
    return derive_fire_handling(game_cfg.get("minimal_action_set", [])), "auto"


def report_actions_for_rule(minimal_action_set, rule):
    """Return the action labels descriptors are computed over for this FIRE rule."""
    aset = sorted(int(a) for a in minimal_action_set)
    if rule == "collapse":
        movement = sorted({COLLAPSE_TO_MOVEMENT.get(a, a) for a in aset})
        return movement
    if rule == "exclude":
        return [a for a in aset if a != FIRE_ACTION]
    return aset  # none, keep


def apply_fire_rule(actions, rule):
    """Apply the FIRE rule to an action sequence before computing descriptors."""
    a = np.asarray(actions, dtype=int)
    if rule == "exclude":
        return a[a != FIRE_ACTION]
    if rule == "collapse":
        return np.array([COLLAPSE_TO_MOVEMENT.get(int(x), int(x)) for x in a], dtype=int)
    return a  # none, keep


# ============================================================================
# Loading
# ============================================================================
def find_csvs(root: Path, game: str, patterns):
    """Return sorted CSVs under root/<game>/ matching any of the given glob patterns."""
    gdir = root / game
    if not gdir.is_dir():
        return []
    found = []
    for pat in patterns:
        found.extend(sorted(gdir.glob(pat)))
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def validate_columns(path: Path):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        raise ValueError(
            f"{path.name} missing required columns: {missing[:6]}"
            f"{'...' if len(missing) > 6 else ''}"
        )


def _read_log(path: Path):
    """Read a 265-col log; ram_post columns are loaded when present."""
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in LOAD_COLS if c in cols]
    return pd.read_csv(path, usecols=use)


def load_source(files, source: str, game: str):
    """Load ALL files for one source. Episodes are renumbered across files so
    that (file, episode) pairs stay distinct. orig_episode keeps the logged id."""
    parts, ep_counter = [], 0
    raw_mult = HUMAN_RAW_FRAMES_PER_ROW if source == "Human" else AGENT_RAW_FRAMES_PER_ROW
    for f in files:
        df = _read_log(f)
        for orig_ep, sub in df.groupby("episode", sort=True):
            ep_counter += 1
            sub = sub.sort_values("step").copy()
            meta = pd.DataFrame({
                "source": source, "game": game, "orig_file": f.name,
                "orig_episode": int(orig_ep),
                "global_episode": ep_counter,
                "raw_frames_per_row": raw_mult,
            }, index=sub.index)
            parts.append(pd.concat([sub, meta], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else None


# ============================================================================
# Source preparation: make every source use the same encoding and units
# ============================================================================
def resolve_action_encoding(actions, minimal_action_set, override=None):
    """Decide whether logged actions are minimal-set indices or ALE IDs.

    Returns (encoding, how). encoding is "index" or "ale".
    Gym/SB3 agents and the aligned human files log the index into the minimal
    action set; the descriptors below are defined on ALE IDs (0-17).
    """
    if override in {"index", "ale"}:
        return override, "config-override"
    ms = [int(a) for a in minimal_action_set]
    if ms == list(range(len(ms))):
        return "ale", "identical (index == ALE id for this game)"
    vals = {int(a) for a in np.unique(actions)}
    fits_index = vals <= set(range(len(ms)))
    fits_ale = vals <= set(ms)
    if fits_index and not fits_ale:
        return "index", "auto"
    if fits_ale and not fits_index:
        return "ale", "auto"
    if fits_index and fits_ale:
        return "index", "auto-AMBIGUOUS (assumed index; set action_encoding in config to be sure)"
    raise ValueError(f"actions {sorted(vals)} fit neither minimal-set indices nor ALE ids {ms}")


def actions_to_ale(actions, minimal_action_set):
    ms = np.asarray([int(a) for a in minimal_action_set])
    return ms[np.asarray(actions, dtype=int)]


def _bcd(x):
    x = np.asarray(x, dtype=int)
    return (x >> 4) * 10 + (x & 15)


def score_from_ram(ram: np.ndarray, score_bytes):
    """Score per row from a (n, 128) RAM array using config score_bytes."""
    s = np.zeros(len(ram), dtype=float)
    for b in score_bytes:
        x = ram[:, int(b["byte"])].astype(int)
        if "mask" in b:
            x = x & int(b["mask"])
        v = _bcd(x) if b.get("format", "bcd") == "bcd" else x
        s += float(b.get("weight", 1)) * float(b.get("scale", 1)) * v
    return s


def lives_from_ram(ram: np.ndarray, game_cfg):
    """Lives count per row from RAM, or None if the config has no count byte.

    Uses `lives_count` {byte, mask?, offset?} if present, otherwise
    `lives` with mode == "byte". terminal_value modes are game-over detectors,
    not counts, so they return None.
    """
    lc = game_cfg.get("lives_count")
    if lc is None and game_cfg.get("lives", {}).get("mode") == "byte":
        lc = game_cfg["lives"]
    if not lc:
        return None
    x = ram[:, int(lc["byte"])].astype(int)
    if "mask" in lc:
        x = x & int(lc["mask"])
    return x + int(lc.get("offset", 0))


def _lives_known(df):
    return bool((df["lives_pre"] >= 0).any())


def merge_life_episodes(df):
    """Join logged episodes that belong to the same game.

    An episode continues the previous game (same file) if either:
      (a) life split (SB3 EpisodicLifeEnv): the previous episode ended with a
          life loss and this one starts with the lives that were left; or
      (b) logger-cap split: the previous episode hit the per-episode row cap
          (CAP_THRESHOLD_LOGGED_ROWS) without a reset, and this one starts with
          the same lives (or lives are unknown).
    Returns (df, n_games_before, n_games_after).
    """
    n_before = df["global_episode"].nunique()
    known = _lives_known(df)
    game_of_ep, game, prev_last, prev_len = {}, 0, None, 0
    for ep, sub in df.groupby("global_episode", sort=True):
        first = sub.iloc[0]
        continues = False
        if prev_last is not None and prev_last["orig_file"] == first["orig_file"]:
            life_split = known and (
                prev_last["lives_post"] < prev_last["lives_pre"]
                and first["lives_pre"] == prev_last["lives_post"])
            cap_split = prev_len >= CAP_THRESHOLD_LOGGED_ROWS and (
                not known or (prev_last["lives_post"] == prev_last["lives_pre"]
                              and first["lives_pre"] == prev_last["lives_pre"]))
            continues = life_split or cap_split
        if not continues:
            game += 1
        game_of_ep[ep] = game
        prev_last, prev_len = sub.iloc[-1], len(sub)
    df = df.copy()
    df["global_episode"] = df["global_episode"].map(game_of_ep)
    df = df.sort_values(["global_episode", "orig_episode", "step"]).reset_index(drop=True)
    df["step"] = df.groupby("global_episode").cumcount()
    last = df.groupby("global_episode")["step"].transform("max") == df["step"]
    df["done"] = np.where(last, df["done"], 0)
    return df, n_before, df["global_episode"].nunique()


def _game_over_seen(df, game_over_cfg):
    """Per row: does RAM show game over (in ram_pre or ram_post)?
    Config "game_over": {"byte": b, "nonzero": true} or {"byte": b, "value": v}.
    Returns None when the game has no such detector in the config."""
    if not game_over_cfg:
        return None
    b = int(game_over_cfg["byte"])
    seen = np.zeros(len(df), bool)
    for col in (f"ram_pre_{b}", f"ram_post_{b}"):
        if col in df:
            x = df[col].to_numpy(dtype=int)
            seen |= (x != 0) if game_over_cfg.get("nonzero") else (x == int(game_over_cfg["value"]))
    return pd.Series(seen, index=df.index)


def mark_reset_rows(df, auto_reset=False, max_frames=ENV_MAX_FRAMES, game_over_cfg=None):
    """Classify how each game ended and flag rows whose ram_post / lives_post
    already show the NEXT game (env auto-reset; RAM reward there is unusable).

    auto_reset=True (agent logs): the last row of every finished game is an
    auto-reset row. For other sources only a lives jump marks it.

    Per game, from first match:
      incomplete_log_end   log or recording stopped before game over: last
                           game in a file ending in a life loss with lives
                           left, ending on the logger's row cap, or (with a
                           "game_over" detector in the config) a recording in
                           which game over never appears -- excluded from
                           the performance summary
      truncated_time_limit game as long as the env limit (108,000 frames)
      game_over_lives_left ended before the last life (e.g. Space Invaders
                           invaders reaching the ground)
      natural_terminal     ended on the last life
    Game over on the last life always counts that life as lost.
    """
    df = df.copy()
    last = df.groupby("global_episode")["step"].transform("max") == df["step"]
    known = _lives_known(df)
    frames = df.groupby("global_episode")["raw_frames_per_row"].transform("sum")
    near_limit = frames >= ENV_LIMIT_TOLERANCE * max_frames
    last_game_in_file = df.groupby("orig_file")["global_episode"].transform("max") == df["global_episode"]

    incomplete = last & last_game_in_file & known & (df["lives_post"] < df["lives_pre"]) & (df["lives_post"] >= 1)
    # a file that ends on the logger's row cap also stopped mid-game
    seg_rows = df.groupby(["global_episode", "orig_episode"])["step"].transform("size") \
        if "orig_episode" in df else df.groupby("global_episode")["step"].transform("size")
    incomplete |= last & last_game_in_file & (seg_rows >= CAP_THRESHOLD_LOGGED_ROWS) & ~near_limit
    # files without an auto-reset row (human recordings): if the config has a
    # game-over detector and it never fires in a game, the recording stopped
    # before game over (e.g. Seaquest stopped while the last sub was in play)
    go = _game_over_seen(df, game_over_cfg)
    if go is not None and not auto_reset:
        game_ended = go.groupby(df["global_episode"]).transform("any")
        incomplete |= last & ~game_ended
    if auto_reset:
        reset = last & (df["done"] > 0) & ~incomplete
    else:
        reset = last & known & (df["lives_post"] > df["lives_pre"]) & (df["lives_pre"] >= 0)

    on_last_life = df["lives_pre"] <= 1
    if known:
        df.loc[reset & on_last_life, "lives_post"] = 0
        df.loc[reset & ~on_last_life, "lives_post"] = df.loc[reset & ~on_last_life, "lives_pre"]
        # files without an auto-reset row: ALE ends some games (Asterix,
        # Seaquest) just before the lives counter drops -- count that life
        final_life = last & ~reset & ~incomplete & (df["done"] > 0) & (df["lives_pre"] >= 0) \
            & on_last_life & (df["lives_post"] > 0)
        df.loc[final_life, "lives_post"] = 0
    df["reset_row"] = reset

    def games(mask):
        return set(df.loc[mask, "global_episode"])
    g_incomplete = games(incomplete)
    if known:   # lives left at a reset this long -> cut off by the limit
        g_limit = games(reset & ~on_last_life & near_limit) - g_incomplete
    else:       # no lives information: game length alone
        g_limit = games(last & near_limit) - g_incomplete
    g_lives_left = games(last & (df["done"] > 0) & known & ~on_last_life & ~near_limit
                         & (df["lives_post"] > 0)) - g_incomplete - g_limit
    df["incomplete_log_end"] = df["global_episode"].isin(g_incomplete)
    df["truncated_time_limit"] = df["global_episode"].isin(g_limit)
    df["game_over_lives_left"] = df["global_episode"].isin(g_lives_left)
    return df


def recompute_rewards(df, game_cfg):
    """Compute every source's rewards from RAM, so all sources use one
    definition: reward = score(ram_post) - score(ram_pre), from config
    score_bytes (transcribed from ALE's game source).

    - Scores that roll over (Space Invaders at 10,000) use "score_wrap".
    - Rows whose ram_post already shows the next game (env auto-reset) keep
      the logged reward. They are found from lives (mark_reset_rows) and,
      for games without lives, as the last row of a logged episode whose RAM
      reward contradicts the logged one.
    - Safety check: if RAM and logged rewards agree in sign on < 95% of the
      other rows, the logged rewards are kept and the source is flagged.
    Returns (df, note).
    """
    sb = game_cfg.get("score_bytes") or []
    if not sb:
        return df, "logged (no score_bytes in config)"
    if not set(RAM_POST_COLS).issubset(df.columns):
        return df, "logged (no ram_post columns)"
    pre = score_from_ram(df[RAM_COLS].to_numpy(), sb)
    post = score_from_ram(df[RAM_POST_COLS].to_numpy(), sb)
    ram_r = post - pre
    wrap = game_cfg.get("score_wrap")
    if wrap:
        ram_r = np.where(ram_r < 0, ram_r + float(wrap), ram_r)
    if game_cfg.get("reward_clip_0_1"):          # ALE clips Freeway to [0, 1]
        ram_r = np.clip(ram_r, 0, 1)
    logged = df["reward"].to_numpy(dtype=float)
    clipped_log = set(np.unique(logged)) <= {-1.0, 0.0, 1.0}

    reset = df["reset_row"].to_numpy().copy() if "reset_row" in df else np.zeros(len(df), bool)
    # last row of each logged episode that ended in done: if RAM contradicts
    # the logged reward there, ram_post is the reset state of the next game
    seg_last = (df.groupby(["global_episode", "orig_episode"])["step"].transform("max") == df["step"]).to_numpy().copy() \
        if "orig_episode" in df else np.zeros(len(df), bool)
    seg_last &= df["done"].to_numpy() > 0
    contradicts = (np.sign(ram_r) != np.sign(logged)) if clipped_log else ~np.isclose(ram_r, logged)
    reset |= seg_last & contradicts
    df = df.copy()
    df["reset_row"] = reset

    ok = ~reset
    agree = float(np.mean(np.sign(ram_r[ok]) == np.sign(logged[ok]))) if ok.any() else np.nan
    exact = float(np.mean(np.isclose(ram_r[ok], logged[ok]))) if ok.any() else np.nan
    logged_empty = not np.any(logged != 0) and np.any(ram_r[ok] != 0)
    if logged_empty:
        # nothing to check against (e.g. an aligned file made without score
        # bytes): use RAM, which was validated against ALE's game code
        df["reward"] = np.where(reset, logged, ram_r)
        note = "RAM (logged rewards were all zero)"
        df, pen_note = _rebuild_terminal_penalty(df, ram_r, reset, game_cfg)
        return df, note + pen_note
    if not agree >= REWARD_SIGN_AGREEMENT_MIN:
        tag = "CLIPPED, " if clipped_log and np.abs(ram_r[ok]).max(initial=0) > 1 else ""
        return df, f"logged, {tag}NOT REPLACED (RAM sign agreement only {agree:.1%})"
    df["reward"] = np.where(reset, logged, ram_r)
    df, pen_note = _rebuild_terminal_penalty(df, ram_r, reset, game_cfg)
    n_disagree = int(np.sum(ok & (np.sign(ram_r) != np.sign(logged))))
    n_reset_clipped = int(np.sum(reset & clipped_log & (np.abs(ram_r) > 1)))
    if np.isclose(exact, 1.0):
        note = "RAM (identical to logged)"
    elif clipped_log and np.abs(ram_r[ok]).max(initial=0) > 1:
        note = f"RAM (logged was clipped; sign agreement {agree:.1%})"
    else:
        note = f"RAM (logged differed on {1 - exact:.1%} of rows; sign agreement {agree:.1%})"
    if reset.any():
        note += f"; {int(reset.sum())} reset row(s) keep logged reward"
        if n_reset_clipped:
            note += " (clipped)"
    if n_disagree:
        # even a handful of these usually means a missing/wrong score byte
        note += f"; CHECK SCORE BYTES: {n_disagree} row(s) disagree in sign with logged reward"
    if pen_note:
        note = note.replace(" (clipped)", "") + pen_note
    return df, note


def _rebuild_terminal_penalty(df, ram_r, reset, game_cfg):
    """Games whose score gets a lump penalty on the final step (Skiing: 5 s per
    gate not passed) lose it on auto-reset rows, where only the logged
    (often clipped) reward survives. Rebuild that row from ram_pre:
    reward = typical step reward of the game + scale * counter value.
    Config: "terminal_penalty": {"byte", "format", "scale", "name"}."""
    tp = game_cfg.get("terminal_penalty")
    if not tp or not reset.any():
        return df, ""
    x = df[f"ram_pre_{int(tp['byte'])}"].to_numpy(dtype=int)
    val = _bcd(x) if tp.get("format", "bcd") == "bcd" else x
    step = pd.Series(np.where(reset, np.nan, ram_r)).groupby(df["global_episode"].to_numpy()).transform("median").to_numpy()
    rebuilt = np.where(reset, np.nan_to_num(step) + float(tp["scale"]) * val, df["reward"].to_numpy())
    df = df.copy()
    df["reward"] = rebuilt
    return df, f"; final-step {tp.get('name', 'penalty')} penalty rebuilt from RAM byte {tp['byte']} on {int(reset.sum())} reset row(s)"


def trim_pregame_rows(df, game_cfg):
    """Drop leading rows before the game starts, for games whose config marks
    lives byte == 0 as 'not started yet' (require_nonzero_first)."""
    if not game_cfg.get("lives", {}).get("require_nonzero_first") or not _lives_known(df):
        return df, 0
    keep = []
    for _, g in df.groupby("global_episode", sort=True):
        started = (g["lives_pre"] > 0).cummax()
        keep.append(g[started])
    out = pd.concat(keep).reset_index(drop=True)
    out["step"] = out.groupby("global_episode").cumcount()
    return out, len(df) - len(out)


def remove_stuck_stretches(df):
    """Cut the no-progress tail of games that did not end naturally.

    A game is affected only if it was cut off by the environment's time limit
    (truncated_time_limit). Its rows after the last 'event' -- the last
    non-zero reward or last life loss -- are dropped. Naturally ended games
    are left whole, so the human data is normally unaffected. Games with no
    event at all are kept whole (nothing to anchor the cut on).
    Returns (df, rows_removed, games_affected).
    """
    if "truncated_time_limit" not in df:
        return df, 0, 0
    keep, removed, affected = [], 0, 0
    for _, g in df.groupby("global_episode", sort=True):
        g = g.sort_values("step")
        if not bool(g["truncated_time_limit"].iloc[0]):
            keep.append(g)
            continue
        event = (g["reward"].to_numpy() != 0) | (g["lives_post"].to_numpy() < g["lives_pre"].to_numpy())
        if not event.any():
            keep.append(g)
            continue
        last = int(np.flatnonzero(event).max())
        keep.append(g.iloc[: last + 1])
        removed += len(g) - (last + 1)
        affected += 1
    return pd.concat(keep, ignore_index=True), removed, affected


def prepare_source(df, source, game_cfg):
    """Bring one source to common units. Returns (df, notes dict)."""
    notes = {}
    minimal = game_cfg.get("minimal_action_set", list(range(18)))

    # 1. actions -> ALE ids
    enc_cfg = game_cfg.get("action_encoding")
    override = enc_cfg.get(source) if isinstance(enc_cfg, dict) else enc_cfg
    enc, how = resolve_action_encoding(df["action"].to_numpy(), minimal, override)
    if enc == "index":
        df = df.copy()
        df["action"] = actions_to_ale(df["action"].to_numpy(), minimal)
    notes["action_encoding"] = f"{enc} ({how})"

    # 2. lives: from RAM for every source when the game has a lives counter
    #    (one definition for all sources); games without lives get -1 (unknown)
    ram_pre = df[RAM_COLS].to_numpy()
    lv_pre = lives_from_ram(ram_pre, game_cfg)
    df = df.copy()
    if lv_pre is not None and set(RAM_POST_COLS).issubset(df.columns):
        logged_known = _lives_known(df)
        m = float(np.mean(lv_pre == df["lives_pre"].to_numpy())) if logged_known else None
        df["lives_pre"] = lv_pre
        df["lives_post"] = lives_from_ram(df[RAM_POST_COLS].to_numpy(), game_cfg)
        notes["lives"] = "from RAM" + (f" (logged lives matched {m:.1%})" if m is not None else " (not logged)")
    elif lv_pre is None:
        df["lives_pre"] = -1
        df["lives_post"] = -1
        notes["lives"] = "none (game has no lives counter)"
    else:
        notes["lives"] = "logged (no ram_post to read lives from)"

    # 3. merge per-life episodes into games, flag auto-reset rows
    df, n_before, n_after = merge_life_episodes(df)
    notes["episodes"] = f"{n_before} logged -> {n_after} games" if n_before != n_after else f"{n_after} games"
    df = mark_reset_rows(df, auto_reset=(source != "Human"), game_over_cfg=game_cfg.get("game_over"))
    extra = []
    for col, label in (("truncated_time_limit", "cut off by env time limit"),
                       ("game_over_lives_left", "ended with lives left"),
                       ("incomplete_log_end", "unfinished (log/recording stopped before game over), left out of performance summary")):
        n = df.loc[df[col], "global_episode"].nunique()
        if n:
            extra.append(f"{n} {label}")
    if extra:
        notes["episodes"] += " (" + "; ".join(extra) + ")"

    # 4. rewards in real points
    df, notes["reward"] = recompute_rewards(df, game_cfg)

    # 5. drop pre-game rows (human recordings start before the game does)
    df, n_trim = trim_pregame_rows(df, game_cfg)
    if n_trim:
        notes["trimmed_pregame_rows"] = n_trim

    # 6. episode_return consistent with (possibly recomputed) rewards
    df["episode_return"] = df.groupby("global_episode")["reward"].cumsum()
    return df, notes


# ============================================================================
# 1. Performance descriptors
# ============================================================================
def episode_performance(raw_all, game, game_cfg=None):
    rows = []
    for (source, ep), g in raw_all.groupby(["source", "global_episode"], sort=True):
        g = g.sort_values("step")
        raw_mult = int(g["raw_frames_per_row"].iloc[0])
        logged_rows = len(g)
        raw_frames = logged_rows * raw_mult

        rewards = g["reward"].to_numpy()
        nz = np.flatnonzero(rewards != 0)
        first_reward_latency_raw = (int(nz[0]) + 1) * raw_mult if len(nz) else np.nan

        if {"lives_pre", "lives_post"}.issubset(g.columns) and (g["lives_pre"] >= 0).any():
            life_losses = int((g["lives_post"].to_numpy() < g["lives_pre"].to_numpy()).sum())
        else:
            life_losses = np.nan   # lives unknown -- not the same as zero losses

        # the logger cap applies to one logged episode; a game counts as capped
        # only if its LAST logged segment hit the cap (earlier ones were merged)
        seg_len = (g["orig_episode"] == g["orig_episode"].iloc[-1]).sum() if "orig_episode" in g else logged_rows
        capped = seg_len >= CAP_THRESHOLD_LOGGED_ROWS
        time_limited = bool(g["truncated_time_limit"].iloc[0]) if "truncated_time_limit" in g else False
        incomplete = bool(g["incomplete_log_end"].iloc[0]) if "incomplete_log_end" in g else False
        lives_left = bool(g["game_over_lives_left"].iloc[0]) if "game_over_lives_left" in g else False
        final_return = float(g["episode_return"].iloc[-1])
        extra = {}
        tp = (game_cfg or {}).get("terminal_penalty")
        if tp:
            x = int(g[f"ram_pre_{int(tp['byte'])}"].iloc[-1])
            n_left = int(_bcd(x)) if tp.get("format", "bcd") == "bcd" else x
            extra[tp.get("name", "penalty_count")] = n_left
            if (game_cfg or {}).get("score_is_time"):
                # score = -(time + penalty) in time units; split them
                tu = float(game_cfg.get("time_units_per_second", 100))
                extra["run_time_s"] = (float(tp["scale"]) * n_left - final_return) / tu
                extra["penalty_s"] = -float(tp["scale"]) * n_left / tu
        if (game_cfg or {}).get("score_is_time"):
            # reward comes every step, so these two say nothing about play
            first_reward_latency_raw = np.nan
        rows.append({
            "game": game, "source": source, "episode": ep,
            "orig_file": g["orig_file"].iloc[0],
            "logged_rows": logged_rows, "raw_frames": raw_frames,
            "final_return": final_return,
            "reward_sum": float(g["reward"].sum()),
            "reward_density_per_1000_raw": (np.nan if (game_cfg or {}).get("score_is_time")
                                            else final_return / raw_frames * 1000 if raw_frames else np.nan),
            "first_reward_latency_raw": first_reward_latency_raw,
            "life_losses": life_losses,
            "done_last": int(g["done"].iloc[-1]),
            "end_type": ("incomplete_log_end" if incomplete
                         else "truncated_time_limit" if time_limited
                         else "game_over_lives_left" if lives_left
                         else "truncated_at_cap" if capped else "natural_terminal"),
            **extra,
        })
    return pd.DataFrame(rows)


def summarize_numeric(df, metrics, group_col="source"):
    rows = []
    for group, g in df.groupby(group_col):
        row = {group_col: group, "n": len(g)}
        for m in metrics:
            x = g[m].dropna()
            row[f"{m}_mean"] = x.mean()
            row[f"{m}_sd"] = x.std(ddof=1)
            row[f"{m}_median"] = x.median()
            row[f"{m}_min"] = x.min()
            row[f"{m}_max"] = x.max()
        rows.append(row)
    present = [s for s in SOURCE_ORDER if s in df[group_col].unique()]
    return pd.DataFrame(rows).set_index(group_col).reindex(present).reset_index()


# ============================================================================
# 2. Temporal alignment
# ============================================================================
def modal_action_earliest_tie(actions):
    actions = list(map(int, actions))
    counts = Counter(actions)
    mx = max(counts.values())
    tied = {a for a, c in counts.items() if c == mx}
    return next(a for a in actions if a in tied)


def aggregate_human_to_windows(human_raw, game):
    parts = []
    for ep, g in human_raw.groupby("global_episode", sort=True):
        g = g.sort_values("step").reset_index(drop=True)
        win = np.arange(len(g)) // 4
        ram_arr = g[RAM_COLS].to_numpy(dtype=np.uint8)
        meta_rows, ram_rows = [], []
        for w in np.unique(win):
            idx = np.flatnonzero(win == w)
            sub = g.iloc[idx]
            meta_rows.append({
                "game": game, "source": "Human", "global_episode": ep, "step": int(w),
                "action": modal_action_earliest_tie(sub["action"].tolist()),
                "reward": float(sub["reward"].sum()),
                "done": int(sub["done"].max()),
                "episode_return": float(sub["episode_return"].iloc[-1]),
                "lives_pre": int(sub["lives_pre"].iloc[0]),
                "lives_post": int(sub["lives_post"].iloc[-1]),
                "orig_file": sub["orig_file"].iloc[0],
                "raw_frames_per_row": 4,
            })
            ram_rows.append(ram_arr[idx[0]])
        meta = pd.DataFrame(meta_rows)
        ram_df = pd.DataFrame(np.vstack(ram_rows), columns=RAM_COLS)
        parts.append(pd.concat([meta.reset_index(drop=True), ram_df], axis=1))
    return pd.concat(parts, ignore_index=True)


def build_aligned(human_raw, agent_raws, game):
    frames = []
    if human_raw is not None:
        # Human files in aligned/<game>/ are already frameskip-4 (one row = one
        # decision = 4 raw frames), the same cadence as the agent logs, so they
        # are pooled as-is -- no second windowing. aggregate_human_to_windows()
        # is retained for optional raw 60 Hz input but is not used in this flow.
        h = human_raw.copy()
        h["raw_frames_per_row"] = AGENT_RAW_FRAMES_PER_ROW
        frames.append(h)
    for src, df in agent_raws.items():
        a = df.copy()
        a["raw_frames_per_row"] = 4
        keep = ["game", "source", "global_episode", "step", "action", "reward",
                "done", "episode_return", "lives_pre", "lives_post",
                "orig_file", "raw_frames_per_row"] + RAM_COLS
        frames.append(a[keep])
    cols = ["game", "source", "global_episode", "step", "action", "reward",
            "done", "episode_return", "lives_pre", "lives_post",
            "orig_file", "raw_frames_per_row"] + RAM_COLS
    frames = [f[cols] for f in frames]
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# 3. Action-structure descriptors
# ============================================================================
def action_entropy_bits(actions, report_actions):
    actions = np.asarray(actions, dtype=int)
    if len(actions) == 0:
        return np.nan
    counts = pd.Series(actions).value_counts().reindex(report_actions, fill_value=0).to_numpy()
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def switching_rate(actions):
    a = np.asarray(actions, dtype=int)
    if len(a) < 2:
        return np.nan
    return float(np.mean(a[1:] != a[:-1]))


def mean_run_length(actions):
    a = list(map(int, actions))
    if not a:
        return np.nan
    runs, cur, length = [], a[0], 1
    for x in a[1:]:
        if x == cur:
            length += 1
        else:
            runs.append(length); cur = x; length = 1
    runs.append(length)
    return float(np.mean(runs))


def ngram_diversity(actions, n):
    a = list(map(int, actions))
    if len(a) < n:
        return np.nan
    grams = [tuple(a[i:i+n]) for i in range(len(a) - n + 1)]
    return len(set(grams)) / len(grams)


def windowed_ngram_diversity(actions, n, window=NGRAM_WINDOW):
    """Mean n-gram diversity over non-overlapping windows of `window` decisions.
    NaN if the sequence is shorter than one window."""
    a = list(map(int, actions))
    if len(a) < window:
        return np.nan
    return float(np.mean([ngram_diversity(a[i:i + window], n)
                          for i in range(0, len(a) - window + 1, window)]))


def action_structure(aligned, game, fire_rule, report_actions, has_fire=False,
                     window=NGRAM_WINDOW, also_unadjusted=False):
    rows = []
    for (source, ep), g in aligned.groupby(["source", "global_episode"], sort=True):
        g = g.sort_values("step")
        actions_full = g["action"].astype(int).to_numpy()
        actions = apply_fire_rule(actions_full, fire_rule)
        n_fire_excluded = int(np.sum(actions_full == FIRE_ACTION)) if fire_rule == "exclude" else 0
        row = {
            "game": game, "source": source, "episode": ep,
            "n_windows_full": len(actions_full),
            "n_fire_excluded": n_fire_excluded,
            "n_windows_analyzed": len(actions),
            "action_entropy_bits": action_entropy_bits(actions, report_actions),
            # entropy / log2(number of analysed actions): 0 = always the same
            # action, 1 = all actions equally often. Comparable across games.
            "action_entropy_norm": (action_entropy_bits(actions, report_actions) / np.log2(len(report_actions))
                                    if len(report_actions) > 1 else np.nan),
            "switching_rate": switching_rate(actions),
            "mean_run_length": mean_run_length(actions),
            # in fixed windows (see NGRAM_WINDOW); NaN for games shorter than one
            "trigram_diversity": windowed_ngram_diversity(actions, 3, window),
            "fourgram_diversity": windowed_ngram_diversity(actions, 4, window),
            "n_windows_ngram": len(actions) // window,
            # share of decisions that include FIRE (FIRE or any FIRE+direction),
            # on the actions BEFORE the FIRE rule -- so shooting/serving stays
            # visible even when the rule collapses or excludes FIRE.
            # NaN for games without a FIRE action.
            "fire_rate": (float(np.mean(np.isin(actions_full, list(FIRE_COMBO_ACTIONS | {FIRE_ACTION}))))
                          if has_fire and len(actions_full) else np.nan),
        }
        if also_unadjusted:   # original whole-game measure, depends on game length
            row["trigram_diversity_UNADJUSTED_whole_game"] = ngram_diversity(actions, 3)
            row["fourgram_diversity_UNADJUSTED_whole_game"] = ngram_diversity(actions, 4)
        aa = np.asarray(actions, dtype=int)
        denom = len(aa)
        for lab in report_actions:
            name = ACTION_NAMES.get(lab, str(lab))
            row[f"prop_{name}"] = float(np.mean(aa == lab)) if denom else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def motif_table(aligned, game, fire_rule, k=10):
    rows = []
    for source, g in aligned.groupby("source", sort=True):
        # count n-grams within each episode so no motif spans two episodes
        counters = {3: Counter(), 4: Counter()}
        for _, ep in g.groupby("global_episode", sort=True):
            seq = apply_fire_rule(ep.sort_values("step")["action"].astype(int).to_numpy(), fire_rule)
            for n in (3, 4):
                counters[n].update(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
        for n in (3, 4):
            total = sum(counters[n].values())
            for motif, count in counters[n].most_common(k):
                rows.append({
                    "game": game, "source": source, "n": n,
                    "motif": "-".join(ACTION_NAMES.get(a, str(a)) for a in motif),
                    "count": count,
                    "share": count / total if total else np.nan,   # comparable across sources
                })
    return pd.DataFrame(rows)


# ============================================================================
# 4. RAM-state visitation
# ============================================================================
def js_divergence(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def total_variation_distance(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p = p / p.sum(); q = q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def ram_exact_uniqueness(aligned, game):
    rows, state_sets = [], {}
    for source, g in aligned.groupby("source", sort=True):
        arr = g[RAM_COLS].to_numpy(dtype=np.uint8)
        state_set = set(map(bytes, arr))
        state_sets[source] = state_set
        rows.append({
            "game": game, "source": source, "aligned_rows": len(arr),
            "unique_exact_ram_states": len(state_set),
            "unique_share_of_rows": len(state_set) / len(arr) if len(arr) else np.nan,
            "jaccard_overlap": np.nan,
        })
    present = [s for s in SOURCE_ORDER if s in state_sets]
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            inter = len(state_sets[a] & state_sets[b])
            union = len(state_sets[a] | state_sets[b])
            rows.append({
                "game": game, "source": f"{a} n {b}", "aligned_rows": np.nan,
                "unique_exact_ram_states": inter, "unique_share_of_rows": np.nan,
                "jaccard_overlap": inter / union if union else np.nan,
            })
    return pd.DataFrame(rows)


def _balanced_fit_index(aligned, rng, n_per_source):
    """Row positions for fitting: n rows per source, each game of a source
    equally likely to contribute (row weight = 1 / game length)."""
    idx = []
    for _, sub in aligned.groupby("source", sort=True):
        w = 1.0 / sub.groupby("global_episode")["step"].transform("size").to_numpy(dtype=float)
        n = min(n_per_source, len(sub))
        idx.append(rng.choice(sub.index.to_numpy(), size=n, replace=False, p=w / w.sum()))
    return np.concatenate(idx)


def _fit_and_assign(ram_full, fit_idx, k, seed):
    """Select informative bytes, scale and fit k-means on fit_idx rows, then
    assign every row. Returns (labels or None, keep_mask, byte_std)."""
    std = ram_full[fit_idx].std(axis=0)
    keep = std > NEAR_CONST_STD_THRESHOLD
    if keep.sum() < 2:
        return None, keep, std
    scaler = StandardScaler().fit(ram_full[fit_idx][:, keep])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=KMEANS_N_INIT)
    km.fit(scaler.transform(ram_full[fit_idx][:, keep]))
    return km.predict(scaler.transform(ram_full[:, keep])), keep, std


def _visit_shares(labels, mask, k):
    """Time-weighted cluster visitation share for the rows in mask."""
    c = np.bincount(labels[mask], minlength=k).astype(float)
    return c / c.sum() if c.sum() else c


def _distance_rows(game, k, labels, aligned, seed):
    """Between-source distances plus a within-source baseline (alternate games
    of one source split into half A and half B; needs >= 2 games)."""
    src = aligned["source"].to_numpy()
    present = [s for s in SOURCE_ORDER if s in set(src)]
    sh = {s: _visit_shares(labels, src == s, k) for s in present}
    rows = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            rows.append({"game": game, "k": k, "seed": seed, "comparison": f"{a} vs {b}",
                         "kind": "between sources",
                         "js_time": js_divergence(sh[a], sh[b]),
                         "tv_time": total_variation_distance(sh[a], sh[b])})
    ep = aligned["global_episode"].to_numpy()
    for s in present:
        games = sorted(set(ep[src == s]))
        if len(games) < 2:
            rows.append({"game": game, "k": k, "seed": seed, "comparison": f"{s} half A vs {s} half B",
                         "kind": "within-source baseline (needs >= 2 games)",
                         "js_time": np.nan, "tv_time": np.nan})
            continue
        half_a = (src == s) & np.isin(ep, games[0::2])
        half_b = (src == s) & np.isin(ep, games[1::2])
        pa, pb = _visit_shares(labels, half_a, k), _visit_shares(labels, half_b, k)
        rows.append({"game": game, "k": k, "seed": seed, "comparison": f"{s} half A vs {s} half B",
                     "kind": "within-source baseline",
                     "js_time": js_divergence(pa, pb), "tv_time": total_variation_distance(pa, pb)})
    return rows


def _summarise_distances(per_seed):
    g = per_seed.groupby(["game", "k", "kind", "comparison"], sort=False)
    out = g.agg(js_mean=("js_time", "mean"), js_sd=("js_time", "std"),
                js_min=("js_time", "min"), js_max=("js_time", "max"),
                tv_mean=("tv_time", "mean"), tv_sd=("tv_time", "std"),
                n_repeats=("seed", "nunique")).reset_index()
    return out


def ram_cluster_analysis(aligned, game, out_dirs, make_figs=True, also_unadjusted=False):
    """RAM-state visitation with balanced fitting.

    For each of RAM_N_REPEATS samples: an equal number of rows per source
    (= the smallest source, capped at RAM_FIT_CAP_PER_SOURCE; every game of a
    source equally likely to contribute) is used to select informative bytes,
    scale them and fit k-means; then every row is assigned to a cluster.
    Visitation shares use all rows. Distances (JS, TV) are reported as mean,
    SD and range over the repeats, for k = 15 and the sensitivity k's, with a
    within-source baseline. PCA is for the figure only.
    also_unadjusted: also report the original method (one fit on all rows
    pooled), labelled 'unadjusted'.
    """
    if not _HAVE_SKLEARN:
        return {"error": "sklearn not available; RAM clustering skipped"}
    aligned = aligned.reset_index(drop=True)
    ram_full = aligned[RAM_COLS].to_numpy(dtype=np.float64)
    n_per_source = min(RAM_FIT_CAP_PER_SOURCE, int(aligned["source"].value_counts().min()))
    results = {}

    per_seed, shares_k15_seed0, fit0 = [], None, None
    for seed in range(RAM_N_REPEATS):
        rng = np.random.default_rng(seed)
        fit_idx = _balanced_fit_index(aligned, rng, n_per_source)
        if seed == 0:
            fit0 = fit_idx
        for k in [K_MAIN] + K_SENSITIVITY:
            labels, keep, std = _fit_and_assign(ram_full, fit_idx, k, seed)
            if labels is None:
                results["error"] = f"only {int(keep.sum())} informative RAM bytes; clustering skipped"
                return results
            per_seed.extend(_distance_rows(game, k, labels, aligned, seed))
            if seed == 0 and k == K_MAIN:
                src = aligned["source"].to_numpy()
                present = [s for s in SOURCE_ORDER if s in set(src)]
                shares_k15_seed0 = pd.DataFrame({s: _visit_shares(labels, src == s, k) for s in present},
                                                index=pd.Index([f"cluster_{c}" for c in range(k)], name="cluster"))
                results["byte_variance"] = pd.DataFrame({
                    "ram_byte": RAM_COLS, "std_balanced_sample": std,
                    "std_all_rows": ram_full.std(axis=0), "kept": keep,
                }).sort_values("std_balanced_sample", ascending=False).reset_index(drop=True)

    per_seed = pd.DataFrame(per_seed)
    summary = _summarise_distances(per_seed)
    results["cluster_distances_per_repeat"] = per_seed
    results["cluster_distances_sensitivity"] = summary
    results["cluster_distances_k15"] = summary[summary["k"] == K_MAIN].reset_index(drop=True)
    results["cluster_shares_k15"] = shares_k15_seed0
    results["fit_sample"] = pd.DataFrame([{
        "rows_per_source_in_fit": n_per_source, "repeats": RAM_N_REPEATS,
        "rows_total": len(aligned),
        **{f"rows_{s}": int((aligned['source'] == s).sum()) for s in SOURCE_ORDER if s in set(aligned['source'])}}])

    if also_unadjusted:
        rows = []
        for k in [K_MAIN] + K_SENSITIVITY:
            labels, keep, _ = _fit_and_assign(ram_full, np.arange(len(aligned)), k, RANDOM_STATE)
            if labels is not None:
                rows.extend(r for r in _distance_rows(game, k, labels, aligned, RANDOM_STATE)
                            if r["kind"] == "between sources")
        results["cluster_distances_UNADJUSTED_pooled_fit"] = pd.DataFrame(rows).drop(columns=["seed"])

    # figures: PCA fitted on the seed-0 balanced sample; heatmap of seed-0 shares
    if make_figs:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            keep = ram_full[fit0].std(axis=0) > NEAR_CONST_STD_THRESHOLD
            scaler = StandardScaler().fit(ram_full[fit0][:, keep])
            pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaler.transform(ram_full[fit0][:, keep]))
            results["pca_explained_variance"] = pd.DataFrame({
                "component": ["PC1", "PC2"], "explained_variance_ratio": pca.explained_variance_ratio_})
            coords = pca.transform(scaler.transform(ram_full[:, keep]))
            pca_df = aligned[["source"]].copy()
            pca_df["PC1"], pca_df["PC2"] = coords[:, 0], coords[:, 1]
            fig_dir = out_dirs["figures"]
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = {"Human": "#1f77b4", "PPO": "#ff7f0e", "DQN": "#2ca02c"}
            for source in SOURCE_ORDER:
                sub = pca_df[pca_df["source"] == source]
                if len(sub):
                    sub = sub.sample(n=min(len(sub), PCA_SAMPLE_PER_SOURCE), random_state=RANDOM_STATE)
                    ax.scatter(sub["PC1"], sub["PC2"], s=4, alpha=0.4, label=source, color=colors.get(source))
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
            ax.set_title(f"{game}: RAM-state PCA (fitted on balanced sample)")
            ax.legend(markerscale=3)
            fig.tight_layout(); fig.savefig(fig_dir / f"{game}_ram_pca.png", dpi=110); plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 3))
            present = list(shares_k15_seed0.columns)
            im = ax.imshow(shares_k15_seed0[present].to_numpy().T, aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(present))); ax.set_yticklabels(present)
            ax.set_xticks(range(K_MAIN)); ax.set_xticklabels(range(K_MAIN))
            ax.set_xlabel("Cluster"); ax.set_title(f"{game}: RAM cluster visitation (k={K_MAIN}, repeat 1)")
            fig.colorbar(im, ax=ax, label="share")
            fig.tight_layout(); fig.savefig(fig_dir / f"{game}_ram_cluster_heatmap.png", dpi=110); plt.close(fig)
        except Exception as e:  # pragma: no cover
            results["figure_error"] = str(e)
    return results


def analyse(game, status, loaded, fire_rule, report_actions, info, args, gdir, has_fire=False, game_cfg=None):
    also_unadjusted = bool(getattr(args, "also_unadjusted", False))
    window = int(getattr(args, "ngram_window", NGRAM_WINDOW))
    """Compute all descriptor tables for one game into gdir/tables (+figures)."""
    human_raw = loaded.get("Human")
    agent_raws = {s: d for s, d in loaded.items() if s != "Human"}
    tdir = gdir / "tables"
    fdir = gdir / "figures"
    for d in (tdir, fdir):
        d.mkdir(parents=True, exist_ok=True)
    out_dirs = {"tables": tdir, "figures": fdir}

    raw_frames = []
    if human_raw is not None:
        raw_frames.append(human_raw)
    raw_frames.extend(agent_raws.values())
    raw_all = pd.concat(raw_frames, ignore_index=True)

    # ---- 1. performance (only for verified/partial) ----
    if status in PERFORMANCE_STATUSES:
        perf = episode_performance(raw_all, game, game_cfg)
        perf.to_csv(tdir / "01_episode_performance.csv", index=False)
        extra_cols = [c for c in perf.columns if c not in (
            "game", "source", "episode", "orig_file", "logged_rows", "raw_frames", "final_return", "reward_sum",
            "reward_density_per_1000_raw", "first_reward_latency_raw", "life_losses", "done_last", "end_type")]
        perf_summary = summarize_numeric(perf[perf["end_type"] != "incomplete_log_end"], [
            "final_return", "raw_frames", "reward_density_per_1000_raw",
            "first_reward_latency_raw", "life_losses",
        ] + extra_cols)
        if status in LOW_CONFIDENCE_STATUSES:
            perf_summary["performance_confidence"] = "LOW (partial score match)"
        # sources whose rewards are still clipped are not comparable in points
        def _reward_warning(s):
            note = str(info.get(f"{s}_reward", ""))
            if "CLIPPED" in note:
                return "CLIPPED rewards - not comparable"
            if "CHECK SCORE BYTES" in note:
                return "RAM rewards disagree with logged on some rows - check score_bytes"
            if "reset row(s) keep logged reward (clipped)" in note and "penalty rebuilt" not in note:
                return ("last-step reward is clipped in the log - return can be off "
                        "(large in Skiing, where the missed-gate penalty comes on the last step)")
            return ""
        warnings_ = perf_summary["source"].map(_reward_warning)
        if warnings_.any():
            perf_summary["reward_units_warning"] = warnings_
        perf_summary.to_csv(tdir / "02_performance_summary.csv", index=False)
        end_counts = pd.crosstab(perf["source"], perf["end_type"])
        end_counts.to_csv(tdir / "03_episode_end_types.csv")
        info["performance"] = "done" + (" (low-confidence)" if status in LOW_CONFIDENCE_STATUSES else "")
    else:
        info["performance"] = f"skipped (status={status}, no verified score)"

    # ---- 2. alignment + action structure ----
    aligned = build_aligned(human_raw, agent_raws, game)
    aligned.to_parquet(gdir / "aligned.parquet") if args.save_aligned else None

    act = action_structure(aligned, game, fire_rule, report_actions, has_fire, window, also_unadjusted)
    short = act[act["trigram_diversity"].isna()].groupby("source").size()
    if len(short):
        info["ngram_note"] = (f"n-gram diversity in {window}-decision windows; games shorter than one window "
                              "left out: " + ", ".join(f"{s} {n}" for s, n in short.items()))
    act.to_csv(tdir / "04_action_structure_per_episode.csv", index=False)
    act_metrics = ["action_entropy_bits", "action_entropy_norm", "switching_rate", "mean_run_length", "fire_rate",
                   "trigram_diversity", "fourgram_diversity"] + \
                  (["trigram_diversity_UNADJUSTED_whole_game", "fourgram_diversity_UNADJUSTED_whole_game"]
                   if also_unadjusted else []) + \
                  [f"prop_{ACTION_NAMES.get(a, str(a))}" for a in report_actions]
    act_summary = summarize_numeric(act, act_metrics)
    act_summary.to_csv(tdir / "05_action_structure_summary.csv", index=False)
    motifs = motif_table(aligned, game, fire_rule)
    motifs.to_csv(tdir / "06_top_motifs.csv", index=False)
    info["action_structure"] = "done"

    # ---- 3. RAM-state visitation ----
    if not args.skip_ram:
        uniq = ram_exact_uniqueness(aligned, game)
        uniq.to_csv(tdir / "07_ram_exact_uniqueness.csv", index=False)
        ram_res = ram_cluster_analysis(aligned, game, out_dirs, make_figs=not args.no_figures,
                                       also_unadjusted=also_unadjusted)
        for name, tbl in ram_res.items():
            if isinstance(tbl, pd.DataFrame):
                tbl.to_csv(tdir / f"08_ram_{name}.csv", index=name.startswith("cluster_shares"))
        info["ram"] = ram_res.get("error", "done")
    else:
        info["ram"] = "skipped (--skip-ram)"



# ============================================================================
# Per-game driver
# ============================================================================
def run_game(game, game_cfg, args):
    status = game_cfg.get("status", "unknown")
    minimal = game_cfg.get("minimal_action_set", list(range(18)))
    fire_rule, fire_src = resolve_fire_handling(game_cfg)
    report_actions = report_actions_for_rule(minimal, fire_rule)
    has_fire = bool(set(int(a) for a in minimal) & (FIRE_COMBO_ACTIONS | {FIRE_ACTION}))

    human_root = Path(args.human_root)
    agent_root = Path(args.agent_root)
    human_files = find_csvs(human_root, game, ["*.csv"])
    ppo_files = find_csvs(agent_root, game, ["ppo*.csv", "PPO*.csv", "*ppo*.csv"])
    dqn_files = find_csvs(agent_root, game, ["dqn*.csv", "DQN*.csv", "*dqn*.csv"])

    info = {
        "game": game, "status": status,
        "fire_handling": fire_rule, "fire_source": fire_src,
        "n_human_files": len(human_files),
        "has_ppo": bool(ppo_files), "has_dqn": bool(dqn_files),
    }

    if not human_files and not ppo_files and not dqn_files:
        info["result"] = "SKIPPED (no data found)"
        return info

    # load -- files without the 265-column schema (e.g. a raw 60 Hz recording
    # left next to the aligned file) are skipped and listed, not fatal
    skipped = []
    def _valid(files):
        ok = []
        for f in files:
            try:
                validate_columns(f); ok.append(f)
            except ValueError:
                skipped.append(f.name)
        return ok
    human_files, ppo_files, dqn_files = _valid(human_files), _valid(ppo_files), _valid(dqn_files)
    if skipped:
        info["skipped_files"] = "not in the 265-column schema: " + ", ".join(skipped)
    if not human_files and not ppo_files and not dqn_files:
        info["result"] = "SKIPPED (no valid data files)"
        return info

    info["n_ppo_files"] = len(ppo_files)
    info["n_dqn_files"] = len(dqn_files)

    # load every file per source, then bring each source to common units
    loaded = {}
    for src, files in (("Human", human_files), ("PPO", ppo_files), ("DQN", dqn_files)):
        if files:
            loaded[src] = load_source(files, src, game)
    try:
        for src in list(loaded):
            loaded[src], notes = prepare_source(loaded[src], src, game_cfg)
            for key, val in notes.items():
                info[f"{src}_{key}"] = val
    except ValueError as e:
        info["result"] = f"ERROR (preparing data): {e}"
        return info

    # ---- descriptors on all data ----
    out_root = Path(args.out_root)
    analyse(game, status, loaded, fire_rule, report_actions, info, args, out_root / game, has_fire, game_cfg)

    # ---- optional second pass without stuck stretches ----
    if game_cfg.get("report_without_stuck"):
        trimmed, stuck_notes = {}, []
        for src, df in loaded.items():
            trimmed[src], n_cut, n_games = remove_stuck_stretches(df)
            if n_cut:
                stuck_notes.append(f"{src}: {n_cut} rows cut from {n_games} game(s)")
        info["without_stuck"] = "; ".join(stuck_notes) or "nothing to remove"
        sub = dict(info)   # keep the main-pass status fields untouched
        analyse(game, status, trimmed, fire_rule, report_actions, sub, args,
                out_root / game / "without_stuck", has_fire, game_cfg)

    info["result"] = "OK"
    return info


# ============================================================================
# Cross-game master table
# ============================================================================
def build_master_table(out_root: Path, games_run, games_without_stuck=()):
    """One row per game x source with headline descriptors.

    Performance columns are blank for games without a verified score.
    Values are raw within-game numbers -- no cross-game averaging.
    """
    rows = []
    variants = [(g, "all", out_root / g / "tables") for g in games_run] + \
               [(g, "without_stuck", out_root / g / "without_stuck" / "tables") for g in games_without_stuck]
    for game, variant, tdir in variants:
        act_path = tdir / "05_action_structure_summary.csv"
        if not act_path.exists():
            continue
        act = pd.read_csv(act_path)
        perf_path = tdir / "02_performance_summary.csv"
        perf = pd.read_csv(perf_path) if perf_path.exists() else None
        for _, ar in act.iterrows():
            src = ar["source"]
            row = {
                "game": game, "variant": variant, "source": src,
                "entropy_mean": ar.get("action_entropy_bits_mean"),
                "entropy_norm_mean": ar.get("action_entropy_norm_mean"),
                "switching_mean": ar.get("switching_rate_mean"),
                "run_length_mean": ar.get("mean_run_length_mean"),
                "fire_rate_mean": ar.get("fire_rate_mean"),
                "performance_available": perf is not None,
            }
            if perf is not None:
                pr = perf[perf["source"] == src]
                if len(pr):
                    row["return_mean"] = pr.iloc[0].get("final_return_mean")
                    row["reward_density_mean"] = pr.iloc[0].get("reward_density_per_1000_raw_mean")
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="game_config.json")
    ap.add_argument("--human-root", default="aligned")
    ap.add_argument("--agent-root", default="agent_test_data")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--games", nargs="*", default=None,
                    help="specific games to run (default: all usable in config)")
    ap.add_argument("--skip-ram", action="store_true",
                    help="skip PCA/clustering (faster; keeps exact-uniqueness off too)")
    ap.add_argument("--no-figures", action="store_true", help="skip PNG figures")
    ap.add_argument("--save-aligned", action="store_true",
                    help="also write each game's aligned trace as parquet")
    ap.add_argument("--also-unadjusted", action="store_true",
                    help="also report the original (unadjusted) RAM clustering and whole-game "
                         "n-gram diversity, labelled UNADJUSTED, next to the adjusted results")
    ap.add_argument("--ngram-window", type=int, default=NGRAM_WINDOW,
                    help=f"window length in decisions for n-gram diversity (default {NGRAM_WINDOW})")
    args = ap.parse_args()

    cfg = json.load(open(args.config))

    if args.games:
        games = [g for g in args.games if g in cfg]
        missing = [g for g in args.games if g not in cfg]
        for g in missing:
            print(f"  ! '{g}' not in config, skipping")
    else:
        games = sorted(cfg.keys())

    out_root = Path(args.out_root)
    (out_root / "_summary").mkdir(parents=True, exist_ok=True)

    print(f"Running behavioural pipeline on {len(games)} game(s)")
    print(f"  human data : {args.human_root}/<game>/*.csv")
    print(f"  agent data : {args.agent_root}/<game>/(ppo|dqn)*.csv")
    print(f"  outputs    : {args.out_root}/<game>/")
    print("=" * 70)

    infos = []
    for game in games:
        info = run_game(game, cfg[game], args)
        infos.append(info)
        status_msg = info.get("result", "?")
        extra = []
        if "performance" in info:
            extra.append(f"perf={info['performance'].split()[0]}")
        if "action_structure" in info:
            extra.append("action=done")
        if "ram" in info:
            extra.append(f"ram={info['ram'].split()[0] if info['ram'] else 'done'}")
        fire = f"fire={info['fire_handling']}({info['fire_source'][0]})"
        print(f"  {game:20s} {status_msg:22s} {fire:16s} {' '.join(extra)}")
        for key, val in info.items():
            if key.split("_", 1)[0] in SOURCE_ORDER or key in ("without_stuck", "ngram_note", "skipped_files"):
                print(f"      {key:28s} {val}")

    summary = pd.DataFrame(infos)
    summary.to_csv(out_root / "_summary" / "run_summary.csv", index=False)

    fire_tbl = summary[["game", "status", "fire_handling", "fire_source"]].copy()
    fire_tbl.to_csv(out_root / "_summary" / "fire_handling.csv", index=False)

    games_ok = [i["game"] for i in infos if i.get("result") == "OK"]
    if games_ok:
        games_ws = [i["game"] for i in infos if i.get("result") == "OK" and "without_stuck" in i]
        master = build_master_table(out_root, games_ok, games_ws)
        master.to_csv(out_root / "_summary" / "cross_game_master.csv", index=False)
        print("=" * 70)
        print(f"Analysed {len(games_ok)} game(s) with data. "
              f"Cross-game master table: {out_root}/_summary/cross_game_master.csv")
    else:
        print("=" * 70)
        print("No games had data yet. Generate traces into the data folders and re-run.")
        print("Run summary (including auto-derived FIRE rules) written to "
              f"{out_root}/_summary/run_summary.csv")


if __name__ == "__main__":
    main()
