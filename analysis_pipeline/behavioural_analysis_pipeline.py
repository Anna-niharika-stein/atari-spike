#!/usr/bin/env python3
"""
behavioural_analysis_pipeline.py

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
This pipeline runs the full behavioural analysis across all usable Atari games
in game_config.json, producing the three descriptor classes developed in my
MSc thesis and extended here to scale across the full Atari-57 benchmark.

The descriptor logic matches the original per-game thesis notebooks so that
results remain directly comparable. The key addition here is that the pipeline
is fully config-driven -- one script handles all games rather than one notebook
per game, with per-game variation handled through game_config.json.

-------------------------------------------------------------------------------
Descriptor classes
-------------------------------------------------------------------------------
For each game and each available gameplay source (Human, PPO, DQN):

  1. Performance
       Episode return, reward density per 1000 raw frames, first-reward
       latency, life losses, episode end type (natural vs. capped).
       Only computed for games with status = verified or partial in the config.

  2. Action structure
       Action distribution, entropy, switching rate, mean run length,
       trigram and four-gram motif diversity, and top motifs.
       Computed for all games regardless of score-byte status.

  3. RAM-state visitation
       Exact RAM-state uniqueness and Jaccard overlap, PCA projection
       (exploratory only), MiniBatchKMeans clustering (k=15, with sensitivity
       checks at k=10 and k=20), Jensen-Shannon and total-variation distances
       between source visitation distributions.
       Computed for all games regardless of score-byte status.

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
The rule applied to each game is logged to outputs/_summary/fire_handling.csv.

-------------------------------------------------------------------------------
Data layout
-------------------------------------------------------------------------------
  human_raw_test_data/<game>/*.csv     aligned human session CSV (265 cols)
  agent_test_data/<game>/ppo*.csv      PPO agent log (265 cols)
  agent_test_data/<game>/dqn*.csv      DQN agent log (265 cols)

All CSVs must follow the 265-column schema:
  run_ts, episode, step, action, reward, done, episode_return,
  lives_pre, lives_post, ram_pre_0..127, ram_post_0..127

Human sessions should be the aligned output of run_pipeline.py, not the
raw ~60Hz recorder output.

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
python behavioural_analysis_pipeline.py --games breakout --human-root path/to/human_data --agent-root path/to/agent_data
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
HUMAN_RAW_FRAMES_PER_ROW = 1
CAP_THRESHOLD_LOGGED_ROWS = 9990   # episodes capped at ~10,000 logged steps

BASE_COLS = ["run_ts", "episode", "step", "action", "reward", "done",
             "episode_return", "lives_pre", "lives_post"]
RAM_COLS = [f"ram_pre_{i}" for i in range(128)]
REQUIRED_COLS = BASE_COLS + RAM_COLS

# RAM clustering / PCA parameters
NEAR_CONST_STD_THRESHOLD = 1.0
K_MAIN = 15
K_SENSITIVITY = [10, 20]
PCA_SAMPLE_PER_SOURCE = 5000
RANDOM_STATE = 0
KMEANS_N_INIT = 10

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


def load_agent(path: Path, source: str, game: str):
    df = pd.read_csv(path, usecols=REQUIRED_COLS)
    meta = pd.DataFrame({
        "source": source, "game": game, "orig_file": path.name,
        "global_episode": df["episode"].astype(int),
        "raw_frames_per_row": AGENT_RAW_FRAMES_PER_ROW,
    }, index=df.index)
    return pd.concat([df, meta], axis=1)


def load_human(files, game: str):
    parts, ep_counter = [], 0
    for f in files:
        df = pd.read_csv(f, usecols=REQUIRED_COLS)
        for _, sub in df.groupby("episode", sort=False):
            ep_counter += 1
            sub = sub.copy()
            meta = pd.DataFrame({
                "source": "Human", "game": game, "orig_file": f.name,
                "global_episode": ep_counter,
                "raw_frames_per_row": HUMAN_RAW_FRAMES_PER_ROW,
            }, index=sub.index)
            parts.append(pd.concat([sub, meta], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else None


# ============================================================================
# 1. Performance descriptors
# ============================================================================
def episode_performance(raw_all, game):
    rows = []
    for (source, ep), g in raw_all.groupby(["source", "global_episode"], sort=True):
        g = g.sort_values("step")
        raw_mult = int(g["raw_frames_per_row"].iloc[0])
        logged_rows = len(g)
        raw_frames = logged_rows * raw_mult

        rewards = g["reward"].to_numpy()
        nz = np.flatnonzero(rewards != 0)
        first_reward_latency_raw = (int(nz[0]) + 1) * raw_mult if len(nz) else np.nan

        if {"lives_pre", "lives_post"}.issubset(g.columns):
            life_losses = int((g["lives_post"].to_numpy() < g["lives_pre"].to_numpy()).sum())
        else:
            life_losses = np.nan

        capped = logged_rows >= CAP_THRESHOLD_LOGGED_ROWS
        final_return = float(g["episode_return"].iloc[-1])
        rows.append({
            "game": game, "source": source, "episode": ep,
            "orig_file": g["orig_file"].iloc[0],
            "logged_rows": logged_rows, "raw_frames": raw_frames,
            "final_return": final_return,
            "reward_sum": float(g["reward"].sum()),
            "reward_density_per_1000_raw": final_return / raw_frames * 1000 if raw_frames else np.nan,
            "first_reward_latency_raw": first_reward_latency_raw,
            "life_losses": life_losses,
            "done_last": int(g["done"].iloc[-1]),
            "end_type": "truncated_at_cap" if capped else "natural_terminal",
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
        frames.append(aggregate_human_to_windows(human_raw, game))
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


def top_ngrams(actions, n, k=10):
    a = list(map(int, actions))
    if len(a) < n:
        return []
    grams = [tuple(a[i:i+n]) for i in range(len(a) - n + 1)]
    return Counter(grams).most_common(k)


def action_structure(aligned, game, fire_rule, report_actions):
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
            "switching_rate": switching_rate(actions),
            "mean_run_length": mean_run_length(actions),
            "trigram_diversity": ngram_diversity(actions, 3),
            "fourgram_diversity": ngram_diversity(actions, 4),
        }
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
        seq = apply_fire_rule(g.sort_values(["global_episode", "step"])["action"].astype(int).to_numpy(),
                              fire_rule)
        for n in (3, 4):
            for motif, count in top_ngrams(seq, n=n, k=k):
                rows.append({
                    "game": game, "source": source, "n": n,
                    "motif": "-".join(ACTION_NAMES.get(a, str(a)) for a in motif),
                    "count": count,
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


def ram_cluster_analysis(aligned, game, out_dirs, make_figs=True):
    """PCA + k-means visitation, JS/TV cluster distances. Returns dict of tables."""
    if not _HAVE_SKLEARN:
        return {"error": "sklearn not available; RAM clustering skipped"}

    ram_full = aligned[RAM_COLS].to_numpy(dtype=np.float64)
    byte_std = ram_full.std(axis=0)
    keep_mask = byte_std > NEAR_CONST_STD_THRESHOLD
    kept = [RAM_COLS[i] for i in range(128) if keep_mask[i]]

    byte_var_tbl = pd.DataFrame({
        "ram_byte": RAM_COLS, "std": byte_std, "kept": keep_mask,
    }).sort_values("std", ascending=False).reset_index(drop=True)

    results = {"byte_variance": byte_var_tbl}

    if len(kept) < 2:
        results["error"] = f"only {len(kept)} informative RAM bytes; clustering skipped"
        return results

    ram_matrix = aligned[kept].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    ram_scaled = scaler.fit_transform(ram_matrix)

    # PCA (visualization only)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(ram_scaled)
    pca_df = aligned[["source", "global_episode"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    results["pca_explained_variance"] = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })

    # sampled PCA scatter for plotting
    sample_parts = []
    for source, sub in pca_df.groupby("source"):
        n = min(len(sub), PCA_SAMPLE_PER_SOURCE)
        if n > 0:
            sample_parts.append(sub.sample(n=n, random_state=RANDOM_STATE))
    pca_sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pca_df.iloc[0:0]

    # k-means visitation for main k and sensitivity k's
    def cluster_distances(k):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(ram_scaled)
        tmp = aligned[["source", "global_episode"]].copy()
        tmp["cluster"] = labels
        # time-weighted visitation share per source
        shares = {}
        for source, sub in tmp.groupby("source"):
            vc = sub["cluster"].value_counts().reindex(range(k), fill_value=0).to_numpy()
            shares[source] = vc / vc.sum() if vc.sum() else vc
        present = [s for s in SOURCE_ORDER if s in shares]
        dist_rows = []
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                dist_rows.append({
                    "game": game, "k": k, "comparison": f"{a} vs {b}",
                    "js_time": js_divergence(shares[a], shares[b]),
                    "tv_time": total_variation_distance(shares[a], shares[b]),
                })
        share_tbl = pd.DataFrame(shares, index=[f"cluster_{c}" for c in range(k)])
        return pd.DataFrame(dist_rows), share_tbl, tmp

    main_dist, main_shares, main_assign = cluster_distances(K_MAIN)
    results["cluster_distances_k15"] = main_dist
    results["cluster_shares_k15"] = main_shares

    sens_frames = [main_dist]
    for k in K_SENSITIVITY:
        d, _, _ = cluster_distances(k)
        sens_frames.append(d)
    results["cluster_distances_sensitivity"] = pd.concat(sens_frames, ignore_index=True)

    # optional figures
    if make_figs:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_dir = out_dirs["figures"]
            # PCA scatter
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = {"Human": "#1f77b4", "PPO": "#ff7f0e", "DQN": "#2ca02c"}
            for source in SOURCE_ORDER:
                sub = pca_sample[pca_sample["source"] == source]
                if len(sub):
                    ax.scatter(sub["PC1"], sub["PC2"], s=4, alpha=0.4,
                               label=source, color=colors.get(source))
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
            ax.set_title(f"{game}: RAM-state PCA sample")
            ax.legend(markerscale=3)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{game}_ram_pca.png", dpi=110)
            plt.close(fig)

            # cluster visitation heatmap (k=15)
            fig, ax = plt.subplots(figsize=(9, 3))
            present = [s for s in SOURCE_ORDER if s in main_shares.columns]
            mat = main_shares[present].to_numpy().T
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(present))); ax.set_yticklabels(present)
            ax.set_xticks(range(K_MAIN)); ax.set_xticklabels(range(K_MAIN))
            ax.set_xlabel("Cluster"); ax.set_title(f"{game}: RAM cluster visitation (k={K_MAIN})")
            fig.colorbar(im, ax=ax, label="share")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{game}_ram_cluster_heatmap.png", dpi=110)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            results["figure_error"] = str(e)

    return results


# ============================================================================
# Per-game driver
# ============================================================================
def run_game(game, game_cfg, args):
    status = game_cfg.get("status", "unknown")
    minimal = game_cfg.get("minimal_action_set", list(range(18)))
    fire_rule, fire_src = resolve_fire_handling(game_cfg)
    report_actions = report_actions_for_rule(minimal, fire_rule)

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

    # load
    try:
        for p in human_files + ppo_files[:1] + dqn_files[:1]:
            validate_columns(p)
    except ValueError as e:
        info["result"] = f"ERROR (bad schema): {e}"
        return info

    human_raw = load_human(human_files, game) if human_files else None
    agent_raws = {}
    if ppo_files:
        agent_raws["PPO"] = load_agent(ppo_files[0], "PPO", game)
    if dqn_files:
        agent_raws["DQN"] = load_agent(dqn_files[0], "DQN", game)

    out_root = Path(args.out_root)
    gdir = out_root / game
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
        perf = episode_performance(raw_all, game)
        perf.to_csv(tdir / "01_episode_performance.csv", index=False)
        perf_summary = summarize_numeric(perf, [
            "final_return", "raw_frames", "reward_density_per_1000_raw",
            "first_reward_latency_raw", "life_losses",
        ])
        if status in LOW_CONFIDENCE_STATUSES:
            perf_summary["performance_confidence"] = "LOW (partial score match)"
        perf_summary.to_csv(tdir / "02_performance_summary.csv", index=False)
        end_counts = pd.crosstab(perf["source"], perf["end_type"])
        end_counts.to_csv(tdir / "03_episode_end_types.csv")
        info["performance"] = "done" + (" (low-confidence)" if status in LOW_CONFIDENCE_STATUSES else "")
    else:
        info["performance"] = f"skipped (status={status}, no verified score)"

    # ---- 2. alignment + action structure ----
    aligned = build_aligned(human_raw, agent_raws, game)
    aligned.to_parquet(gdir / "aligned.parquet") if args.save_aligned else None

    act = action_structure(aligned, game, fire_rule, report_actions)
    act.to_csv(tdir / "04_action_structure_per_episode.csv", index=False)
    act_metrics = ["action_entropy_bits", "switching_rate", "mean_run_length",
                   "trigram_diversity", "fourgram_diversity"] + \
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
        ram_res = ram_cluster_analysis(aligned, game, out_dirs, make_figs=not args.no_figures)
        for name, tbl in ram_res.items():
            if isinstance(tbl, pd.DataFrame):
                tbl.to_csv(tdir / f"08_ram_{name}.csv", index=False)
        info["ram"] = ram_res.get("error", "done")
    else:
        info["ram"] = "skipped (--skip-ram)"

    info["result"] = "OK"
    return info


# ============================================================================
# Cross-game master table
# ============================================================================
def build_master_table(out_root: Path, games_run):
    """One row per game x source with headline descriptors.

    Performance columns are blank for games without a verified score.
    Values are raw within-game numbers -- no cross-game averaging.
    """
    rows = []
    for game in games_run:
        tdir = out_root / game / "tables"
        act_path = tdir / "05_action_structure_summary.csv"
        if not act_path.exists():
            continue
        act = pd.read_csv(act_path)
        perf_path = tdir / "02_performance_summary.csv"
        perf = pd.read_csv(perf_path) if perf_path.exists() else None
        for _, ar in act.iterrows():
            src = ar["source"]
            row = {
                "game": game, "source": src,
                "entropy_mean": ar.get("action_entropy_bits_mean"),
                "switching_mean": ar.get("switching_rate_mean"),
                "run_length_mean": ar.get("mean_run_length_mean"),
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
    ap.add_argument("--human-root", default="human_raw_test_data")
    ap.add_argument("--agent-root", default="agent_test_data")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--games", nargs="*", default=None,
                    help="specific games to run (default: all usable in config)")
    ap.add_argument("--skip-ram", action="store_true",
                    help="skip PCA/clustering (faster; keeps exact-uniqueness off too)")
    ap.add_argument("--no-figures", action="store_true", help="skip PNG figures")
    ap.add_argument("--save-aligned", action="store_true",
                    help="also write each game's aligned trace as parquet")
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

    summary = pd.DataFrame(infos)
    summary.to_csv(out_root / "_summary" / "run_summary.csv", index=False)

    fire_tbl = summary[["game", "status", "fire_handling", "fire_source"]].copy()
    fire_tbl.to_csv(out_root / "_summary" / "fire_handling.csv", index=False)

    games_ok = [i["game"] for i in infos if i.get("result") == "OK"]
    if games_ok:
        master = build_master_table(out_root, games_ok)
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
