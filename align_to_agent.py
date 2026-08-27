#!/usr/bin/env python3
"""
align_to_agent.py  --  convert a raw browser human session into the SAME schema
as the RL-agent thesis CSVs, so both drop into the identical descriptor pipeline.

It handles the three alignment steps the agent format requires:

  1. Frameskip aggregation  -- the browser logs every 60 Hz frame; the agent logs
     one decision per `--frameskip` frames (default 4). We group frames into
     decision steps, taking the MODAL action over each window (the thesis's
     modal-window temporal alignment).
  2. Action remap           -- the browser logs ALE full-space actions
     [0,1,3,4]; the agent uses the game's MINIMAL action set reindexed to
     [0,1,2,3]. We remap via ale.getMinimalActionSet().
  3. RAM decode             -- reward / lives / return are decoded from RAM using
     game_config.json (same byte map validated by decode_ram.py --validate).

Output columns exactly match the agent CSV:
  run_ts, episode, step, action, reward, done, episode_return,
  lives_pre, lives_post, ram_pre_0..127, ram_post_0..127

Usage:
    python align_to_agent.py human_session_breakout_....csv
    python align_to_agent.py session.csv --frameskip 4 --out human_breakout_aligned.csv
"""
import argparse, json, os, sys
from collections import Counter
import functools
import numpy as np
from atari_common import bcd, load_config, read_session, decode


def full_to_minimal(game, cfg_entry=None):
    """Map ALE full-space action value -> the AGENT's action-space index.

    IMPORTANT: this must match the action space your agents were actually
    trained/evaluated on, which is NOT always ALE's getMinimalActionSet().
    (E.g. this project's Pong agent used the base set {0,1,2,3,4,5} = UP/DOWN,
    whereas ALE's minimal Pong set is {0,1,3,4,11,12} = RIGHT/LEFT — remapping
    to the latter would drop every real paddle move to NOOP.)

    Resolution order:
      1. cfg_entry["agent_action_set"]  -- the agent's true action space
         (add it with set_agent_action_set.py, derived from the agent CSVs).
      2. cfg_entry["minimal_action_set"] -- ALE minimal set (only correct if the
         agent used the minimal set).
      3. query ALE getMinimalActionSet() as a last resort.
    """
    if cfg_entry and cfg_entry.get("agent_action_set"):
        aset = tuple(cfg_entry["agent_action_set"])
    elif cfg_entry and cfg_entry.get("minimal_action_set"):
        aset = tuple(cfg_entry["minimal_action_set"])
    else:
        from ale_py import ALEInterface, roms
        ale = ALEInterface(); ale.loadROM(str(roms.get_rom_path(game)))
        aset = tuple(int(a.value) for a in ale.getMinimalActionSet())
    return {full_val: i for i, full_val in enumerate(aset)}, aset


def timestamp_from_meta(meta):
    """'2026-07-21T11:57:53.077Z' -> '2026-07-21_11-57-53' (agent run_ts style)."""
    ts = meta.get("recorded_utc", "")
    try:
        date, time = ts.split("T")
        time = time.split(".")[0].rstrip("Z")
        return f"{date}_{time.replace(':', '-')}"
    except Exception:
        return "human_session"


def convert(path, frameskip, out):
    cfg_all = load_config()
    meta, actions_full, ram, _ = read_session(path)
    game = meta.get("game", "")
    if game not in cfg_all:
        sys.exit(f"'{game}' not in game_config.json — run discover_bytes.py {game}")
    if not cfg_all[game].get("score_bytes"):
        sys.exit(f"'{game}' is lives-seeded only (no score_bytes) — run discover_bytes.py {game}")
    cfg = cfg_all[game]

    remap, minset = full_to_minimal(game, cfg)
    unknown = 0

    # Per-frame score / reward / lives / terminal from the SAME validated decoder
    # decode_ram uses and --validate proves against ALE. No reward reimplementation
    # here — the window reward is just the sum of validated per-frame rewards.
    dec = decode(ram, cfg)
    score, reward_pf, terminal_pf = dec["score"], dec["reward"], dec["terminal"]
    lives_pf = dec["lives"]                       # None if the game has no lives byte

    n = len(ram)
    run_ts = timestamp_from_meta(meta)

    # ── aggregate frames into decision steps of length `frameskip` ──────────
    # An episode ends on the transition to terminal (once), and the post-game-over
    # dead frames are dropped, because the agent env resets on done and never logs
    # those frames -- keeping them would make the human data non-comparable.
    step_rows = []
    episode = 1
    step_in_ep = 0
    ep_return = 0.0
    start_new_episode = False

    for start in range(0, n - 1, frameskip):
        end = min(start + frameskip, n - 1)
        ram_pre = ram[start]
        ram_post = ram[end]

        pre_term = bool(terminal_pf[start])
        post_term = bool(terminal_pf[end])

        # fully inside a game-over / dead stretch -> skip (agent has no such frames)
        if pre_term and post_term:
            continue

        # a life/game has resumed after a terminal -> begin a fresh episode
        if start_new_episode:
            episode += 1
            step_in_ep = 0
            ep_return = 0.0
            start_new_episode = False

        # modal action over the window, remapped to minimal-set index
        acts = actions_full[start:end]
        modal_full = Counter(acts.tolist()).most_common(1)[0][0]
        act_min = remap.get(modal_full, None)
        if act_min is None:
            act_min = 0
            unknown += 1

        # reward over the window = sum of validated per-frame rewards
        reward = int(reward_pf[start:end].sum())

        lives_pre = int(lives_pf[start]) if lives_pf is not None else -1
        lives_post = int(lives_pf[end]) if lives_pf is not None else -1

        # done fires once, on the transition into terminal
        done = 1 if (post_term and not pre_term) else 0

        ep_return += reward
        step_rows.append((run_ts, episode, step_in_ep, act_min, float(reward), done,
                          float(ep_return), lives_pre, lives_post, ram_pre, ram_post))
        step_in_ep += 1

        # On a terminal step, arm a new episode for the NEXT live window. (Do NOT
        # also bump `episode` here — the start_new_episode handler above owns that,
        # or the double bump skips episode numbers, e.g. 1,3,5…)
        if done:
            start_new_episode = True

    # ── write in the exact agent column order ───────────────────────────────
    out = out or path.rsplit(".", 1)[0] + "_aligned.csv"
    head = (["run_ts", "episode", "step", "action", "reward", "done", "episode_return",
             "lives_pre", "lives_post"]
            + [f"ram_pre_{i}" for i in range(128)]
            + [f"ram_post_{i}" for i in range(128)])
    with open(out, "w") as fh:
        fh.write(",".join(head) + "\n")
        for (ts, ep, st, a, r, dn, ret, lp, lq, rpre, rpost) in step_rows:
            fh.write(",".join([ts, str(ep), str(st), str(a), f"{r:.1f}", str(dn), f"{ret:.1f}",
                               str(lp), str(lq)]
                              + [str(int(x)) for x in rpre]
                              + [str(int(x)) for x in rpost]) + "\n")

    n_ep = max((r[1] for r in step_rows), default=0)
    total_ret = sum(r[4] for r in step_rows)
    steps = len(step_rows)
    print(f"game            : {game}")
    print(f"input frames    : {n}")
    print(f"frameskip       : {frameskip}")
    print(f"decision steps  : {steps}")
    print(f"episodes        : {n_ep}")
    print(f"action remap     : full {minset} -> minimal {list(range(len(minset)))}")
    if unknown:
        pct = 100 * unknown / max(steps, 1)
        print(f"  note: {unknown} steps ({pct:.1f}%) had an action outside the minimal set -> mapped to NOOP")
    print(f"total return    : {total_ret:.0f}")
    print(f"wrote           : {out}")
    # Return a stats dict (run_pipeline reads `unknown`/`steps` for QA). Older
    # callers that expected the path can use info["out"].
    return dict(out=out, game=game, input_frames=n, steps=steps, episodes=n_ep,
                unknown=unknown, total_return=total_ret, minimal_set=list(minset))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="raw browser human session CSV")
    ap.add_argument("--frameskip", type=int, default=4, help="frames per decision step (default 4, matches AtariWrapper)")
    ap.add_argument("--out")
    args = ap.parse_args()
    convert(args.csv, args.frameskip, args.out)


if __name__ == "__main__":
    main()
