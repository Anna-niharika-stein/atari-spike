#!/usr/bin/env python3
"""
set_agent_action_set.py  --  record the action space your AGENTS actually used,
so align_to_agent maps human actions to the SAME indices the agent emitted.

Why this exists
---------------
align_to_agent must remap the human's ALE-full actions to the agent's action
space. ALE's getMinimalActionSet() is NOT always what the agent trained on. For
this project's Pong agent the actions are {0,1,2,3,4,5} (base: NOOP/FIRE/UP/
RIGHT/LEFT/DOWN), while ALE's minimal Pong set is {0,1,3,4,11,12}. Aligning to
the wrong set silently maps real moves to NOOP (run_pipeline now flags this).

This reads one or more agent CSVs (which have an `action` column), takes the
sorted set of distinct action integers the agent ACTUALLY emitted, and stores it
as `agent_action_set` in game_config.json for that game.

Usage:
    python set_agent_action_set.py pong  ppo_agent_pong_30ep.csv
    python set_agent_action_set.py breakout dqn_breakout_*.csv
"""
import argparse, csv, glob, sys
from atari_common import load_config_or_empty, save_config, CONFIG_PATH


def action_set_from_csvs(paths):
    seen = set()
    for p in paths:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            if "action" not in r.fieldnames:
                sys.exit(f"{p}: no 'action' column")
            for row in r:
                seen.add(int(row["action"]))
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game")
    ap.add_argument("csvs", nargs="+", help="agent CSV(s) with an action column (globs ok)")
    args = ap.parse_args()

    paths = [p for pat in args.csvs for p in glob.glob(pat)]
    if not paths:
        sys.exit("no CSVs matched")

    aset = action_set_from_csvs(paths)
    cfg = load_config_or_empty()
    entry = cfg.setdefault(args.game, {"game": args.game, "score_bytes": [],
                                       "lives": {"mode": "none"}, "status": "lives_seeded"})
    entry["agent_action_set"] = aset
    save_config(cfg)

    print(f"{args.game}: agent_action_set = {aset}  (from {len(paths)} file(s))")
    print(f"saved -> {CONFIG_PATH}")
    # sanity note
    if aset == [0, 1, 2, 3, 4, 5]:
        print("  (base 6-action set: NOOP,FIRE,UP,RIGHT,LEFT,DOWN)")


if __name__ == "__main__":
    main()
