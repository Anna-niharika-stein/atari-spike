#!/usr/bin/env python3
"""
run_pipeline.py: raw browser recording(s) -> agent-comparable CSV(s).

Chains every step in the pipeline:

    raw human_session_*.csv
        -> ensure the game's byte map exists (auto-discover if missing)
        -> decode score / lives / reward from RAM        (game_config.json)
        -> aggregate 60 Hz frames into decision steps     (frameskip)
        -> remap actions to the agent's minimal set
        -> write a CSV in the EXACT agent schema
        -> print a QA summary and flag anything off

Single file or a whole folder:

    python run_pipeline.py human_session_breakout_....csv
    python run_pipeline.py ./recordings --out-dir ./aligned
    python run_pipeline.py ./recordings --auto-discover     # set up unseen games automatically

The browser stays a pure recorder; this does everything downstream in one place.
"""
import argparse, csv, glob, os, sys
import numpy as np

import align_to_agent as ata           # convert()
from atari_common import read_session, load_config_or_empty

AGENT_COLS = (["run_ts", "episode", "step", "action", "reward", "done", "episode_return",
               "lives_pre", "lives_post"]
              + [f"ram_pre_{i}" for i in range(128)]
              + [f"ram_post_{i}" for i in range(128)])


def is_raw_session(path):
    """A raw browser recording: header starts frame,t_ms,action,ram_0 (not an aligned/decoded file)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                return line.startswith("frame,t_ms,action,ram_0")
    except Exception:
        return False
    return False


def ensure_game_configured(game, auto_discover):
    import discover_bytes as db
    cfg = load_config_or_empty()
    existing = cfg.get(game)
    # A fully-configured entry HAS score_bytes. A lives-seeded entry (from
    # seed_config.py) has the lives byte but no score_bytes yet -> still needs
    # discovery to fill the score map.
    if existing and existing.get("score_bytes"):
        return existing.get("status", "unknown")
    if not auto_discover:
        hint = "lives-seeded but needs score_bytes" if existing else "not in game_config.json"
        sys.exit(f"\n'{game}' is {hint}.\n"
                 f"Run:  python discover_bytes.py {game}\n"
                 f"or re-run this pipeline with --auto-discover.")
    print(f"[setup] '{game}' needs a score map -> auto-discovering…")
    disc = db.discover(game, seed=0, verbose=False)
    if disc is None:
        sys.exit(f"Could not discover byte map for '{game}'.")
    entry, status = db.build_config_entry(game, disc, seed_entry=existing)
    if entry is None or status == "failed":
        sys.exit(f"Discovery for '{game}' failed: {entry['note'] if entry else 'no candidates'}")
    cfg[game] = entry
    db.save_config(cfg)
    print(f"[setup] '{game}' configured ({status}): {entry['note']}")
    return status


def qa(out_path):
    """Re-read the aligned output and sanity-check it against the agent schema."""
    import csv
    with open(out_path, newline="") as f:
        r = csv.reader(f); header = next(r); rows = [row for row in r]
    idx = {n: i for i, n in enumerate(header)}
    def col(name, typ=float): return np.array([typ(row[idx[name]]) for row in rows]) if rows else np.array([])

    problems = []
    if header != AGENT_COLS:
        problems.append("column schema does NOT match the agent format")
    if not rows:
        problems.append("no decision steps produced")
        return problems, {}

    action = col("action", int); reward = col("reward"); epret = col("episode_return")
    lives_pre = col("lives_pre", int); done = col("done", int); ep = col("episode", int)

    if not np.allclose(np.cumsum(reward), epret):
        # cumulative return should match within each episode
        okc = True
        for e in np.unique(ep):
            m = ep == e
            if not np.allclose(np.cumsum(reward[m]), epret[m]): okc = False
        if not okc: problems.append("cumsum(reward) != episode_return")
    if action.min() < 0:
        problems.append("negative action index")

    switch = float(np.mean(np.diff(action) != 0)) if len(action) > 1 else 0.0
    p = np.bincount(action, minlength=int(action.max()) + 1) / len(action)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))

    stats = dict(steps=len(rows), episodes=int(len(np.unique(ep))),
                 total_return=float(epret[-1] if len(epret) else 0),
                 dones=int(done.sum()), switch=switch, entropy=entropy,
                 action_dist={int(k): int(v) for k, v in zip(*np.unique(action, return_counts=True))})
    return problems, stats


def timing_check(t_ms, frameskip):
    """Flag dropped frames: a browser stall makes 60 Hz assumption (and frameskip) wrong."""
    if t_ms is None or len(t_ms) < 3:
        return None, []
    dt = np.diff(t_ms)
    med = float(np.median(dt))
    # a healthy 60 Hz recording sits ~16.7 ms/frame; flag frames >2x median as stalls
    stalls = int(np.sum(dt > 2 * med)) if med > 0 else 0
    probs = []
    if med > 20:
        probs.append(f"slow capture (~{1000/med:.0f} Hz) — frameskip {frameskip} may not match the agent")
    if stalls > 0:
        probs.append(f"{stalls} dropped-frame stall(s) — frameskip aggregation may be misaligned")
    return med, probs


def process(path, out_dir, frameskip, auto_discover):
    meta, _, _, t_ms = read_session(path)
    game = meta.get("game", "")
    participant = meta.get("participant", "")
    if not game:
        print(f"  ! {os.path.basename(path)}: no game in metadata — skipping"); return None
    ensure_game_configured(game, auto_discover)

    med_dt, timing_probs = timing_check(t_ms, frameskip)

    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(out_dir, base + "_aligned.csv")
    os.makedirs(out_dir, exist_ok=True)

    # suppress convert()'s own prints; we show a unified QA summary instead
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = ata.convert(path, frameskip, out)

    problems, stats = qa(out)
    problems = problems + timing_probs

    # Out-of-minimal-set actions were logged by the human but don't exist in the
    # agent's action space, so they collapsed to NOOP. A high rate means the
    # human's controls don't map cleanly to this game and its data needs a look.
    if info and info.get("steps"):
        unk_pct = 100 * info["unknown"] / info["steps"]
        if unk_pct > 2.0:
            problems.append(f"{info['unknown']} steps ({unk_pct:.1f}%) used an action outside "
                            f"the minimal set {info['minimal_set']} -> logged as NOOP")
    print(f"\n  {os.path.basename(path)}  ->  {os.path.basename(out)}")
    who = f" | participant {participant}" if participant else ""
    print(f"    game {game}{who} | steps {stats.get('steps','?')} | episodes {stats.get('episodes','?')} "
          f"| return {stats.get('total_return','?'):.0f} | done×{stats.get('dones','?')}")
    if stats:
        ad = stats["action_dist"]; tot = stats["steps"]
        dist = "  ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(ad.items()))
        hz = f"{1000/med_dt:.0f}Hz" if med_dt else "?"
        print(f"    actions [{dist}]   entropy {stats['entropy']:.2f} bits   switching {100*stats['switch']:.0f}%   capture {hz}")
    if problems:
        print("    \u26a0 " + "; ".join(problems))
    else:
        print("    \u2713 schema matches agent format, values consistent")

    stats.update(file=os.path.basename(path), participant=participant, game=game,
                 capture_hz=(round(1000/med_dt, 1) if med_dt else ""),
                 flags="; ".join(problems), out=out)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a raw session CSV, or a folder of them")
    ap.add_argument("--out-dir", default="aligned", help="where to write aligned CSVs (default ./aligned)")
    ap.add_argument("--frameskip", type=int, default=4, help="frames per decision step (default 4)")
    ap.add_argument("--auto-discover", action="store_true", help="auto-configure any unseen game")
    ap.add_argument("--combine", metavar="CSV",
                    help="also concatenate all aligned outputs into one dataset CSV (agent schema preserved)")
    ap.add_argument("--no-manifest", action="store_true", help="skip writing the manifest summary")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(f for f in glob.glob(os.path.join(args.input, "*.csv")) if is_raw_session(f))
        if not files:
            sys.exit(f"No raw session CSVs found in {args.input}")
        print(f"Found {len(files)} raw recording(s) in {args.input}")
    else:
        if not is_raw_session(args.input):
            sys.exit(f"{args.input} doesn't look like a raw browser session "
                     f"(expected header: frame,t_ms,action,ram_0,...)")
        files = [args.input]

    results = []
    for f in files:
        try:
            s = process(f, args.out_dir, args.frameskip, args.auto_discover)
            if s: results.append(s)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ! {os.path.basename(f)}: {type(e).__name__}: {e}")

    os.makedirs(args.out_dir, exist_ok=True)

    # manifest: one row per recording, for multi-participant analysis
    if results and not args.no_manifest:
        man = os.path.join(args.out_dir, "manifest.csv")
        fields = ["file", "participant", "game", "steps", "episodes", "total_return",
                  "entropy", "switch", "capture_hz", "flags"]
        with open(man, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for s in results:
                w.writerow({k: (round(s[k], 3) if isinstance(s.get(k), float) else s.get(k, "")) for k in fields})
        print(f"\nmanifest -> {man}")

    # combine: pool all aligned CSVs into one dataset (run_ts keeps sessions distinct)
    if results and args.combine:
        with open(args.combine, "w") as out_fh:
            wrote_header = False
            for s in results:
                with open(s["out"]) as in_fh:
                    header = next(in_fh)
                    if not wrote_header:
                        out_fh.write(header); wrote_header = True
                    for line in in_fh:
                        out_fh.write(line)
        print(f"combined dataset -> {args.combine}")

    print(f"\nDone. {len(results)} file(s) written to {args.out_dir}/")


if __name__ == "__main__":
    main()
