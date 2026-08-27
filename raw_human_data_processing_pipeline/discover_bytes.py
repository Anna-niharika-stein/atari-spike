#!/usr/bin/env python3
"""
discover_bytes.py  --  auto-discover and validate the RAM byte map for any ALE game.

For a new game, run this once:
    python discover_bytes.py breakout
    python discover_bytes.py ms_pacman
    python discover_bytes.py space_invaders

What it does (all validated against ALE's ground-truth reward/lives):
  * runs an agent and records RAM, reward, lives, game-over per frame
  * finds the LIVES byte (exact, or a constant offset e.g. 0-indexed storage)
  * ASSEMBLES the SCORE from one OR MORE bytes with place values (scales),
    solving for the combination whose per-step delta matches ALE reward exactly
  * detects the TERMINAL condition generically (lives->0, or a score cap read
    off the actual game-over frames, or none)
  * scores status with the REAL decoder so 'verified' can't lie

Adding a game requires no code changes here -- run this, then the rest of the
pipeline (decode_ram.py, align_to_agent.py, run_pipeline.py) just works.

Usage:
    python discover_bytes.py <game>
    python discover_bytes.py <game> --min-events 30 --max-frames 400000
    python discover_bytes.py <game> --show-config
    python discover_bytes.py <game> --dry-run
"""
import argparse, json, sys, os
import numpy as np
from atari_common import CONFIG_PATH, bcd, looks_binary, load_config_or_empty, save_config

try:
    from ale_py import ALEInterface, roms as ale_roms
except ImportError:
    sys.exit("ale-py not installed.  pip install ale-py")

VERIFY_THRESHOLD = 0.95   # exact reward-match to call a game 'verified'


# ── ALE runner ───────────────────────────────────────────────────────────────
def make_ale(game, seed=0):
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("color_averaging", False)
    try:
        ale.loadROM(str(ale_roms.get_rom_path(game)))
    except Exception:
        sys.exit(f"ROM not found for '{game}'. Check: python -c \"from ale_py import roms; print(roms.get_all_rom_ids())\"")
    return ale


def run_agent(game, seed, min_events=40, min_frames=20000, max_frames=400000):
    """
    One continuous ALE session. Records the game-over frame BEFORE resetting, so
    terminal states are observable. Uses sticky+FIRE-biased exploration (scores
    far more than uniform random in most games) unless a chase policy is known.

    Returns RAM, REW, LIVES, GO (game-over per frame), hit_cap.
    """
    ale = make_ale(game, seed)
    legal = [int(a.value) for a in ale.getLegalActionSet()]
    has_fire = 1 in legal
    rng = np.random.default_rng(seed)

    known_chase = {
        "breakout": lambda ram, f: 1 if f % 12 == 0 else (3 if ram[99] > ram[72] + 1 else (4 if ram[99] < ram[72] - 1 else 0)),
        "pong":     lambda ram, f: 1 if f % 30 == 0 else (3 if ram[51] > ram[54] + 1 else (4 if ram[51] < ram[54] - 1 else 0)),
    }
    chase = known_chase.get(game)

    RAM, REW, LIVES, GO = [], [], [], []
    events = 0
    last_act = int(rng.choice(legal))
    for f in range(max_frames):
        ram = ale.getRAM().copy()
        go = ale.game_over()
        RAM.append(ram); LIVES.append(ale.lives()); GO.append(go)
        if go:
            ale.reset_game()
            REW.append(0)
            last_act = int(rng.choice(legal))
            continue
        if chase:
            act = chase(ram, f)
        else:
            # sticky exploration + periodic FIRE so we actually start/score
            if rng.random() < 0.12:
                last_act = int(rng.choice(legal))
            act = 1 if (has_fire and f % 20 == 0) else last_act
        r = ale.act(act)
        REW.append(r)
        if r != 0:
            events += 1
        if events >= min_events and f >= min_frames:
            break

    n = len(RAM)
    return (np.array(RAM, dtype=np.int64), np.array(REW[:n], dtype=np.int64),
            np.array(LIVES, dtype=np.int64), np.array(GO, dtype=bool), events < min_events)


# ── format helpers ───────────────────────────────────────────────────────────
def valid_formats(vals):
    """Formats worth trying for a byte: binary always; bcd only if no nibble > 9."""
    return ["binary"] if looks_binary(vals) else ["bcd", "binary"]


def decode_col(col, fmt):
    return bcd(col) if fmt == "bcd" else col.astype(np.int64)


# ── multi-byte score assembly ────────────────────────────────────────────────
def assembled_delta_match(RAM, REW, GO, combo):
    """
    combo = list of (byte, fmt, scale). Score = sum(scale * decode(byte)).
    Compare its forward per-step delta to ALE reward, excluding reset transitions.
    Returns (match_rate, n_events).
    """
    n = len(RAM)
    S = np.zeros(n, dtype=np.int64)
    for (b, fmt, scale) in combo:
        S += scale * decode_col(RAM[:, b], fmt)
    d = np.zeros(n, dtype=np.int64)
    d[:-1] = S[1:] - S[:-1]
    valid = np.ones(n, bool)
    valid[:-1] &= ~GO[:-1]              # d[f] crosses a reset if frame f is game-over
    ev = ((REW != 0) | (d != 0)) & valid
    if ev.sum() == 0:
        return 0.0, 0
    return float((d[ev] == REW[ev]).mean()), int(ev.sum())


def score_candidates(RAM, REW, GO, max_bytes=12):
    """Bytes whose changes coincide with reward events (high purity), incl. rarely
    changing high-order bytes (low coverage but pure)."""
    rew_frames = set((np.nonzero(REW != 0)[0]).tolist())
    # a byte's change at frame f reflects reward at f-1
    cands = []
    for b in range(128):
        ch = np.nonzero(np.diff(RAM[:, b]) != 0)[0]        # change appears at ch+1
        if len(ch) < 2:
            continue
        aligned = set((ch).tolist())                        # reward[f] -> change at f+1 ~ ch=f
        purity = len(aligned & rew_frames) / len(aligned)
        if purity >= 0.35:
            cands.append((b, purity, len(ch)))
    cands.sort(key=lambda x: -x[1])
    return [b for b, _, _ in cands[:max_bytes]]


def assemble_score(RAM, REW, GO):
    """
    Search 1-, 2-, and 3-byte score assemblies with standard place-value scales,
    return the best (combo, rate). Every candidate is verified exactly vs ALE.
    """
    cands = score_candidates(RAM, REW, GO)
    if not cands:
        return None, 0.0
    best_combo, best_rate = None, -1.0

    def consider(combo):
        """Keep the best rate; on a tie prefer FEWER bytes (simplest explanation)."""
        nonlocal best_combo, best_rate
        rate, nev = assembled_delta_match(RAM, REW, GO, combo)
        if nev < 3:
            return
        if (rate > best_rate + 1e-9) or (abs(rate - best_rate) <= 1e-9 and
                                         best_combo is not None and len(combo) < len(best_combo)):
            best_combo, best_rate = combo, rate

    fmts = {b: valid_formats(RAM[:, b]) for b in cands}

    # 1 byte -- exhaustive over all 128 bytes. Cheap, and guarantees a strong
    # single-byte score can never be beaten by a weaker multi-byte fit.
    for b in range(128):
        for f in valid_formats(RAM[:, b]):
            consider([(b, f, 1)])
    if best_rate >= VERIFY_THRESHOLD:
        return best_combo, best_rate

    # 2 bytes: one low (scale 1) + one high (scale S)
    HIGH = [10, 100, 256, 1000, 10000, 65536]
    for lo in cands:
        for hi in cands:
            if hi == lo:
                continue
            for fl in fmts[lo]:
                for fh in fmts[hi]:
                    for S in HIGH:
                        consider([(lo, fl, 1), (hi, fh, S)])
        if best_rate >= VERIFY_THRESHOLD:
            break
    if best_rate >= VERIFY_THRESHOLD:
        return best_combo, best_rate

    # 3 bytes: ones + hundreds + ten-thousands (BCD) or 256^k (binary)
    TRIPLES = [(1, 100, 10000), (1, 256, 65536)]
    top = cands[:5]
    for i in range(len(top)):
        for j in range(len(top)):
            for k in range(len(top)):
                if len({i, j, k}) < 3:
                    continue
                a, b, c = top[i], top[j], top[k]
                for (sa, sb, sc) in TRIPLES:
                    fa = "bcd" if sc == 10000 and not looks_binary(RAM[:, a]) else valid_formats(RAM[:, a])[0]
                    fb = "bcd" if sc == 10000 and not looks_binary(RAM[:, b]) else valid_formats(RAM[:, b])[0]
                    fc = "bcd" if sc == 10000 and not looks_binary(RAM[:, c]) else valid_formats(RAM[:, c])[0]
                    consider([(a, fa, sa), (b, fb, sb), (c, fc, sc)])
        if best_rate >= VERIFY_THRESHOLD:
            break
    return best_combo, best_rate


# ── lives + terminal ─────────────────────────────────────────────────────────
def detect_lives(RAM, LIVES):
    """Exact byte, or byte == lives + constant offset. Returns (byte|None, offset)."""
    if len(np.unique(LIVES)) <= 1:
        return None, 0
    for b in range(128):
        if np.array_equal(RAM[:, b], LIVES):
            return b, 0
    for b in range(128):
        d = RAM[:, b] - LIVES
        if len(np.unique(d)) == 1:
            return b, int(d[0])
    return None, 0


def detect_terminal(RAM, GO, lives_byte, lives_offset, score_bytes):
    """Generic terminal: lives->0 if a lives byte exists; else a score cap read off
    the actual game-over frames; else none (single-episode)."""
    if lives_byte is not None:
        return {"mode": "byte", "byte": lives_byte, "offset": lives_offset}
    go = np.nonzero(GO)[0]
    if len(go) >= 3 and score_bytes:
        for sb in score_bytes:
            vals = decode_col(RAM[go, sb["byte"]], sb["format"])
            v, c = np.unique(vals, return_counts=True)
            if c.max() / len(vals) > 0.6 and v[c.argmax()] > 0:
                return {"mode": "score_cap", "cap": int(v[c.argmax()])}
    return {"mode": "none"}


# ── two-sided (pong-like) score selection ────────────────────────────────────
def two_sided_score(RAM, REW, GO):
    """
    For games where both sides score (pong, boxing), the reward is
    d(player) - d(opponent). Search candidate byte PAIRS and keep the pair whose
    exact per-step delta best matches ALE reward -- same verification standard
    as the single-sided assembly, no heuristics.
    """
    n = len(RAM)
    valid = np.ones(n, bool); valid[:-1] &= ~GO[:-1]

    # plausible counters: small range, monotone-ish increments
    cands = []
    for b in range(128):
        v = RAM[:, b]
        if v.min() < 0 or v.max() > 120:
            continue
        inc = int(np.sum(np.diff(v) > 0))
        if 3 <= inc <= max(50, int((REW != 0).sum()) * 4):
            cands.append(b)
    if len(cands) < 2:
        return None, 0.0

    best, best_rate = None, -1.0
    for p in cands:
        for o in cands:
            if p == o:
                continue
            for fp in valid_formats(RAM[:, p]):
                for fo in valid_formats(RAM[:, o]):
                    S = decode_col(RAM[:, p], fp) - decode_col(RAM[:, o], fo)
                    d = np.zeros(n, dtype=np.int64); d[:-1] = S[1:] - S[:-1]
                    ev = ((REW != 0) | (d != 0)) & valid
                    if ev.sum() < 5:
                        continue
                    rate = float((d[ev] == REW[ev]).mean())
                    if rate > best_rate:
                        best_rate = rate
                        best = [{"byte": p, "format": fp, "weight": 1, "scale": 1},
                                {"byte": o, "format": fo, "weight": -1, "scale": 1}]
    return best, max(best_rate, 0.0)


# ── discovery ────────────────────────────────────────────────────────────────
def discover(game, seed, verbose, min_events=40, max_frames=400000):
    if verbose:
        print(f"Collecting agent data for '{game}' (target {min_events} scoring events)…")
    RAM, REW, LIVES, GO, sparse = run_agent(game, seed, min_events=min_events, max_frames=max_frames)
    n_events = int((REW != 0).sum())
    if verbose:
        print(f"  {len(RAM)} frames, {n_events} scoring events"
              + ("  \u26a0 hit frame cap (low confidence)" if sparse else ""))
    if n_events == 0:
        if verbose:
            print("  no scoring events — needs a game-specific policy or a trained agent.")
        return None

    lives_byte, lives_off = detect_lives(RAM, LIVES)
    pos = int((REW > 0).sum()); neg = int((REW < 0).sum())

    # Always run the exactly-verified assembly search.
    combo, rate = assemble_score(RAM, REW, GO)
    score_bytes = ([{"byte": b, "format": f, "weight": 1, "scale": s} for (b, f, s) in combo]
                   if combo else None)

    # For two-sided games (pong-like) also try player-minus-opponent, and keep it
    # ONLY if it verifies better than the single-sided assembly.
    if neg > 5 and pos > 5:
        ts, ts_rate = two_sided_score(RAM, REW, GO)
        if ts and ts_rate > (rate or 0.0):
            score_bytes, rate = ts, ts_rate

    if not score_bytes:
        if verbose:
            print("  could not identify a score byte.")
        return None

    terminal = detect_terminal(RAM, GO, lives_byte, lives_off, score_bytes)

    if verbose:
        sb = ", ".join(f"byte{s['byte']}({s['format']}\u00d7{s.get('scale',1)}{'' if s.get('weight',1)>0 else ',opp'})"
                       for s in score_bytes)
        print(f"  score  : {sb}")
        print(f"  lives  : " + (f"byte {lives_byte} (offset {lives_off})" if lives_byte is not None else "none"))
        print(f"  terminal: {terminal['mode']}" + (f" (cap {terminal.get('cap')})" if terminal['mode']=='score_cap' else ""))

    return dict(score_bytes=score_bytes, lives=terminal, sparse=sparse,
                n_events=n_events, ram=RAM.astype("uint8"), rew=REW)


# ── config entry (status scored by the REAL decoder) ─────────────────────────
def build_config_entry(game, disc, threshold=VERIFY_THRESHOLD, seed_entry=None):
    from atari_common import decode as _decode
    score_bytes = disc["score_bytes"]
    lives_cfg = disc["lives"]

    # A previously-set terminal that was hand-validated (a lives byte, or a
    # validated terminal_value marker) is authoritative. Discovery's own terminal
    # detection is weaker and can be wrong (e.g. it may guess score_cap for a game
    # whose score keeps climbing). So if the existing entry has a byte/terminal_value
    # terminal, keep it — only take discovery's terminal when there's nothing to
    # preserve. (Score bytes from discovery are always used; only the terminal is
    # protected here.)
    if seed_entry:
        prev = seed_entry.get("lives", {})
        if prev.get("mode") in ("byte", "terminal_value"):
            lives_cfg = prev

    ram, rew = disc.get("ram"), disc.get("rew")
    match_rate = 0.0
    if ram is not None and rew is not None and len(ram):
        dec = _decode(ram, {"score_bytes": score_bytes, "lives": lives_cfg})
        ev = (rew != 0) | (dec["reward"] != 0)
        match_rate = float((dec["reward"][ev] == rew[ev]).mean()) if ev.any() else 0.0

    sparse = disc.get("sparse", False)
    conf = "  [sparse events — confirm on a longer/real recording]" if sparse else ""
    if match_rate >= threshold:
        status, note = "verified", f"{100*match_rate:.1f}% reward match vs ALE ({disc['n_events']} events){conf}"
    elif match_rate >= 0.5:
        status, note = "partial", f"partial match ({100*match_rate:.1f}%, {disc['n_events']} events) — review{conf}"
    else:
        status, note = "failed", f"low match ({100*match_rate:.1f}%) — custom score decode needed{conf}"

    entry = {
        "game": game,
        "score_bytes": score_bytes,
        "lives": lives_cfg,
        "minimal_action_set": [int(a.value) for a in make_ale(game).getMinimalActionSet()],
        "status": status,
        "note": note,
    }
    return entry, status


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", help="ALE game id, e.g. breakout, ms_pacman")
    ap.add_argument("--min-events", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=VERIFY_THRESHOLD)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-config", action="store_true")
    args = ap.parse_args()

    if args.show_config:
        cfg = load_config_or_empty()
        print(json.dumps(cfg, indent=2) if cfg else "(config is empty)")
        return
    if not args.game:
        ap.error("provide a game id (or --show-config)")

    disc = discover(args.game, args.seed, verbose=True,
                    min_events=args.min_events, max_frames=args.max_frames)
    if disc is None:
        sys.exit(1)

    _existing = load_config_or_empty().get(args.game)
    entry, status = build_config_entry(args.game, disc, args.threshold, seed_entry=_existing)
    print("\n" + "\u2500" * 56)
    print(f"result  : {status.upper()}")
    print(f"note    : {entry['note']}")
    print("\u2500" * 56)

    if status == "failed":
        print("Not saving — see note. (Often a multi-byte/pointer score needing a custom decoder.)")
        sys.exit(2)
    if not args.dry_run:
        cfg = load_config_or_empty()
        cfg[args.game] = entry
        save_config(cfg)
        print(f"Saved \u2192 {CONFIG_PATH}")
        print(f"Game '{args.game}' ready ({status}).")
    else:
        print("(dry-run: not saved)")
    sys.exit(0 if status == "verified" else 1)


if __name__ == "__main__":
    main()
