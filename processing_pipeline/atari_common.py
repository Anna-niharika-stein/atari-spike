#!/usr/bin/env python3
"""
atari_common.py  --  shared helpers used across the toolchain.

Keeping bcd / config-loading / session-reading in ONE place means the decode
logic can't drift between decode_ram.py, align_to_agent.py and run_pipeline.py.
"""
import json, os
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_config.json")


def bcd(v):
    """Two-nibble binary-coded-decimal: 0x17 -> 17."""
    v = np.asarray(v, dtype=np.int64)
    return (v >> 4) * 10 + (v & 0x0F)


def decode_byte(col, fmt):
    return bcd(col) if fmt == "bcd" else np.asarray(col, dtype=np.int64)


def looks_binary(vals):
    """A byte cannot be BCD if any observed value has a nibble in 0xA-0xF."""
    v = np.asarray(vals, dtype=np.int64)
    return bool(np.any((v >> 4) > 9) or np.any((v & 0x0F) > 9))


def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No game_config.json at {path}. Run discover_bytes.py first.")
    with open(path) as f:
        return json.load(f)


def load_config_or_empty(path=CONFIG_PATH):
    """Like load_config but returns {} if the file doesn't exist (for tools that create it)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def decode(ram, cfg):
    """Generic RAM decoder driven entirely by a config entry.

    SINGLE SOURCE OF TRUTH for score / reward / lives / terminal. Both
    decode_ram.py (validation + standalone decode) and align_to_agent.py
    (agent-schema conversion) import this, so the reward that goes into the
    agent-comparable CSVs is exactly the reward `decode_ram --validate` proves
    matches ALE. Do not reimplement reward logic anywhere else.

    Returns dict(score, lives, reward, terminal) as per-frame numpy arrays
    (lives is None when the game has no lives byte).
    """
    n = len(ram)
    sbs = cfg["score_bytes"]

    score = np.zeros(n, dtype=np.int64)
    for sb in sbs:
        score += sb.get("weight", 1) * sb.get("scale", 1) * decode_byte(ram[:, sb["byte"]], sb["format"])

    lv = cfg["lives"]
    mode = lv.get("mode", "none")
    if mode == "byte":
        lives = ram[:, lv["byte"]].astype(np.int64) - lv.get("offset", 0)
        terminal = lives <= 0
    elif mode == "terminal_value":
        # a marker byte equals a constant only at game over (validated per game)
        terminal = ram[:, lv["byte"]] == lv["value"]
        # If the marker value also appears on the pre-game attract screen, require
        # a non-marker value to have occurred first, so leading attract frames
        # aren't flagged terminal (e.g. Asterix byte83==0 at startup).
        if lv.get("require_nonzero_first"):
            non_marker = ram[:, lv["byte"]] != lv["value"]
            seen = np.cumsum(non_marker) > 0        # True from first non-marker on
            terminal = terminal & seen
        lives = None
    elif mode == "score_cap":
        cap = lv.get("cap", 21)
        terminal = np.zeros(n, bool)
        for sb in sbs:
            terminal |= decode_byte(ram[:, sb["byte"]], sb["format"]) >= cap
        lives = None
    else:                                  # no detectable terminal -> single episode
        terminal = np.zeros(n, bool)
        lives = None

    delta = np.zeros(n, dtype=np.int64)
    delta[:-1] = score[1:] - score[:-1]

    positive_only = all(sb.get("weight", 1) >= 0 for sb in sbs)
    single_bcd = (len(sbs) == 1 and sbs[0]["format"] == "bcd"
                  and sbs[0].get("weight", 1) > 0 and sbs[0].get("scale", 1) == 1)

    if single_bcd:                                   # BCD 99->0 wrap is really +1
        delta = np.where(delta < -50, delta + 100, delta)

    reward = delta.copy()

    # Zero reward only at TRUE episode resets. Detect a reset as the assembled
    # score collapsing toward zero (new game), not merely any decrease -- a
    # per-byte "big drop" test wrongly kills legitimate rewards when a low byte
    # rolls over into a higher place value (verified on seaquest/asterix).
    boundary = np.zeros(n, bool)
    if mode == "byte":
        col = ram[:, lv["byte"]].astype(np.int64)
        boundary[:-1] |= col[1:] > col[:-1]          # lives replenished = new game
    if positive_only:
        prev, nxt = score[:-1], score[1:]
        boundary[:-1] |= (nxt < prev) & (nxt <= np.maximum(prev * 0.5, 5))
    reward[boundary] = 0

    if positive_only:
        reward = np.where(reward < 0, 0, reward)

    return dict(score=score, lives=lives, reward=reward, terminal=terminal)


def read_session(path):
    """
    Read a raw browser session CSV.
    Returns (meta_dict, actions[int], ram[N,128] uint8, t_ms[float] or None).
    Metadata lines start with '#'.
    """
    meta, header, rows = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            if header is None:
                header = line.split(","); continue
            rows.append(line.split(","))
    if header is None or not rows:
        raise ValueError(f"No data rows in {path}")
    arr = np.array(rows)
    cols = {n: i for i, n in enumerate(header)}
    actions = arr[:, cols["action"]].astype(np.int64) if "action" in cols else None
    t_ms = arr[:, cols["t_ms"]].astype(float) if "t_ms" in cols else None
    ram = arr[:, [cols[f"ram_{i}"] for i in range(128)]].astype(np.uint8)
    return meta, actions, ram, t_ms
