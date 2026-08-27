#!/usr/bin/env python3
"""
seed_config.py  --  pre-seed game_config.json with lives bytes taken directly
from ALE source (authoritative), so we don't rely on stochastic discovery to
find them — especially for sparse-reward games where a random agent rarely dies
enough for detect_lives() to work.

Two payoffs:
  1. The browser recorder reads these lives bytes from game_config.json to
     auto-stop each take exactly at game over (lives -> 0), matching the agent.
  2. discover_bytes.py can cross-check its own lives detection against them.

A seeded entry has the lives byte but NO score_bytes yet, so it is marked
status="lives_seeded". decode_ram / align_to_agent / run_pipeline treat that as
"not ready to decode" and ask you to run discover_bytes.py to fill score_bytes.
Re-running discovery MERGES: it keeps the seeded lives if its own detection is
uncertain.

Lives bytes below were extracted from
  Arcade-Learning-Environment/src/ale/games/supported/<Game>.cpp
via  m_lives = readRam(&system, 0x??)   ->   ram index = 0x?? & 0x7F.

Usage:
    python seed_config.py            # merge seeds into game_config.json
    python seed_config.py --show     # print what would be seeded
"""
import argparse, sys
from atari_common import load_config_or_empty, save_config, CONFIG_PATH

# game_id -> lives RAM index (already masked to 0..127)
LIVES_BYTES = {
    "assault": 101, "asterix": 83, "atlantis": 113, "bank_heist": 85,
    "battle_zone": 58, "centipede": 109, "chopper_command": 100,
    "crazy_climber": 42, "defender": 66, "gravitar": 4, "hero": 51,
    "name_this_game": 71, "phoenix": 75, "seaquest": 59, "space_invaders": 73,
}


def seed(show=False):
    cfg = load_config_or_empty()
    added, updated, skipped = [], [], []

    for game, byte in LIVES_BYTES.items():
        lives = {"mode": "byte", "byte": byte, "offset": 0}
        if game not in cfg:
            if not show:
                cfg[game] = {
                    "game": game,
                    "score_bytes": [],            # discover_bytes fills this
                    "lives": lives,
                    "minimal_action_set": [],
                    "status": "lives_seeded",
                    "note": f"lives byte {byte} seeded from ALE source; run discover_bytes for score",
                }
            added.append(game)
        else:
            e = cfg[game]
            # Only (re)write lives if it's missing or disagrees; never clobber a
            # fully-discovered entry's score_bytes/status.
            if e.get("lives", {}).get("mode") != "byte" or e["lives"].get("byte") != byte:
                if not show:
                    e["lives"] = lives
                updated.append(game)
            else:
                skipped.append(game)

    if show:
        print("Would seed lives bytes (game: ram_index):")
        for g in sorted(LIVES_BYTES):
            print(f"  {g:<18} ram[{LIVES_BYTES[g]}]")
        print(f"\nnew entries: {added}\nlives-updated: {updated}\nalready-correct: {skipped}")
        return

    save_config(cfg)
    print(f"Seeded {len(LIVES_BYTES)} lives bytes -> {CONFIG_PATH}")
    if added:   print(f"  created (lives-only) : {', '.join(sorted(added))}")
    if updated: print(f"  lives updated        : {', '.join(sorted(updated))}")
    if skipped: print(f"  already correct      : {', '.join(sorted(skipped))}")
    print("\nNext: run discover_bytes.py <game> (or run_pipeline --auto-discover) "
          "to fill score_bytes for the seeded games.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="print seeds without writing")
    args = ap.parse_args()
    seed(show=args.show)


if __name__ == "__main__":
    main()
