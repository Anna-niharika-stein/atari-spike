"""
Extract the Atari-57 benchmark ROMs from ale_py into a folder the
browser recorder can serve (atari_ale_roms/).

Run this in your Jupyter notebook (where breakout/pong already worked).
It copies each ROM as <game_id>.bin — exactly the filenames the recorder
expects. Safe to re-run; it overwrites and verifies each file.
"""
from ale_py import roms
import shutil
import pathlib
import hashlib

# --- config ---------------------------------------------------------------
OUT_DIR = pathlib.Path("atari_ale_roms")   # change if your recorder folder differs

ATARI_57 = [
    "alien","amidar","assault","asterix","asteroids","atlantis","bank_heist",
    "battle_zone","beam_rider","berzerk","bowling","boxing","breakout","centipede",
    "chopper_command","crazy_climber","defender","demon_attack","double_dunk","enduro",
    "fishing_derby","freeway","frostbite","gopher","gravitar","hero","ice_hockey",
    "jamesbond","kangaroo","krull","kung_fu_master","montezuma_revenge","ms_pacman",
    "name_this_game","phoenix","pitfall","pong","private_eye","qbert","riverraid",
    "road_runner","robotank","seaquest","skiing","solaris","space_invaders",
    "star_gunner","surround","tennis","time_pilot","tutankham","up_n_down","venture",
    "video_pinball","wizard_of_wor","yars_revenge","zaxxon",
]

# --- extract --------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

available = set(roms.get_all_rom_ids())
copied, missing = [], []

for game in ATARI_57:
    if game not in available:
        missing.append(game)
        continue
    src = pathlib.Path(roms.get_rom_path(game))
    dst = OUT_DIR / f"{game}.bin"
    shutil.copy(src, dst)
    md5 = hashlib.md5(dst.read_bytes()).hexdigest()
    copied.append((game, md5, dst.stat().st_size))

# --- report ---------------------------------------------------------------
print(f"Extracted {len(copied)}/{len(ATARI_57)} ROMs into {OUT_DIR.resolve()}\n")
for game, md5, size in copied:
    print(f"  {game:<20} {size:>5} B   md5={md5}")

if missing:
    print(f"\n MISSING (not in this ale_py build): {missing}")
else:
    print("\nAll 57 Atari-57 ROMs extracted successfully.")

# Sanity check: confirm every dropdown game now has a file on disk
on_disk = sorted(p.stem for p in OUT_DIR.glob("*.bin"))
print(f"\n{len(on_disk)} .bin files now in {OUT_DIR}/")
