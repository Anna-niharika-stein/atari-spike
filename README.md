# atari-spike

**Human Atari Gameplay Recording, Processing, and Behavioural Analysis Pipeline**

This repository contains the full data collection and analysis infrastructure developed as part of the [GAMECHAR project](https://cordis.europa.eu/project/id/101220528).

The project compares human and reinforcement-learning agent gameplay across Atari 2600 games, extracting behavioural descriptors across three classes: performance, action structure, and RAM-based state visitation.

---

## Repository Structure

```
atari-spike/
├── human_gameplay_interface/           # Browser-based Atari game play recorder
├── raw_human_data_processing_pipeline/ # Raw human csv processing
└── analysis_pipeline/                  # Behavioural analysis and agent CSV generation
```

---

## Games

The recorder in `human_gameplay_interface` is operational for **44 out of 57** Atari benchmark games. The remaining 13 were excluded due to impractical controls, menu-based mechanics, or score signals that are not recoverable from RAM.

The analysis pipeline in `analysis_pipeline` is currently being validated for the selected six games, selected to span a range of complexity and difficulty for both humans and RL agents:

| Game | Genre | Human difficulty | Agent difficulty |
|---|---|---|---|
| Freeway | Navigation | Easy | Easy to Moderate |
| Skiing | Continuous Control | Easy | Hard |
| Breakout | Ball-and-Paddle | Easy to Moderate | Easy to Moderate |
| Space Invaders | Fixed Shooter | Moderate | Moderate |
| Asterix | Collection & Navigation | Moderate | Moderate to Hard |
| Seaquest | Resource Management & Shooter | Hard | Hard |

Validation status for all 57 games is tracked per game in `game_config.json`.

---

## human_gameplay_interface

A browser-based Atari gameplay recorder built on [Javatari.js](https://javatari.org/). Displays the live game and logs per-frame RAM state and player actions at ~60Hz in ALE's full 18-action space. Operational for 44 of the 57 Atari benchmark games.

**Files**
- `index.html` — open in a browser to play and record
- `javatari.js` — the Javatari emulator

**Output** — raw human session CSV (131 columns):
`frame, t_ms, action, ram_0..ram_127`

---

## raw_human_data_processing_pipeline

Converts raw human recordings into structured, agent-comparable aligned CSVs through a multi-step decoding process:

1. **Score decoding** — raw RAM bytes are decoded into a reward signal using per-game byte configurations in `game_config.json`. Score bytes are identified and validated via ALE agent rollouts (`discover_bytes.py`) or, for games a random agent cannot score in, via human session analysis (`discover_bytes_human.py`). Formats include binary and BCD (binary-coded decimal), with per-byte scale weights.
2. **Terminal detection** — episode boundaries are identified from RAM-based terminal signals (lives counters, terminal byte values) defined per game in `game_config.json`.
3. **Temporal alignment** — raw ~60Hz human frames are aggregated into non-overlapping 4-frame decision windows to match the agent's frameskip-4 action cadence. The modal action per window is taken as the representative action, with the first frame's RAM state as the pre-action snapshot.
4. **Post-terminal trimming** — frames recorded after the terminal signal are dropped to ensure clean episode boundaries.

**Files**
- `atari_common.py` — shared utilities (BCD decode, config loading, score assembly)
- `discover_bytes.py` — automated score-byte discovery via ALE agent rollouts
- `discover_bytes_human.py` — score-byte discovery from human sessions (for games a random agent cannot score in)
- `align_to_agent.py` — aligns raw human frames to agent 4-frame decision windows
- `run_pipeline.py` — end-to-end pipeline runner for a single game session
- `game_config.json` — master config for all 57 Atari games (score bytes, terminal detection, lives, action sets, validation status)

**Output** — aligned session CSV (265 columns):
`run_ts, episode, step, action, reward, done, episode_return, lives_pre, lives_post, ram_pre_0..127, ram_post_0..127`

---

## analysis_pipeline

Runs the full behavioural comparison analysis across all usable games and generates agent evaluation CSVs. Currently validated for the six games listed above.

**Files**
- `behavioural_pipeline.py` — unified analysis pipeline; computes all three descriptor classes (performance, action structure, RAM-state visitation) across all usable games in `game_config.json`. FIRE handling is auto-derived per game from its minimal action set and can be overridden in `game_config.json`.
- `generate_agent_csvs.ipynb` — generates 265-column agent CSVs for the six validation games using pretrained sb3 DQN models (Breakout, Space Invaders, Seaquest) and locally trained PPO agents (Freeway, Asterix, Skiing)

**Output per game:**
- Performance tables (episode return, reward density, first-reward latency)
- Action-structure tables (entropy, switching rate, run length, motif diversity)
- RAM-state visitation tables (exact uniqueness, Jensen-Shannon divergence, cluster distances)
- PCA and cluster heatmap figures
- Cross-game master summary table

---

## Data

Raw recordings, aligned CSVs, agent logs, and trained model checkpoints are not included in this repository. Data is stored locally and shared separately within the GAMECHAR project.

---

## Setup

```bash
git clone https://github.com/Anna-niharika-stein/atari-spike.git
cd atari-spike

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install ale-py gymnasium[atari,accept-rom-license] stable-baselines3[extra] huggingface-sb3 pandas numpy scikit-learn matplotlib
```

