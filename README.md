# Post-market monitoring under EU AI Act Chapter IX

Code for *Post-Market Monitoring of Adaptive AI Systems under EU AI Act Chapter IX*. Nineteen drift detectors watch a tabular Q-learning agent after deployment:

- **Maze** (environmental drift): 4x3 grid world, goal moves from C0 to C1-C4.
- **Tic-tac-toe** (adversarial drift): opponent switches from random (R0) to R1-R5.

## Files

| File | Contents |
|---|---|
| `envs.py` | grid world, tic-tac-toe, opponents R0-R5 |
| `agent.py` | Q-learning agent |
| `detectors.py` | all 13 [Frouros](https://github.com/IFCA-Advanced-Computing/frouros) concept drift detectors, Frouros KS and MMD, DAWIDD, edit-operation |
| `experiment.py` | settings, training, deployment, scoring |
| `run.py` | runs everything, writes the tables |

## Running

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps --ignore-requires-python frouros==0.9.0   # Python 3.13+ only
python run.py                   # 20 seeds, about 40 minutes
python run.py --seeds 2 --out quick   # quick check, leaves results/ alone
pytest
```

Output goes to `results/`: `tables.md` (Tables 3-6 of the paper) and per-seed alarms in `maze.json` / `tictactoe.json`. C0 and R0 are no-drift controls, used for the false alarm rates.

## Settings

Episode counts, grace window and alarm suppression are at the top of `experiment.py`; detector settings are in `detectors.py`.

Frouros detectors use their defaults and see one value per episode: the loss (return scaled to [0, 1] with the baseline range) or a 0/1 failure flag (goal not reached / game not won). Two exceptions: Geometric MA uses threshold 0.3 (the default 1.0 can't be reached on [0, 1]), and alarms on a constant stream are ignored (Frouros' STEPD divides 0 by 0 there).
