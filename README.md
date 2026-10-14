# Post-market monitoring under EU AI Act Chapter IX

Code for *Post-Market Monitoring of Adaptive AI Systems under EU AI Act Chapter IX*. 
- **Maze** (environmental drift): 4x3 grid world, goal moves from C0 to C1-C4.
- **Tic-tac-toe** (adversarial drift): opponent switches from random (R0) to R1-R5.

## Running

```bash
python -m venv .venv
.venv\Scripts\activate          
pip install -r requirements.txt
pip install --no-deps --ignore-requires-python frouros==0.9.0   # Python 3.13+ only
python run.py                   
python run.py --seeds 2 --out quick   
pytest
```

Output goes to `results/.
