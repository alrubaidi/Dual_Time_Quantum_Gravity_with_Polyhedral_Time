# Validation Suite

Computational validation tools for a manuscript submitted to *Physical Review D* (accession DP13988).

## Validation Tracks

| Track | Script | Status |
|-------|--------|--------|
| A | `constraint_algebra.py` | ✅ |
| B | `polyhedral_dynamics.py` | ✅ |
| C | `synchronization.py` | ✅ |
| D | `decoherence.py` | ✅ |
| E | `cosmology.py` | ✅ |
| F | `decoherence.py` | ✅ |
| G | `cosmology.py` | ✅ |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run all tests
```bash
python -m pytest tests/ -v
```

### Regenerate figures
```bash
python generate_figures.py
```

### Jupyter notebooks
```bash
jupyter notebook notebooks/
```

## Output

Generated figures are saved to `outputs/figures/`.

## Dependencies

See `requirements.txt` for full list. Key packages:
- numpy, scipy: Numerical computation
- sympy: Symbolic algebra verification
- matplotlib: Visualization
- pytest: Testing framework
