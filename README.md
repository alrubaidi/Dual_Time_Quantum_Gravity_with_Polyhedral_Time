# Dual-Time Quantum Gravity — Validation Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Tests](https://img.shields.io/badge/tests-43%2F43%20passing-brightgreen.svg)](#running-the-tests)

## Overview

This repository contains the **computational validation suite** accompanying a manuscript submitted. The manuscript itself is **not** included and has not been publicly distributed.

The code independently verifies the numerical results and mathematical consistency checks reported in the paper. Reviewers and readers can reproduce all computational claims by running the tests and scripts below.

## Repository Structure

```
validation_suite/
├── src/                        # Validation source modules
│   ├── constraint_algebra.py
│   ├── polyhedral_dynamics.py
│   ├── synchronization.py
│   ├── decoherence.py
│   └── cosmology.py
│
├── tests/                      # Unit tests (43/43 passing)
│   ├── test_constraints.py
│   ├── test_polyhedral.py
│   └── test_synchronization.py
│
├── notebooks/                  # Interactive Jupyter notebooks
├── outputs/figures/            # Generated figures
├── generate_figures.py         # Regenerate all figures
└── requirements.txt            # Python dependencies
```

## Running the Tests

```bash
cd validation_suite
pip install -r requirements.txt

# Run all 43 unit tests
python -m pytest tests/ -v

# Regenerate publication figures
python generate_figures.py

# Interactive exploration
jupyter notebook notebooks/
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

**Ali Al-Rubaidi**  
Independent Researcher, Sana'a, Yemen  
Email: ali.alrubaidi@gmail.com

---

*بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ*

*In the Name of Allah, the Most Gracious, the Most Merciful*
