# CS6216 – Offline RL in Practice: code

Code for the Group 11a reading material and project video (CQL, Kumar et al. 2020; Levine et al. 2020).

```
cs6216_code/
├── data/
│   └── cql_table1.csv          # Kumar et al. (2020) Table 1, p. 8 (four-seed averages)
├── src/
│   ├── make_cql_gaps_figure.py # rebuilds figures/cql_wins_losses.pdf
│   └── tabular_cql_demo.py     # tabular demo: naive offline Q vs CQL(H) penalty
├── figures/                    # generated figures (.pdf + .png)
├── results/                    # generated JSON outputs
├── notebooks/                  # scratch / exploration
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run (from this folder)

```bash
python src/make_cql_gaps_figure.py          # -> figures/cql_wins_losses.pdf
python src/tabular_cql_demo.py --alpha 1.0  # -> results/tabular_cql_seed0.json
```

`make_cql_gaps_figure.py` also checks the numbers quoted in `sections/03_practice.tex`
(median gap −0.4 single-policy, +23.3 multi-policy; CQL behind on 6 of 18 datasets, by at most 3.4).

`tabular_cql_demo.py` is the "feasible course extension" from deck Part 3, slide 8. The
logged data never tries action *right* in state 2. Plain offline Q-iteration keeps that
unseen Q-value at its optimistic initial value and overestimates V(s0) (8.1 learned vs 0.0
true); the CQL(H) penalty pushes the unseen action down and the overestimation mostly goes away.

These scripts use only the published numbers and a toy MDP; nothing here reproduces the
D4RL experiments.
