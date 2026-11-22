# CS6216 – Offline RL in Practice: code

Code for the Group 11a reading material and project video (CQL, Kumar et al. 2020; Levine et al. 2020).

```
code/
├── data/
│   └── cql_table1.csv          # Kumar et al. (2020) Table 1, p. 8 (four-seed averages)
├── src/
│   ├── make_cql_gaps_figure.py # rebuilds figures/cql_wins_losses.pdf
│   └── tabular_cql_demo.py     # tabular CQL demonstration -> Table 5
├── figures/                    # generated figures (.pdf + .png)
├── results/                    # generated outputs (JSON + LaTeX table)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run (from this folder)

```bash
python src/make_cql_gaps_figure.py   # -> figures/cql_wins_losses.pdf
python src/tabular_cql_demo.py       # seed 0 -> Table 5: results/table5_seed0.tex and results/tabular_cql_seed0.json
```

### `make_cql_gaps_figure.py`

Plots the gap between CQL(H) and the best of five baselines on the 18 D4RL gym datasets (the `cql_wins_losses` figure in §3).
It also checks the numbers quoted in §3 of the reading material: a median gap of −0.4 on
single-policy datasets and +23.3 on multi-policy datasets, and CQL behind on 6 of 18
datasets by at most 3.4 points.

### `tabular_cql_demo.py` (Table 5)

The tabular demonstration in §3 ("A tabular demonstration") and Appendix D. It needs only NumPy and runs in a few seconds.

- **MDP.** A chain of four states s0–s3. Every action moves one step to the right, and the
  episode ends after the action in s3, so every episode has four decisions. There are three
  actions per state and γ = 0.9: *usual* pays N(1.0, 0.5²), *rare-bad* pays N(−1.0, 3²) and
  *rare-good* pays N(1.3, 3²).
- **Behaviour policy.** 90/5/5: usual, rare-bad and rare-good with probabilities 0.90, 0.05 and
  0.05 in every state.
- **Data.** 1,000 independent datasets, each with 60 transitions per state (240 in all). Each
  rare action is seen about three times per state.
- **Fit.** Tabular Q-iteration with the optimality backup, for α ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}.
  For α > 0, each iteration minimises the CQL(H) penalty by gradient descent. α = 0 is naive
  offline Q-learning.
- **Evaluation.** The greedy policy of each fit is scored by its exact true return from s0,
  computed by backward recursion, so there are no rollouts or simulation noise. The reference
  returns are 4.47 for the optimal policy (always rare-good) and 3.44 for imitation (always usual).

**Seed 0 (the default) reproduces Table 5 exactly.** The script prints the table and writes
`results/table5_seed0.tex`, the LaTeX table body that goes into the paper. Use `--seed` and
`--datasets` to try other settings.

The learning rate and iteration counts (20 outer iterations × 100 gradient steps, lr 0.5) are
our own choices for this small problem. They do not follow the paper's deep-network settings,
and the results do not change with more iterations.

Nothing here reproduces the paper's D4RL experiments. Table 5 is teaching evidence only.
