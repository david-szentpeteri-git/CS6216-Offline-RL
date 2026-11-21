"""Small tabular demo of the course-extension idea (deck Part 3, slide 8):
naive offline Q-iteration trusts unseen actions; the CQL(H) penalty pushes them down.

Setup: a chain MDP with N states and 2 actions (0 = left, 1 = right).
Going right from the last state gives reward 1; everything else gives 0.
The logged dataset comes from a behaviour policy that almost never tries
action 1 in a few "gap" states, so those (s, a) pairs have little or no data.

Run:  python src/tabular_cql_demo.py [--alpha 1.0] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_STATES, N_ACTIONS, GAMMA = 6, 2, 0.9


def step(s, a):
    if a == 1:
        if s == N_STATES - 1:
            return 0, 1.0
        return s + 1, 0.0
    return max(s - 1, 0), 0.0


def true_q(policy):
    """Exact Q of a deterministic policy by iterating the Bellman operator."""
    q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(2000):
        new = np.zeros_like(q)
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                s2, r = step(s, a)
                new[s, a] = r + GAMMA * q[s2, policy[s2]]
        q = new
    return q


def collect(rng, n_transitions, gap_states):
    data = []
    s = 0
    for _ in range(n_transitions):
        p_right = 0.0 if s in gap_states else 0.7
        a = int(rng.random() < p_right)
        s2, r = step(s, a)
        data.append((s, a, r, s2))
        s = s2 if rng.random() > 0.1 else int(rng.integers(N_STATES))
    return data


def fit_q(data, alpha, iters=300, inner=50, lr=0.5, q_init=10.0):
    """Fitted Q-iteration on the dataset. alpha = 0 is plain offline Q-learning."""
    q = np.full((N_STATES, N_ACTIONS), q_init)
    counts = np.zeros((N_STATES, N_ACTIONS))
    for s, a, _, _ in data:
        counts[s, a] += 1
    state_n = counts.sum(1, keepdims=True)
    beta = np.divide(counts, state_n, out=np.zeros_like(counts), where=state_n > 0)
    seen_state = state_n > 0

    for _ in range(iters):
        target = np.zeros_like(q)
        for s, a, r, s2 in data:
            target[s, a] += r + GAMMA * q[s2].max()
        target = np.divide(target, counts, out=q.copy(), where=counts > 0)
        # Per state: minimise  E_{a~beta}[(Q - target)^2]/2 + alpha*(logsumexp Q - E_{a~beta} Q)
        for _ in range(inner):
            grad_bell = beta * (q - target)
            soft = np.exp(q - q.max(1, keepdims=True))
            soft /= soft.sum(1, keepdims=True)
            grad_cql = alpha * (soft - beta)
            q -= lr * (grad_bell + grad_cql) * seen_state
    return q, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data = collect(rng, args.n, gap_states={2})
    results = {}
    for name, alpha in [("naive", 0.0), ("cql", args.alpha)]:
        q, counts = fit_q(data, alpha)
        policy = q.argmax(1)
        qt = true_q(policy)
        results[name] = {
            "policy": policy.tolist(),
            "Q_learned_s0": float(q[0, policy[0]]),
            "Q_true_s0": float(qt[0, policy[0]]),
            "overestimation_s0": float(q[0, policy[0]] - qt[0, policy[0]]),
        }
        print(f"\n== {name} (alpha={alpha}) ==")
        print("counts per (s,a):\n", counts.astype(int))
        print("learned Q:\n", np.round(q, 3))
        print("greedy policy:", policy, " learned V(s0) =", round(q[0, policy[0]], 3),
              " true V(s0) =", round(qt[0, policy[0]], 3))

    out = ROOT / "results" / f"tabular_cql_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
