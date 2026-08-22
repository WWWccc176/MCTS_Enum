# Flash v0.2

Flash v0.2 is the hybrid CPU successor to the v0.1 full-depth persistent MCTS baseline.

```text
MCTS upper-prefix scheduler
        -> fplll Schnorr-Euchner/KFP lower-subtree enumeration
        -> exact GMP candidate verification
        -> radius update / MCTS backup
```

Key rules:

- initial radius is the shortest row among all current basis rows;
- bottom subtree size is controlled by `--rollout-dimensions` (default 10);
- each CPU worker owns a private fplll enumeration context;
- fplll rollout runs outside the shared MCTS mutex;
- actions in the upper tree are generated lazily in SE order, with no full legal-interval sort and no 65536-action hard failure;
- `--node-budget` counts tree nodes + fplll enumeration nodes;
- every returned solution is checked with exact GMP `Bz` and squared norm;
- refresh retains a non-basis candidate independently from the incumbent;
- refresh is insert + LLL + zero-row removal + LLL + determinant/potential transaction gate;
- Flash v0.2 refresh does not run BKZ20;
- Flash v0.2 is CPU-only.

Build and run commands are in the repository root `README.md`.
