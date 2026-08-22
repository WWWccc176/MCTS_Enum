SVP Challenge generator.

The official SVP Challenge instances require the pre-NTL-9.4 pseudorandom
generator. This project therefore builds generate_random against a private
NTL 9.3.0 installation at:

    mcts_env/ntl-9.3.0

Do not link this generator against the system NTL if the system version is
9.4 or newer. NTL 9.4 changed the PRG, so the same dimension/seed would
produce a different lattice.

Build:
    ./build.sh

Run:
    ./generate_random --dim 40 --seed 5778
