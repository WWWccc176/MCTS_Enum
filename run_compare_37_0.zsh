cd ~/DRL/MCTS_Enum

PYTHON_BIN="$(uv run python -c 'import sys; print(sys.executable)')"

./run.zsh Flash v0.1 \
  --reduction-level LLL \
  --node-budget 5000000 \
  --refresh-cycles 1 \
  Dataset/LLL/Basis/dim37_seed0.txt

./run.zsh Flash v0.2 \
  --reduction-level LLL \
  --node-budget 5000000 \
  --refresh-cycles 1 \
  --rollout-dimensions 10 \
  Dataset/LLL/Basis/dim37_seed0.txt
