for dir in data/datasets/*/; do
    name=$(basename "$dir")
    python analyze_pos.py -n "$name"
done