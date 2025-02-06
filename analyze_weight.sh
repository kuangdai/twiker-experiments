for dir in data/datasets/*/; do
    name=$(basename "$dir")
    python analyze_weight.py -n "$name" -c 15
done