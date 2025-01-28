for file in result_weights/*.pt; do
    name=$(basename "$file" .pt)
    python analyze_weights.py -o 10 -n "$name"
done