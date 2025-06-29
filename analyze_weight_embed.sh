for file in weights/embed/*.pt; do
    name=$(basename "$file" .pt)
    python analyze_weight_embed.py -n "$name" -c 20
done