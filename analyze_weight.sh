for dir in data/datasets/*/; do
    name=$(basename "$dir")
    python analyze_weight.py -n "$name" -c 20
done

python analyze_weight.py -n victorian -c 2
python analyze_weight.py -n modern -c 10
python analyze_weight.py -n shakes -c 10
