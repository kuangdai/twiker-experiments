mkdir -p glue_metrics/glue
curl -L \
  https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/glue/glue.py \
  -o glue_metrics/glue/glue.py
touch glue_metrics/glue/__init__.py