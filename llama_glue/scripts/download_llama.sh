#!/bin/bash
mkdir -p ../models/llama-8b
cd ../models/llama-8b
echo "Downloading Meta‑Llama‑3‑8B via Huggingface…"
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B .
