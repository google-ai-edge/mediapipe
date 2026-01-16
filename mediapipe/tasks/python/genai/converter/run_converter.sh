#!/bin/bash
# Sample script to run the Python converter.
# Can be used with a checkpoint from HF such as
# https://huggingface.co/google/gemma-2b/tree/main

# Define paths and model parameters
MODEL_TYPE="GEMMA3_1B"
CKPT_DIR="/tmp/checkpoint"
VOCAB_FILE="${CKPT_DIR}/tokenizer.model"
OUTPUT_DIR="${CKPT_DIR}/converted"
OUTPUT_TFLITE="${OUTPUT_DIR}/model_gpu.tflite"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the Python converter
bazel run //third_party/odml/infra/genai/tools:converter_main \
  --config=gce \
  --define ENABLE_ODML_CONVERTER=1 \
  --nocheck_visibility -- \
  --input_ckpt="${CKPT_DIR}" \
  --ckpt_format="safetensors" \
  --backend="gpu" \
  --model_type="${MODEL_TYPE}" \
  --vocab_model_file="${VOCAB_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_tflite_file="${OUTPUT_TFLITE}"

echo "Conversion complete. Output TFLite file is at: ${OUTPUT_TFLITE}"
