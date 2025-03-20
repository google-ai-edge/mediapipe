# Copyright 2024 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to perform llm packing."""

import dataclasses
import enum
from typing import List, Optional

from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils
from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2
import sentencepiece


@dataclasses.dataclass(frozen=True)
class BundleConfig:
  """Config for LLM Bundler.

  Attributes:
    tflite_model: Path to the multi-signature tflite model with "prefill" and
      "decode" signatures converted using ODML Transformers APIs.
    tokenizer_model: Path to the tokenizer model. Currently only SentencePience
      tokenizer is supported. As such, tokenizer.model proto is expected to be
      passed here.
    start_token: Token that will be used to signify the beginning of a sequence.
    stop_tokens: Tokens that will be used to signify the end of a sequence.
    output_filename: Name of the generated `.task` file containing the Bundle.
    enable_bytes_to_unicode_mapping: Enables GPT-2 style bytes to unicode
      mapping. For more details see:
      https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    system_prompt: The starting text would be feed to the model on the start of
      each session, commonly called the System prompt. This is useful for
      instruction tuned models and pre-conditioning the model behavior.
    prompt_prefix: The prefix that will be added to each prompt passed to the
      model.
    prompt_suffix: The suffix that will be added to at the end of user prompt
      just before generating the response.
  """

  tflite_model: str
  tokenizer_model: str
  start_token: str
  stop_tokens: List[str]
  output_filename: str
  enable_bytes_to_unicode_mapping: bool = False
  system_prompt: Optional[str] = None
  prompt_prefix: Optional[str] = None
  prompt_suffix: Optional[str] = None


class _BundleTags(enum.Enum):
  """Bundle tags."""

  TF_LITE_PREFILL_DECODE = 1
  TOKENIZER_MODEL = 2
  METADATA = 3


def _validate_config(config: BundleConfig):
  """Validates the given config.

  Args:
    config: The config to validate.

  Raises:
    ValueError if the config is invalid. Invalid configs can be:
    - tokenizer_model is not a valid SentencePiece model.
    - start_token is not a valid token in the tokenizer_model.
    - stop_tokens is not a list of valid tokens in the tokenizer_model.
  """
  if not isinstance(config.stop_tokens, list):
    raise ValueError("stop_tokens must be a list of strings.")
  if not config.stop_tokens:
    raise ValueError("stop_tokens must be non-empty.")

  try:
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(config.tokenizer_model)
  except Exception as e:
    raise ValueError(
        f"Failed to load tokenizer model from {config.tokenizer_model}. "
        "Please ensure you are passing a valid SentencePiece model."
    ) from e


def create_bundle(config: BundleConfig):
  """Creates a bundle from the given config."""
  _validate_config(config)

  artifacts = {}
  with open(config.tflite_model, "rb") as f:
    artifacts[_BundleTags.TF_LITE_PREFILL_DECODE.name] = f.read()

  with open(config.tokenizer_model, "rb") as f:
    artifacts[_BundleTags.TOKENIZER_MODEL.name] = f.read()

  params = llm_params_pb2.LlmParameters()
  params.start_token = config.start_token
  params.stop_tokens.extend(config.stop_tokens)
  if config.enable_bytes_to_unicode_mapping:
    params.input_output_normalizations.append(
        llm_params_pb2.LlmParameters.INPUT_OUTPUT_NORMALIZATION_BYTES_TO_UNICODE
    )
  if config.system_prompt:
    params.prompt_template.session_prefix = config.system_prompt
  if config.prompt_prefix:
    params.prompt_template.prompt_prefix = config.prompt_prefix
  if config.prompt_suffix:
    params.prompt_template.prompt_suffix = config.prompt_suffix
  artifacts[_BundleTags.METADATA.name] = params.SerializeToString()

  output_filename = config.output_filename
  if not output_filename.endswith(".task"):
    output_filename = config.output_filename + ".task"

  model_asset_bundle_utils.create_model_asset_bundle(
      artifacts,
      output_filename,
  )
