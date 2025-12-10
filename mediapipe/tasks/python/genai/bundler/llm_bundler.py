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

from mediapipe.tasks.python.genai.converter import external_dependencies
from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils
from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2

sentencepiece = external_dependencies.sentencepiece


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
      instruction tuned models and pre-conditioning the model behavior. This
      field is mutually exclusive with prompt_prefix_system and
      prompt_suffix_system.
    prompt_prefix_user: The prefix that should be prepended to each user portion
      of prompts passed to the model.
    prompt_suffix_user: The suffix that should be appended to each user portion
      of prompts passed to the model.
    prompt_prefix_model: The prefix that should be prepended to each model
      portion of prompts passed to the model.
    prompt_suffix_model: The suffix that should be appended to each model
      portion of prompts passsed to the model.
    prompt_prefix_system: The prefix that should be prepended to each system
      portion of prompts passed to the model.
    prompt_suffix_system: The suffix that should be appended to each system
      portion of prompts passsed to the model.
    user_role_token: The token that indicates the user's turn.
    system_role_token: The token that indicates the system's turn.
    model_role_token: The token that indicates the model's turn.
    end_role_token: The token that indicates the end of the prompt.
    tflite_embedder: Path to the embedding model if the embedding lookup is done
      externally.
    tflite_per_layer_embedder: Path to the per layer embedding model if the per
      layer embedding lookup is done externally.
    tflite_vision_encoder: Path to the vision encoder model.
    tflite_vision_adapter: Path to the vision adapter model.
  """

  tflite_model: str
  tokenizer_model: str
  start_token: str
  stop_tokens: List[str]
  output_filename: str
  enable_bytes_to_unicode_mapping: bool = False
  system_prompt: Optional[str] = None
  prompt_prefix_user: Optional[str] = None
  prompt_suffix_user: Optional[str] = None
  prompt_prefix_model: Optional[str] = None
  prompt_suffix_model: Optional[str] = None
  prompt_prefix_system: Optional[str] = None
  prompt_suffix_system: Optional[str] = None
  user_role_token: Optional[str] = None
  system_role_token: Optional[str] = None
  model_role_token: Optional[str] = None
  end_role_token: Optional[str] = None
  tflite_embedder: Optional[str] = None
  tflite_per_layer_embedder: Optional[str] = None
  tflite_vision_encoder: Optional[str] = None
  tflite_vision_adapter: Optional[str] = None


class _BundleTags(enum.Enum):
  """Bundle tags."""

  TF_LITE_PREFILL_DECODE = 1
  TOKENIZER_MODEL = 2
  METADATA = 3
  TF_LITE_EMBEDDER = 4
  TF_LITE_PER_LAYER_EMBEDDER = 5
  TF_LITE_VISION_ENCODER = 6
  TF_LITE_VISION_ADAPTER = 7


def _validate_config(config: BundleConfig):
  """Validates the given config.

  Args:
    config: The config to validate.

  Raises:
    ValueError if the config is invalid. Invalid configs can be:
    - tokenizer_model is not a valid SentencePiece model.
    - start_token is not a valid token in the tokenizer_model.
    - stop_tokens is not a list of valid tokens in the tokenizer_model.
    - system_prompt and prompt_*_system are both set.
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

  if config.system_prompt and (
      config.prompt_prefix_system or config.prompt_suffix_system
  ):
    raise ValueError(
        "system_prompt and prompt_*_system are mutually exclusive."
    )


def create_bundle(config: BundleConfig):
  """Creates a bundle from the given config."""
  _validate_config(config)

  artifacts = {}
  with open(config.tflite_model, "rb") as f:
    artifacts[_BundleTags.TF_LITE_PREFILL_DECODE.name] = f.read()

  with open(config.tokenizer_model, "rb") as f:
    artifacts[_BundleTags.TOKENIZER_MODEL.name] = f.read()

  params = create_metadata_pb(config)
  artifacts[_BundleTags.METADATA.name] = params.SerializeToString()

  if config.tflite_embedder:
    with open(config.tflite_embedder, "rb") as f:
      artifacts[_BundleTags.TF_LITE_EMBEDDER.name] = f.read()

  if config.tflite_per_layer_embedder:
    with open(config.tflite_per_layer_embedder, "rb") as f:
      artifacts[_BundleTags.TF_LITE_PER_LAYER_EMBEDDER.name] = f.read()

  if config.tflite_vision_encoder:
    with open(config.tflite_vision_encoder, "rb") as f:
      artifacts[_BundleTags.TF_LITE_VISION_ENCODER.name] = f.read()

  if config.tflite_vision_adapter:
    with open(config.tflite_vision_adapter, "rb") as f:
      artifacts[_BundleTags.TF_LITE_VISION_ADAPTER.name] = f.read()

  output_filename = config.output_filename
  if not output_filename.endswith(".task"):
    output_filename = config.output_filename + ".task"

  model_asset_bundle_utils.create_model_asset_bundle(
      artifacts,
      output_filename,
  )


def create_metadata_pb(config: BundleConfig):
  """Creates a Metadata proto from the given config."""
  params = llm_params_pb2.LlmParameters()
  params.start_token = config.start_token
  params.stop_tokens.extend(config.stop_tokens)
  if config.enable_bytes_to_unicode_mapping:
    params.input_output_normalizations.append(
        llm_params_pb2.LlmParameters.INPUT_OUTPUT_NORMALIZATION_BYTES_TO_UNICODE
    )
  if config.system_prompt:
    params.prompt_template.session_prefix = config.system_prompt
  else:
    if config.prompt_prefix_system:
      params.prompt_templates.system_template.prompt_prefix = (
          config.prompt_prefix_system
      )
    if config.prompt_suffix_system:
      params.prompt_templates.system_template.prompt_suffix = (
          config.prompt_suffix_system
      )
  if config.prompt_prefix_user:
    params.prompt_template.prompt_prefix = config.prompt_prefix_user
    params.prompt_templates.user_template.prompt_prefix = (
        config.prompt_prefix_user
    )
  if config.prompt_suffix_user:
    params.prompt_template.prompt_suffix = (
        config.prompt_suffix_user + config.prompt_prefix_model
    )
    params.prompt_templates.user_template.prompt_suffix = (
        config.prompt_suffix_user
    )
  if config.prompt_prefix_model:
    params.prompt_templates.model_template.prompt_prefix = (
        config.prompt_prefix_model
    )
  if config.prompt_suffix_model:
    params.prompt_templates.model_template.prompt_suffix = (
        config.prompt_suffix_model
    )
  if config.user_role_token:
    params.user_role_token = config.user_role_token
  if config.system_role_token:
    params.system_role_token = config.system_role_token
  if config.model_role_token:
    params.model_role_token = config.model_role_token
  if config.end_role_token:
    params.end_role_token = config.end_role_token
  return params
