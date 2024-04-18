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
from typing import List

from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils
from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2


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
    output_filename: Name of the generated `.task` file containg the Bundle.
  """

  tflite_model: str
  tokenizer_model: str
  start_token: str
  stop_tokens: List[str]
  output_filename: str


class _BundleTags(enum.Enum):
  """Bundle tags."""

  TF_LITE_PREFILL_DECODE = 1
  TOKENIZER_MODEL = 2
  METADATA = 3


def create_bundle(config: BundleConfig):
  """Creates a bundle from the given config."""
  artifacts = {}
  with open(config.tflite_model, "rb") as f:
    artifacts[_BundleTags.TF_LITE_PREFILL_DECODE.name] = f.read()

  with open(config.tokenizer_model, "rb") as f:
    artifacts[_BundleTags.TOKENIZER_MODEL.name] = f.read()

  params = llm_params_pb2.LlmParameters()
  params.start_token = config.start_token
  params.stop_tokens.extend(config.stop_tokens)
  artifacts[_BundleTags.METADATA.name] = params.SerializeToString()

  output_filename = config.output_filename
  if not output_filename.endswith(".task"):
    output_filename = config.output_filename + ".task"

  model_asset_bundle_utils.create_model_asset_bundle(
      artifacts,
      output_filename,
  )
