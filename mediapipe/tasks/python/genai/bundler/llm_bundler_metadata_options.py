# Copyright 2026 The MediaPipe Authors.
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
"""LlmBundlerMetadataOptions Python dataclass."""

import ctypes
import dataclasses
from typing import Optional

from mediapipe.tasks.python.genai.bundler import llm_bundler_metadata_options_c


@dataclasses.dataclass
class LlmBundlerMetadataOptions:
  """Options for bundling LLM metadata.

  Attributes:
    start_token: Token that will be used to signify the beginning of a sequence.
    stop_tokens: Tokens that will be used to signify the end of a sequence.
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
  """

  start_token: str
  stop_tokens: list[str]
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

  def to_ctypes(
      self,
  ) -> llm_bundler_metadata_options_c.LlmBundlerMetadataOptionsC:
    """Converts the object to a ctypes structure."""
    stop_tokens_c = (ctypes.c_char_p * len(self.stop_tokens))()
    for i, token in enumerate(self.stop_tokens):
      stop_tokens_c[i] = token.encode("utf-8")
    start_token_c = self.start_token.encode("utf-8")
    system_prompt_c = (
        self.system_prompt.encode("utf-8") if self.system_prompt else None
    )
    prompt_prefix_user_c = (
        self.prompt_prefix_user.encode("utf-8")
        if self.prompt_prefix_user
        else None
    )
    prompt_suffix_user_c = (
        self.prompt_suffix_user.encode("utf-8")
        if self.prompt_suffix_user
        else None
    )
    prompt_prefix_model_c = (
        self.prompt_prefix_model.encode("utf-8")
        if self.prompt_prefix_model
        else None
    )
    prompt_suffix_model_c = (
        self.prompt_suffix_model.encode("utf-8")
        if self.prompt_suffix_model
        else None
    )
    prompt_prefix_system_c = (
        self.prompt_prefix_system.encode("utf-8")
        if self.prompt_prefix_system
        else None
    )
    prompt_suffix_system_c = (
        self.prompt_suffix_system.encode("utf-8")
        if self.prompt_suffix_system
        else None
    )
    user_role_token_c = (
        self.user_role_token.encode("utf-8") if self.user_role_token else None
    )
    system_role_token_c = (
        self.system_role_token.encode("utf-8")
        if self.system_role_token
        else None
    )
    model_role_token_c = (
        self.model_role_token.encode("utf-8") if self.model_role_token else None
    )
    end_role_token_c = (
        self.end_role_token.encode("utf-8") if self.end_role_token else None
    )
    return llm_bundler_metadata_options_c.LlmBundlerMetadataOptionsC(
        start_token=start_token_c,
        stop_tokens=stop_tokens_c,
        num_stop_tokens=len(self.stop_tokens),
        enable_bytes_to_unicode_mapping=self.enable_bytes_to_unicode_mapping,
        system_prompt=system_prompt_c,
        prompt_prefix_user=prompt_prefix_user_c,
        prompt_suffix_user=prompt_suffix_user_c,
        prompt_prefix_model=prompt_prefix_model_c,
        prompt_suffix_model=prompt_suffix_model_c,
        prompt_prefix_system=prompt_prefix_system_c,
        prompt_suffix_system=prompt_suffix_system_c,
        user_role_token=user_role_token_c,
        system_role_token=system_role_token_c,
        model_role_token=model_role_token_c,
        end_role_token=end_role_token_c,
    )
