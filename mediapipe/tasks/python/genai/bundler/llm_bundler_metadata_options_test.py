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

from absl.testing import absltest
from mediapipe.tasks.python.genai.bundler import llm_bundler_metadata_options


class LlmBundlerMetadataOptionsTest(absltest.TestCase):

  def test_to_ctypes(self):
    options = llm_bundler_metadata_options.LlmBundlerMetadataOptions(
        start_token="start",
        stop_tokens=["stop1", "stop2"],
        enable_bytes_to_unicode_mapping=True,
        system_prompt="system",
        prompt_prefix_user="user_prefix",
        prompt_suffix_user="user_suffix",
        prompt_prefix_model="model_prefix",
        prompt_suffix_model="model_suffix",
        prompt_prefix_system=None,
        prompt_suffix_system=None,
        user_role_token="user_role",
        system_role_token=None,
        model_role_token="model_role",
        end_role_token="end_role",
    )

    options_c = options.to_ctypes()

    self.assertEqual(options_c.start_token, b"start")
    self.assertEqual(options_c.stop_tokens[0], b"stop1")
    self.assertEqual(options_c.stop_tokens[1], b"stop2")
    self.assertEqual(options_c.num_stop_tokens, 2)
    self.assertTrue(options_c.enable_bytes_to_unicode_mapping)
    self.assertEqual(options_c.system_prompt, b"system")
    self.assertEqual(options_c.prompt_prefix_user, b"user_prefix")
    self.assertEqual(options_c.prompt_suffix_user, b"user_suffix")
    self.assertEqual(options_c.prompt_prefix_model, b"model_prefix")
    self.assertEqual(options_c.prompt_suffix_model, b"model_suffix")
    self.assertIsNone(options_c.prompt_prefix_system)
    self.assertIsNone(options_c.prompt_suffix_system)
    self.assertEqual(options_c.user_role_token, b"user_role")
    self.assertIsNone(options_c.system_role_token)
    self.assertEqual(options_c.model_role_token, b"model_role")
    self.assertEqual(options_c.end_role_token, b"end_role")


if __name__ == "__main__":
  absltest.main()
