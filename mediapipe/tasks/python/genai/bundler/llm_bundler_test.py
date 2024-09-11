# Copyright 2024 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for llm_bundler."""

import os
import zipfile

from absl.testing import absltest

from mediapipe.tasks.python.genai.bundler import llm_bundler


class LlmBundlerTest(absltest.TestCase):

  def _create_test_bundle(self, out_dir: str):
    """Helper function to create test bundle."""
    tflite_file_path = os.path.join(out_dir, "test.tflite")
    with open(tflite_file_path, "w") as f:
      f.write("tflite_model")
    sp_model_file_path = os.path.join(out_dir, "sp.model")
    with open(sp_model_file_path, "w") as f:
      f.write("sp_model")

    output_file = os.path.join(out_dir, "test.task")
    config = llm_bundler.BundleConfig(
        tflite_model=tflite_file_path,
        tokenizer_model=sp_model_file_path,
        start_token="BOS",
        stop_tokens=["EOS1", "EOS2"],
        output_filename=output_file,
        enable_bytes_to_unicode_mapping=True,
        prompt_prefix="<start_of_turn>user\n ",
        prompt_suffix="<end_of_turn>\n<start_of_turn>model\n"
    )
    llm_bundler.create_bundle(config)
    return output_file

  def test_can_create_bundle(self):
    tempdir = self.create_tempdir()
    output_file = self._create_test_bundle(tempdir.full_path)
    self.assertTrue(os.path.exists(output_file))

  def test_verify_content(self):
    tempdir = self.create_tempdir()
    output_file = self._create_test_bundle(tempdir.full_path)
    with zipfile.ZipFile(output_file) as zip_file:
      self.assertLen(zip_file.filelist, 3)
      self.assertEqual(zip_file.filelist[0].filename, "TF_LITE_PREFILL_DECODE")
      self.assertEqual(zip_file.filelist[1].filename, "TOKENIZER_MODEL")
      self.assertEqual(zip_file.filelist[2].filename, "METADATA")


if __name__ == "__main__":
  absltest.main()
