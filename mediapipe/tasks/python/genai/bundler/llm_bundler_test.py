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

import os
import string
import zipfile

from absl.testing import absltest

from mediapipe.tasks.python.genai.bundler import llm_bundler
from sentencepiece import sentencepiece_model_pb2


class LlmBundlerTest(absltest.TestCase):

  BOS = "[BOS]"
  EOS = "[EOS]"

  def _create_sp_model(self, out_dir: str, corrupt: bool = False) -> str:
    """Helper function to create test SentencePiece model."""
    sp_file_path = os.path.join(out_dir, "sp.model")
    if corrupt:
      with open(sp_file_path, "w") as f:
        f.write("sp_model")
      return sp_file_path

    model = sentencepiece_model_pb2.ModelProto()
    # Add an unk token.
    model.pieces.add(
        piece="<unk>",
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.UNKNOWN,
    )
    # BOS and EOS are special control tokens.
    model.pieces.add(
        piece=self.BOS,
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.CONTROL,
    )
    model.pieces.add(
        piece=self.EOS,
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.CONTROL,
    )
    # Add a few user defined tokens.
    model.pieces.add(
        piece="[INST]",
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
    )
    model.pieces.add(
        piece="[/INST]",
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
    )
    model.pieces.add(
        piece="[SYS_S]",
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
    )
    model.pieces.add(
        piece="[SYS_E]",
        score=0.0,
        type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
    )
    # Add the rest of the alphabet as tokens
    for letter in string.ascii_lowercase + string.ascii_uppercase:
      model.pieces.add(
          piece=letter,
          score=0.0,
          type=sentencepiece_model_pb2.ModelProto.SentencePiece.NORMAL,
      )

    model.normalizer_spec.add_dummy_prefix = False

    with open(sp_file_path, "wb") as f:
      f.write(model.SerializeToString())
    return sp_file_path

  def _create_tflite_model(self, out_dir: str) -> str:
    """Helper function to create test tflite model."""
    tflite_file_path = os.path.join(out_dir, "test.tflite")
    with open(tflite_file_path, "w") as f:
      f.write("tflite_model")
    return tflite_file_path

  def test_can_create_bundle(self):
    tempdir = self.create_tempdir()
    sp_file_path = self._create_sp_model(tempdir.full_path)
    tflite_file_path = self._create_tflite_model(tempdir.full_path)
    output_file = os.path.join(tempdir, "test.task")
    config = llm_bundler.BundleConfig(
        tflite_model=tflite_file_path,
        tokenizer_model=sp_file_path,
        start_token=self.BOS,
        stop_tokens=[self.EOS],
        output_filename=output_file,
        enable_bytes_to_unicode_mapping=True,
        prompt_prefix="<start_of_turn>user\n ",
        prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
    )
    llm_bundler.create_bundle(config)
    self.assertTrue(os.path.exists(output_file))
    with zipfile.ZipFile(output_file) as zip_file:
      self.assertLen(zip_file.filelist, 3)
      self.assertEqual(zip_file.filelist[0].filename, "TF_LITE_PREFILL_DECODE")
      self.assertEqual(zip_file.filelist[1].filename, "TOKENIZER_MODEL")
      self.assertEqual(zip_file.filelist[2].filename, "METADATA")

  def test_invalid_stop_tokens_raises_value_error(self):
    tempdir = self.create_tempdir()
    sp_file_path = self._create_sp_model(tempdir.full_path)
    tflite_file_path = self._create_tflite_model(tempdir.full_path)
    output_file = os.path.join(tempdir, "test.task")
    config = llm_bundler.BundleConfig(
        tflite_model=tflite_file_path,
        tokenizer_model=sp_file_path,
        start_token=self.BOS,
        stop_tokens=self.EOS,
        output_filename=output_file,
    )
    with self.assertRaisesRegex(
        ValueError, "stop_tokens must be a list of strings"
    ):
      llm_bundler.create_bundle(config)

    config = llm_bundler.BundleConfig(
        tflite_model=tflite_file_path,
        tokenizer_model=sp_file_path,
        start_token=self.BOS,
        stop_tokens=[],
        output_filename=output_file,
    )
    with self.assertRaisesRegex(ValueError, "stop_tokens must be non-empty"):
      llm_bundler.create_bundle(config)

  def test_invalid_tokenizer_model_raises_value_error(self):
    tempdir = self.create_tempdir()
    sp_file_path = self._create_sp_model(tempdir.full_path, corrupt=True)
    tflite_file_path = self._create_tflite_model(tempdir.full_path)
    output_file = os.path.join(tempdir, "test.task")
    config = llm_bundler.BundleConfig(
        tflite_model=tflite_file_path,
        tokenizer_model=sp_file_path,
        start_token=self.BOS,
        stop_tokens=[self.EOS],
        output_filename=output_file,
    )
    with self.assertRaisesRegex(
        ValueError,
        "Failed to load tokenizer model from",
    ):
      llm_bundler.create_bundle(config)


if __name__ == "__main__":
  absltest.main()
