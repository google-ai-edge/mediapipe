# Copyright 2022 The MediaPipe Authors.
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
# ==============================================================================
"""Tests for metadata info classes."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import flatbuffers

from mediapipe.tasks.metadata import metadata_schema_py_generated as _metadata_fb
from mediapipe.tasks.metadata import schema_py_generated as _schema_fb
from mediapipe.tasks.python.metadata import metadata as _metadata
from mediapipe.tasks.python.metadata.metadata_writers import metadata_info
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/tasks/testdata/metadata"
_SCORE_CALIBRATION_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "score_calibration.txt"))


class GeneralMdTest(absltest.TestCase):

  _EXPECTED_GENERAL_META_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "general_meta.json"))

  def test_create_metadata_should_succeed(self):
    general_md = metadata_info.GeneralMd(
        name="model",
        version="v1",
        description="A ML model.",
        author="MediaPipe",
        licenses="Apache")
    general_metadata = general_md.create_metadata()

    # Create the Flatbuffers object and convert it to the json format.
    builder = flatbuffers.Builder(0)
    builder.Finish(
        general_metadata.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_json = _metadata.convert_to_json(bytes(builder.Output()))

    with open(self._EXPECTED_GENERAL_META_JSON, "r") as f:
      expected_json = f.read()

    self.assertEqual(metadata_json, expected_json)


class AssociatedFileMdTest(absltest.TestCase):

  _EXPECTED_META_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "associated_file_meta.json"))

  def test_create_metadata_should_succeed(self):
    file_md = metadata_info.AssociatedFileMd(
        file_path="label.txt",
        description="The label file.",
        file_type=_metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS,
        locale="en")
    file_metadata = file_md.create_metadata()

    # Create the Flatbuffers object and convert it to the json format.
    model_metadata = _metadata_fb.ModelMetadataT()
    model_metadata.associatedFiles = [file_metadata]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_metadata.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_json = _metadata.convert_to_json(bytes(builder.Output()))

    with open(self._EXPECTED_META_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class TensorMdTest(parameterized.TestCase):

  _TENSOR_NAME = "input"
  _TENSOR_DESCRIPTION = "The input tensor."
  _TENSOR_MIN = 0
  _TENSOR_MAX = 1
  _LABEL_FILE_EN = "labels.txt"
  _LABEL_FILE_CN = "labels_cn.txt"  # Locale label file in Chinese.
  _EXPECTED_FEATURE_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "feature_tensor_meta.json"))
  _EXPECTED_IMAGE_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "image_tensor_meta.json"))
  _EXPECTED_BOUNDING_BOX_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "bounding_box_tensor_meta.json"))

  @parameterized.named_parameters(
      {
          "testcase_name": "feature_tensor",
          "content_type": _metadata_fb.ContentProperties.FeatureProperties,
          "golden_json": _EXPECTED_FEATURE_TENSOR_JSON
      }, {
          "testcase_name": "image_tensor",
          "content_type": _metadata_fb.ContentProperties.ImageProperties,
          "golden_json": _EXPECTED_IMAGE_TENSOR_JSON
      }, {
          "testcase_name": "bounding_box_tensor",
          "content_type": _metadata_fb.ContentProperties.BoundingBoxProperties,
          "golden_json": _EXPECTED_BOUNDING_BOX_TENSOR_JSON
      })
  def test_create_metadata_should_succeed(self, content_type, golden_json):
    associated_file1 = metadata_info.AssociatedFileMd(
        file_path=self._LABEL_FILE_EN, locale="en")
    associated_file2 = metadata_info.AssociatedFileMd(
        file_path=self._LABEL_FILE_CN, locale="cn")

    tensor_md = metadata_info.TensorMd(
        name=self._TENSOR_NAME,
        description=self._TENSOR_DESCRIPTION,
        min_values=[self._TENSOR_MIN],
        max_values=[self._TENSOR_MAX],
        content_type=content_type,
        associated_files=[associated_file1, associated_file2])
    tensor_metadata = tensor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(tensor_metadata))
    with open(golden_json, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class InputImageTensorMdTest(parameterized.TestCase):

  _NAME = "image"
  _DESCRIPTION = "The input image."
  _NORM_MEAN = (0, 127.5, 255)
  _NORM_STD = (127.5, 127.5, 127.5)
  _COLOR_SPACE_TYPE = _metadata_fb.ColorSpaceType.RGB
  _EXPECTED_FLOAT_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "input_image_tensor_float_meta.json"))
  _EXPECTED_UINT8_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "input_image_tensor_uint8_meta.json"))
  _EXPECTED_UNSUPPORTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "input_image_tensor_unsupported_meta.json"))

  @parameterized.named_parameters(
      {
          "testcase_name": "float",
          "tensor_type": _schema_fb.TensorType.FLOAT32,
          "golden_json": _EXPECTED_FLOAT_TENSOR_JSON
      }, {
          "testcase_name": "uint8",
          "tensor_type": _schema_fb.TensorType.UINT8,
          "golden_json": _EXPECTED_UINT8_TENSOR_JSON
      }, {
          "testcase_name": "unsupported_tensor_type",
          "tensor_type": _schema_fb.TensorType.INT16,
          "golden_json": _EXPECTED_UNSUPPORTED_TENSOR_JSON
      })
  def test_create_metadata_should_succeed(self, tensor_type, golden_json):
    tesnor_md = metadata_info.InputImageTensorMd(
        name=self._NAME,
        description=self._DESCRIPTION,
        norm_mean=list(self._NORM_MEAN),
        norm_std=list(self._NORM_STD),
        color_space_type=self._COLOR_SPACE_TYPE,
        tensor_type=tensor_type)
    tensor_metadata = tesnor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(tensor_metadata))
    with open(golden_json, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

  def test_init_should_throw_exception_with_incompatible_mean_and_std(self):
    norm_mean = [0]
    norm_std = [1, 2]
    with self.assertRaises(ValueError) as error:
      metadata_info.InputImageTensorMd(norm_mean=norm_mean, norm_std=norm_std)
    self.assertEqual(
        f"norm_mean and norm_std are expected to be the same dim. But got "
        f"{len(norm_mean)} and {len(norm_std)}", str(error.exception))


class InputTextTensorMdTest(absltest.TestCase):

  _NAME = "input text"
  _DESCRIPTION = "The input string."
  _VOCAB_FILE = "vocab.txt"
  _DELIM_REGEX_PATTERN = r"[^\w\']+"
  _EXPECTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "input_text_tensor_meta.json"))
  _EXPECTED_TENSOR_DEFAULT_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "input_text_tensor_default_meta.json"))

  def test_create_metadata_should_succeed(self):
    regex_tokenizer_md = metadata_info.RegexTokenizerMd(
        self._DELIM_REGEX_PATTERN, self._VOCAB_FILE)

    text_tensor_md = metadata_info.InputTextTensorMd(self._NAME,
                                                     self._DESCRIPTION,
                                                     regex_tokenizer_md)

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(
            text_tensor_md.create_metadata()))
    with open(self._EXPECTED_TENSOR_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

  def test_create_metadata_by_default_should_succeed(self):
    text_tensor_md = metadata_info.InputTextTensorMd()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(
            text_tensor_md.create_metadata()))
    with open(self._EXPECTED_TENSOR_DEFAULT_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class ClassificationTensorMdTest(parameterized.TestCase):

  _NAME = "probability"
  _DESCRIPTION = "The classification result tensor."
  _LABEL_FILE_EN = "labels.txt"
  _LABEL_FILE_CN = "labels_cn.txt"  # Locale label file in Chinese.
  _CALIBRATION_DEFAULT_SCORE = 0.2
  _EXPECTED_FLOAT_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "classification_tensor_float_meta.json"))
  _EXPECTED_UINT8_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "classification_tensor_uint8_meta.json"))
  _EXPECTED_UNSUPPORTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR,
                   "classification_tensor_unsupported_meta.json"))

  @parameterized.named_parameters(
      {
          "testcase_name": "float",
          "tensor_type": _schema_fb.TensorType.FLOAT32,
          "golden_json": _EXPECTED_FLOAT_TENSOR_JSON
      }, {
          "testcase_name": "uint8",
          "tensor_type": _schema_fb.TensorType.UINT8,
          "golden_json": _EXPECTED_UINT8_TENSOR_JSON
      }, {
          "testcase_name": "unsupported_tensor_type",
          "tensor_type": _schema_fb.TensorType.INT16,
          "golden_json": _EXPECTED_UNSUPPORTED_TENSOR_JSON
      })
  def test_create_metadata_should_succeed(self, tensor_type, golden_json):
    label_file_en = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_EN, locale="en")
    label_file_cn = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_CN, locale="cn")
    score_calibration_md = metadata_info.ScoreCalibrationMd(
        _metadata_fb.ScoreTransformationType.IDENTITY,
        self._CALIBRATION_DEFAULT_SCORE, _SCORE_CALIBRATION_FILE)

    tesnor_md = metadata_info.ClassificationTensorMd(
        name=self._NAME,
        description=self._DESCRIPTION,
        label_files=[label_file_en, label_file_cn],
        tensor_type=tensor_type,
        score_calibration_md=score_calibration_md)
    tensor_metadata = tesnor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(tensor_metadata))
    with open(golden_json, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class ScoreCalibrationMdTest(absltest.TestCase):
  _DEFAULT_VALUE = 0.2
  _EXPECTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "score_calibration_tensor_meta.json"))
  _EXPECTED_MODEL_META_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "score_calibration_file_meta.json"))

  def test_create_metadata_should_succeed(self):
    score_calibration_md = metadata_info.ScoreCalibrationMd(
        _metadata_fb.ScoreTransformationType.LOG, self._DEFAULT_VALUE,
        _SCORE_CALIBRATION_FILE)
    score_calibration_metadata = score_calibration_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_process_uint(
            score_calibration_metadata))
    with open(self._EXPECTED_TENSOR_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

  def test_create_score_calibration_file_md_should_succeed(self):
    score_calibration_md = metadata_info.ScoreCalibrationMd(
        _metadata_fb.ScoreTransformationType.LOG, self._DEFAULT_VALUE,
        _SCORE_CALIBRATION_FILE)
    score_calibration_file_md = (
        score_calibration_md.create_score_calibration_file_md())
    file_metadata = score_calibration_file_md.create_metadata()

    # Create the Flatbuffers object and convert it to the json format.
    model_metadata = _metadata_fb.ModelMetadataT()
    model_metadata.associatedFiles = [file_metadata]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_metadata.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_json = _metadata.convert_to_json(bytes(builder.Output()))

    with open(self._EXPECTED_MODEL_META_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

  def test_create_score_calibration_file_fails_with_less_colunms(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      malformed_calibration_file = test_utils.create_calibration_file(
          temp_dir, content="1.0,0.2")

      with self.assertRaisesRegex(
          ValueError,
          "Expected empty lines or 3 or 4 parameters per line in score" +
          " calibration file, but got 2."):
        metadata_info.ScoreCalibrationMd(
            _metadata_fb.ScoreTransformationType.LOG, self._DEFAULT_VALUE,
            malformed_calibration_file)

  def test_create_score_calibration_file_fails_with_negative_scale(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      malformed_calibration_file = test_utils.create_calibration_file(
          temp_dir, content="-1.0,0.2,0.1")

      with self.assertRaisesRegex(
          ValueError,
          "Expected scale to be a non-negative value, but got -1.0."):
        metadata_info.ScoreCalibrationMd(
            _metadata_fb.ScoreTransformationType.LOG, self._DEFAULT_VALUE,
            malformed_calibration_file)


class ScoreThresholdingMdTest(absltest.TestCase):
  _DEFAULT_GLOBAL_THRESHOLD = 0.5
  _EXPECTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "score_thresholding_meta.json"))

  def test_create_metadata_should_succeed(self):
    score_thresholding_md = metadata_info.ScoreThresholdingMd(
        global_score_threshold=self._DEFAULT_GLOBAL_THRESHOLD)

    score_thresholding_metadata = score_thresholding_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_process_uint(
            score_thresholding_metadata))
    with open(self._EXPECTED_TENSOR_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class BertTokenizerMdTest(absltest.TestCase):

  _VOCAB_FILE = "vocab.txt"
  _EXPECTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "bert_tokenizer_meta.json"))

  def test_create_metadata_should_succeed(self):
    tokenizer_md = metadata_info.BertTokenizerMd(self._VOCAB_FILE)
    tokenizer_metadata = tokenizer_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_process_uint(tokenizer_metadata))
    with open(self._EXPECTED_TENSOR_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class SentencePieceTokenizerMdTest(absltest.TestCase):

  _VOCAB_FILE = "vocab.txt"
  _SP_MODEL = "sp.model"
  _EXPECTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "sentence_piece_tokenizer_meta.json"))

  def test_create_metadata_should_succeed(self):
    tokenizer_md = metadata_info.SentencePieceTokenizerMd(
        self._SP_MODEL, self._VOCAB_FILE)
    tokenizer_metadata = tokenizer_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_process_uint(tokenizer_metadata))
    with open(self._EXPECTED_TENSOR_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class CategoryTensorMdTest(parameterized.TestCase, absltest.TestCase):
  _NAME = "category"
  _DESCRIPTION = "The category tensor."
  _LABEL_FILE_EN = "labels.txt"
  _LABEL_FILE_CN = "labels_cn.txt"  # Locale label file in Chinese.
  _EXPECTED_TENSOR_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "category_tensor_float_meta.json")
  )

  def test_create_metadata_should_succeed(self):
    label_file_en = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_EN, locale="en"
    )
    label_file_cn = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_CN, locale="cn"
    )
    tensor_md = metadata_info.CategoryTensorMd(
        name=self._NAME,
        description=self._DESCRIPTION,
        label_files=[label_file_en, label_file_cn],
    )
    tensor_metadata = tensor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(tensor_metadata)
    )
    with open(self._EXPECTED_TENSOR_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class TensorGroupMdMdTest(absltest.TestCase):
  _NAME = "detection_result"
  _TENSOR_NAMES = ["location", "category", "score"]
  _EXPECTED_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "tensor_group_meta.json")
  )

  def test_create_metadata_should_succeed(self):
    tensor_group_md = metadata_info.TensorGroupMd(
        name=self._NAME, tensor_names=self._TENSOR_NAMES
    )
    tensor_group_metadata = tensor_group_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor_group(tensor_group_metadata)
    )
    with open(self._EXPECTED_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


class SegmentationMaskMdTest(absltest.TestCase):
  _NAME = "segmentation_masks"
  _DESCRIPTION = "Masks over the target objects."
  _EXPECTED_JSON = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, "segmentation_mask_meta.json")
  )

  def test_create_metadata_should_succeed(self):
    segmentation_mask_md = metadata_info.SegmentationMaskMd(
        name=self._NAME, description=self._DESCRIPTION
    )
    metadata = segmentation_mask_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata_with_tensor(metadata)
    )
    with open(self._EXPECTED_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


def _create_dummy_model_metadata_with_tensor(
    tensor_metadata: _metadata_fb.TensorMetadataT) -> bytes:
  # Create a dummy model using the tensor metadata.
  subgraph_metadata = _metadata_fb.SubGraphMetadataT()
  subgraph_metadata.inputTensorMetadata = [tensor_metadata]
  model_metadata = _metadata_fb.ModelMetadataT()
  model_metadata.subgraphMetadata = [subgraph_metadata]

  # Create the Flatbuffers object and convert it to the json format.
  builder = flatbuffers.Builder(0)
  builder.Finish(
      model_metadata.Pack(builder),
      _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  return bytes(builder.Output())


def _create_dummy_model_metadata_with_process_uint(
    process_unit_metadata: _metadata_fb.ProcessUnitT) -> bytes:
  # Create a dummy model using the tensor metadata.
  subgraph_metadata = _metadata_fb.SubGraphMetadataT()
  subgraph_metadata.inputProcessUnits = [process_unit_metadata]
  model_metadata = _metadata_fb.ModelMetadataT()
  model_metadata.subgraphMetadata = [subgraph_metadata]

  # Create the Flatbuffers object and convert it to the json format.
  builder = flatbuffers.Builder(0)
  builder.Finish(
      model_metadata.Pack(builder),
      _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER,
  )
  return bytes(builder.Output())


def _create_dummy_model_metadata_with_tensor_group(
    tensor_group: _metadata_fb.TensorGroupT,
) -> bytes:
  # Creates a dummy model using the tensor group.
  subgraph_metadata = _metadata_fb.SubGraphMetadataT()
  subgraph_metadata.outputTensorGroups = [tensor_group]
  model_metadata = _metadata_fb.ModelMetadataT()
  model_metadata.subgraphMetadata = [subgraph_metadata]

  # Create the Flatbuffers object and convert it to the json format.
  builder = flatbuffers.Builder(0)
  builder.Finish(
      model_metadata.Pack(builder),
      _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  return bytes(builder.Output())


if __name__ == "__main__":
  absltest.main()
