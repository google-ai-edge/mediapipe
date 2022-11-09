# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Helper classes for common model metadata information."""

import csv
import os
from typing import List, Optional, Type

from mediapipe.tasks.metadata import metadata_schema_py_generated as _metadata_fb
from mediapipe.tasks.metadata import schema_py_generated as _schema_fb

# Min and max values for UINT8 tensors.
_MIN_UINT8 = 0
_MAX_UINT8 = 255

# Default description for vocabulary files.
_VOCAB_FILE_DESCRIPTION = ("Vocabulary file to convert natural language "
                           "words to embedding vectors.")


class GeneralMd:
  """A container for common metadata information of a model.

  Attributes:
    name: name of the model.
    version: version of the model.
    description: description of what the model does.
    author: author of the model.
    licenses: licenses of the model.
  """

  def __init__(self,
               name: Optional[str] = None,
               version: Optional[str] = None,
               description: Optional[str] = None,
               author: Optional[str] = None,
               licenses: Optional[str] = None) -> None:
    self.name = name
    self.version = version
    self.description = description
    self.author = author
    self.licenses = licenses

  def create_metadata(self) -> _metadata_fb.ModelMetadataT:
    """Creates the model metadata based on the general model information.

    Returns:
      A Flatbuffers Python object of the model metadata.
    """
    model_metadata = _metadata_fb.ModelMetadataT()
    model_metadata.name = self.name
    model_metadata.version = self.version
    model_metadata.description = self.description
    model_metadata.author = self.author
    model_metadata.license = self.licenses
    return model_metadata


class AssociatedFileMd:
  """A container for common associated file metadata information.

  Attributes:
    file_path: path to the associated file.
    description: description of the associated file.
    file_type: file type of the associated file [1].
    locale: locale of the associated file [2].
    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L77
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L176
  """

  def __init__(
      self,
      file_path: str,
      description: Optional[str] = None,
      file_type: Optional[int] = _metadata_fb.AssociatedFileType.UNKNOWN,
      locale: Optional[str] = None) -> None:
    self.file_path = file_path
    self.description = description
    self.file_type = file_type
    self.locale = locale

  def create_metadata(self) -> _metadata_fb.AssociatedFileT:
    """Creates the associated file metadata.

    Returns:
      A Flatbuffers Python object of the associated file metadata.
    """
    file_metadata = _metadata_fb.AssociatedFileT()
    file_metadata.name = os.path.basename(self.file_path)
    file_metadata.description = self.description
    file_metadata.type = self.file_type
    file_metadata.locale = self.locale
    return file_metadata


class LabelFileMd(AssociatedFileMd):
  """A container for label file metadata information."""

  _LABEL_FILE_DESCRIPTION = ("Labels for categories that the model can "
                             "recognize.")
  _FILE_TYPE = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS

  def __init__(self, file_path: str, locale: Optional[str] = None) -> None:
    """Creates a LabelFileMd object.

    Args:
      file_path: file_path of the label file.
      locale: locale of the label file [1].
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L176
    """
    super().__init__(file_path, self._LABEL_FILE_DESCRIPTION, self._FILE_TYPE,
                     locale)


class ScoreCalibrationMd:
  """A container for score calibration [1] metadata information.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L456
  """

  _SCORE_CALIBRATION_FILE_DESCRIPTION = (
      "Contains sigmoid-based score calibration parameters. The main purposes "
      "of score calibration is to make scores across classes comparable, so "
      "that a common threshold can be used for all output classes.")
  _FILE_TYPE = _metadata_fb.AssociatedFileType.TENSOR_AXIS_SCORE_CALIBRATION

  def __init__(self,
               score_transformation_type: _metadata_fb.ScoreTransformationType,
               default_score: float, file_path: str) -> None:
    """Creates a ScoreCalibrationMd object.

    Args:
      score_transformation_type: type of the function used for transforming the
        uncalibrated score before applying score calibration.
      default_score: the default calibrated score to apply if the uncalibrated
        score is below min_score or if no parameters were specified for a given
        index.
      file_path: file_path of the score calibration file [1].
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L133

    Raises:
      ValueError: if the score_calibration file is malformed.
    """
    self._score_transformation_type = score_transformation_type
    self._default_score = default_score
    self._file_path = file_path

    # Sanity check the score calibration file.
    with open(self._file_path) as calibration_file:
      csv_reader = csv.reader(calibration_file, delimiter=",")
      for row in csv_reader:
        if row and len(row) != 3 and len(row) != 4:
          raise ValueError(
              f"Expected empty lines or 3 or 4 parameters per line in score"
              f" calibration file, but got {len(row)}.")

        if row and float(row[0]) < 0:
          raise ValueError(
              f"Expected scale to be a non-negative value, but got "
              f"{float(row[0])}.")

  def create_metadata(self) -> _metadata_fb.ProcessUnitT:
    """Creates the score calibration metadata based on the information.

    Returns:
      A Flatbuffers Python object of the score calibration metadata.
    """
    score_calibration = _metadata_fb.ProcessUnitT()
    score_calibration.optionsType = (
        _metadata_fb.ProcessUnitOptions.ScoreCalibrationOptions)
    options = _metadata_fb.ScoreCalibrationOptionsT()
    options.scoreTransformation = self._score_transformation_type
    options.defaultScore = self._default_score
    score_calibration.options = options
    return score_calibration

  def create_score_calibration_file_md(self) -> AssociatedFileMd:
    return AssociatedFileMd(self._file_path,
                            self._SCORE_CALIBRATION_FILE_DESCRIPTION,
                            self._FILE_TYPE)


class ScoreThresholdingMd:
  """A container for score thresholding [1] metadata information.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L468
  """

  def __init__(self, global_score_threshold: float) -> None:
    """Creates a ScoreThresholdingMd object.

    Args:
      global_score_threshold: The recommended global threshold below which
        results are considered low-confidence and should be filtered out.
    """
    self._global_score_threshold = global_score_threshold

  def create_metadata(self) -> _metadata_fb.ProcessUnitT:
    """Creates the score thresholding metadata based on the information.

    Returns:
      A Flatbuffers Python object of the score thresholding metadata.
    """
    score_thresholding = _metadata_fb.ProcessUnitT()
    score_thresholding.optionsType = (
        _metadata_fb.ProcessUnitOptions.ScoreThresholdingOptions)
    options = _metadata_fb.ScoreThresholdingOptionsT()
    options.globalScoreThreshold = self._global_score_threshold
    score_thresholding.options = options
    return score_thresholding


class RegexTokenizerMd:
  """A container for the Regex tokenizer [1] metadata information.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L500
  """

  def __init__(self, delim_regex_pattern: str, vocab_file_path: str):
    """Initializes a RegexTokenizerMd object.

    Args:
      delim_regex_pattern: the regular expression to segment strings and create
        tokens.
      vocab_file_path: path to the vocabulary file.
    """
    self._delim_regex_pattern = delim_regex_pattern
    self._vocab_file_path = vocab_file_path

  def create_metadata(self) -> _metadata_fb.ProcessUnitT:
    """Creates the Regex tokenizer metadata based on the information.

    Returns:
      A Flatbuffers Python object of the Regex tokenizer metadata.
    """
    vocab = _metadata_fb.AssociatedFileT()
    vocab.name = self._vocab_file_path
    vocab.description = _VOCAB_FILE_DESCRIPTION
    vocab.type = _metadata_fb.AssociatedFileType.VOCABULARY

    # Create the RegexTokenizer.
    tokenizer = _metadata_fb.ProcessUnitT()
    tokenizer.optionsType = (
        _metadata_fb.ProcessUnitOptions.RegexTokenizerOptions)
    tokenizer.options = _metadata_fb.RegexTokenizerOptionsT()
    tokenizer.options.delimRegexPattern = self._delim_regex_pattern
    tokenizer.options.vocabFile = [vocab]
    return tokenizer


class TensorMd:
  """A container for common tensor metadata information.

  Attributes:
    name: name of the tensor.
    description: description of what the tensor is.
    min_values: per-channel minimum value of the tensor.
    max_values: per-channel maximum value of the tensor.
    content_type: content_type of the tensor.
    associated_files: information of the associated files in the tensor.
    tensor_name: name of the corresponding tensor [1] in the TFLite model. It is
      used to locate the corresponding tensor and decide the order of the tensor
      metadata [2] when populating model metadata.
    [1]:
      https://github.com/tensorflow/tensorflow/blob/cb67fef35567298b40ac166b0581cd8ad68e5a3a/tensorflow/lite/schema/schema.fbs#L1129-L1136
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L623-L640
  """

  def __init__(
      self,
      name: Optional[str] = None,
      description: Optional[str] = None,
      min_values: Optional[List[float]] = None,
      max_values: Optional[List[float]] = None,
      content_type: int = _metadata_fb.ContentProperties.FeatureProperties,
      associated_files: Optional[List[Type[AssociatedFileMd]]] = None,
      tensor_name: Optional[str] = None) -> None:
    self.name = name
    self.description = description
    self.min_values = min_values
    self.max_values = max_values
    self.content_type = content_type
    self.associated_files = associated_files
    self.tensor_name = tensor_name

  def create_metadata(self) -> _metadata_fb.TensorMetadataT:
    """Creates the input tensor metadata based on the information.

    Returns:
      A Flatbuffers Python object of the input metadata.
    """
    tensor_metadata = _metadata_fb.TensorMetadataT()
    tensor_metadata.name = self.name
    tensor_metadata.description = self.description

    # Create min and max values
    stats = _metadata_fb.StatsT()
    stats.max = self.max_values
    stats.min = self.min_values
    tensor_metadata.stats = stats

    # Create content properties
    content = _metadata_fb.ContentT()
    if self.content_type is _metadata_fb.ContentProperties.FeatureProperties:
      content.contentProperties = _metadata_fb.FeaturePropertiesT()
    elif self.content_type is _metadata_fb.ContentProperties.ImageProperties:
      content.contentProperties = _metadata_fb.ImagePropertiesT()
    elif self.content_type is (
        _metadata_fb.ContentProperties.BoundingBoxProperties):
      content.contentProperties = _metadata_fb.BoundingBoxPropertiesT()
    elif self.content_type is _metadata_fb.ContentProperties.AudioProperties:
      content.contentProperties = _metadata_fb.AudioPropertiesT()

    content.contentPropertiesType = self.content_type
    tensor_metadata.content = content

    # TODO: check if multiple label files have populated locale.
    # Create associated files
    if self.associated_files:
      tensor_metadata.associatedFiles = [
          file.create_metadata() for file in self.associated_files
      ]
    return tensor_metadata


class InputImageTensorMd(TensorMd):
  """A container for input image tensor metadata information.

  Attributes:
    norm_mean: the mean value used in tensor normalization [1].
    norm_std: the std value used in the tensor normalization [1]. norm_mean and
      norm_std must have the same dimension.
    color_space_type: the color space type of the input image [2].
    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L198
  """

  # Min and max float values for image pixels.
  _MIN_PIXEL = 0.0
  _MAX_PIXEL = 255.0

  def __init__(
      self,
      name: Optional[str] = None,
      description: Optional[str] = None,
      norm_mean: Optional[List[float]] = None,
      norm_std: Optional[List[float]] = None,
      color_space_type: Optional[int] = _metadata_fb.ColorSpaceType.UNKNOWN,
      tensor_type: Optional["_schema_fb.TensorType"] = None) -> None:
    """Initializes the instance of InputImageTensorMd.

    Args:
      name: name of the tensor.
      description: description of what the tensor is.
      norm_mean: the mean value used in tensor normalization [1].
      norm_std: the std value used in the tensor normalization [1]. norm_mean
        and norm_std must have the same dimension.
      color_space_type: the color space type of the input image [2].
      tensor_type: data type of the tensor.
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L198

    Raises:
      ValueError: if norm_mean and norm_std have different dimensions.
    """
    if norm_std and norm_mean and len(norm_std) != len(norm_mean):
      raise ValueError(
          f"norm_mean and norm_std are expected to be the same dim. But got "
          f"{len(norm_mean)} and {len(norm_std)}")

    if tensor_type is _schema_fb.TensorType.UINT8:
      min_values = [_MIN_UINT8]
      max_values = [_MAX_UINT8]
    elif tensor_type is _schema_fb.TensorType.FLOAT32 and norm_std and norm_mean:
      min_values = [
          float(self._MIN_PIXEL - mean) / std
          for mean, std in zip(norm_mean, norm_std)
      ]
      max_values = [
          float(self._MAX_PIXEL - mean) / std
          for mean, std in zip(norm_mean, norm_std)
      ]
    else:
      # Uint8 and Float32 are the two major types currently. And Task library
      # doesn't support other types so far.
      min_values = None
      max_values = None

    super().__init__(name, description, min_values, max_values,
                     _metadata_fb.ContentProperties.ImageProperties)
    self.norm_mean = norm_mean
    self.norm_std = norm_std
    self.color_space_type = color_space_type

  def create_metadata(self) -> _metadata_fb.TensorMetadataT:
    """Creates the input image metadata based on the information.

    Returns:
      A Flatbuffers Python object of the input image metadata.
    """
    tensor_metadata = super().create_metadata()
    tensor_metadata.content.contentProperties.colorSpace = self.color_space_type
    # Create normalization parameters
    if self.norm_mean and self.norm_std:
      normalization = _metadata_fb.ProcessUnitT()
      normalization.optionsType = (
          _metadata_fb.ProcessUnitOptions.NormalizationOptions)
      normalization.options = _metadata_fb.NormalizationOptionsT()
      normalization.options.mean = self.norm_mean
      normalization.options.std = self.norm_std
      tensor_metadata.processUnits = [normalization]
    return tensor_metadata


class InputTextTensorMd(TensorMd):
  """A container for the input text tensor metadata information.

  Attributes:
    tokenizer_md: information of the tokenizer in the input text tensor, if any.
  """

  def __init__(self,
               name: Optional[str] = None,
               description: Optional[str] = None,
               tokenizer_md: Optional[RegexTokenizerMd] = None):
    """Initializes the instance of InputTextTensorMd.

    Args:
      name: name of the tensor.
      description: description of what the tensor is.
      tokenizer_md: information of the tokenizer in the input text tensor, if
        any. Only `RegexTokenizer` [1] is currenly supported. If the tokenizer
        is `BertTokenizer` [2] or `SentencePieceTokenizer` [3], refer to
        `BertInputTensorsMd` class.
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L500
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L477
      [3]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L485
    """
    super().__init__(name, description)
    self.tokenizer_md = tokenizer_md

  def create_metadata(self) -> _metadata_fb.TensorMetadataT:
    """Creates the input text metadata based on the information.

    Returns:
      A Flatbuffers Python object of the input text metadata.

    Raises:
      ValueError: if the type of tokenizer_md is unsupported.
    """
    if not isinstance(self.tokenizer_md, (type(None), RegexTokenizerMd)):
      raise ValueError(
          f"The type of tokenizer_options, {type(self.tokenizer_md)}, is "
          f"unsupported")

    tensor_metadata = super().create_metadata()
    if self.tokenizer_md:
      tensor_metadata.processUnits = [self.tokenizer_md.create_metadata()]
    return tensor_metadata


class ClassificationTensorMd(TensorMd):
  """A container for the classification tensor metadata information.

  Attributes:
    label_files: information of the label files [1] in the classification
      tensor.
    score_calibration_md: information of the score calibration operation [2] in
      the classification tensor.
    score_thresholding_md: information of the score thresholding [3] in the
        classification tensor.
    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L456
    [3]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L468
  """

  # Min and max float values for classification results.
  _MIN_FLOAT = 0.0
  _MAX_FLOAT = 1.0

  def __init__(
      self,
      name: Optional[str] = None,
      description: Optional[str] = None,
      label_files: Optional[List[LabelFileMd]] = None,
      tensor_type: Optional[int] = None,
      score_calibration_md: Optional[ScoreCalibrationMd] = None,
      tensor_name: Optional[str] = None,
      score_thresholding_md: Optional[ScoreThresholdingMd] = None) -> None:
    """Initializes the instance of ClassificationTensorMd.

    Args:
      name: name of the tensor.
      description: description of what the tensor is.
      label_files: information of the label files [1] in the classification
        tensor.
      tensor_type: data type of the tensor.
      score_calibration_md: information of the score calibration files operation
        [2] in the classification tensor.
      tensor_name: name of the corresponding tensor [3] in the TFLite model. It
        is used to locate the corresponding classification tensor and decide the
        order of the tensor metadata [4] when populating model metadata.
      score_thresholding_md: information of the score thresholding [5] in the
        classification tensor.
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L456
      [3]:
        https://github.com/tensorflow/tensorflow/blob/cb67fef35567298b40ac166b0581cd8ad68e5a3a/tensorflow/lite/schema/schema.fbs#L1129-L1136
      [4]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L623-L640
      [5]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L468
    """
    self.score_calibration_md = score_calibration_md
    self.score_thresholding_md = score_thresholding_md

    if tensor_type is _schema_fb.TensorType.UINT8:
      min_values = [_MIN_UINT8]
      max_values = [_MAX_UINT8]
    elif tensor_type is _schema_fb.TensorType.FLOAT32:
      min_values = [self._MIN_FLOAT]
      max_values = [self._MAX_FLOAT]
    else:
      # Uint8 and Float32 are the two major types currently. And Task library
      # doesn't support other types so far.
      min_values = None
      max_values = None

    associated_files = label_files or []
    if self.score_calibration_md:
      associated_files.append(
          score_calibration_md.create_score_calibration_file_md())

    super().__init__(name, description, min_values, max_values,
                     _metadata_fb.ContentProperties.FeatureProperties,
                     associated_files, tensor_name)

  def create_metadata(self) -> _metadata_fb.TensorMetadataT:
    """Creates the classification tensor metadata based on the information."""
    tensor_metadata = super().create_metadata()
    if self.score_calibration_md:
      tensor_metadata.processUnits = [
          self.score_calibration_md.create_metadata()
      ]
    if self.score_thresholding_md:
      if tensor_metadata.processUnits:
        tensor_metadata.processUnits.append(
            self.score_thresholding_md.create_metadata())
      else:
        tensor_metadata.processUnits = [
            self.score_thresholding_md.create_metadata()
        ]
    return tensor_metadata
