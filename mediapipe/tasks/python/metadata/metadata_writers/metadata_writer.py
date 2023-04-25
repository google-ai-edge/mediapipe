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
"""Generic metadata writer."""

import collections
import csv
import dataclasses
import os
import tempfile
from typing import List, Optional, Tuple, Union

import flatbuffers
from mediapipe.tasks.metadata import metadata_schema_py_generated as metadata_fb
from mediapipe.tasks.python.metadata import metadata
from mediapipe.tasks.python.metadata.metadata_writers import metadata_info
from mediapipe.tasks.python.metadata.metadata_writers import writer_utils

_INPUT_IMAGE_NAME = 'image'
_INPUT_IMAGE_DESCRIPTION = 'Input image to be processed.'
_INPUT_REGEX_TEXT_NAME = 'input_text'
_INPUT_REGEX_TEXT_DESCRIPTION = ('Embedding vectors representing the input '
                                 'text to be processed.')
_OUTPUT_CLASSIFICATION_NAME = 'score'
_OUTPUT_CLASSIFICATION_DESCRIPTION = 'Score of the labels respectively.'
_OUTPUT_SEGMENTATION_MASKS_NAME = 'segmentation_masks'
_OUTPUT_SEGMENTATION_MASKS_DESCRIPTION = (
    'Masks over the target objects with high accuracy.'
)
# Detection tensor result to be grouped together.
_DETECTION_GROUP_NAME = 'detection_result'
# File name to export score calibration parameters.
_SCORE_CALIBATION_FILENAME = 'score_calibration.txt'


@dataclasses.dataclass
class CalibrationParameter:
  """Parameters for score calibration [1].

  Score calibration is performed on an output tensor through sigmoid functions.
  One of the main purposes of score calibration is to make scores across classes
  comparable, so that a common threshold can be used for all output classes.

  For each index in the output tensor, this applies:
    * `f(x) = scale / (1 + e^-(slope * g(x) + offset))` if `x > min_score` or if
      no `min_score` has been specified.
    * `f(x) = default_score` otherwise or if no scale, slope and offset have
      been specified.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L434
  """
  scale: float
  slope: float
  offset: float
  min_score: Optional[float] = None


@dataclasses.dataclass
class LabelItem:
  """Label item for labels per locale.

  Attributes:
    filename: The file name to save the labels.
    names: A list of label names.
    locale: The specified locale for labels.
  """
  filename: str
  names: List[str]
  locale: Optional[str] = None


@dataclasses.dataclass
class ScoreThresholding:
  """Parameters to performs thresholding on output tensor values [1].

  Attributes:
    global_score_threshold: The recommended global threshold below which results
      are considered low-confidence and should be filtered out.  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L468
  """
  global_score_threshold: float


@dataclasses.dataclass
class RegexTokenizer:
  """Parameters of the Regex tokenizer [1] metadata information.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L500

  Attributes:
    delim_regex_pattern: the regular expression to segment strings and create
      tokens.
    vocab_file_path: path to the vocabulary file.
  """
  delim_regex_pattern: str
  vocab_file_path: str


@dataclasses.dataclass
class BertTokenizer:
  """Parameters of the Bert tokenizer [1] metadata information.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L477

  Attributes:
    vocab_file_path: path to the vocabulary file.
  """
  vocab_file_path: str


@dataclasses.dataclass
class SentencePieceTokenizer:
  """Parameters of the sentence piece tokenizer tokenizer [1] metadata information.

  [1]:
    https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L485

  Attributes:
    sentence_piece_model_path: path to the sentence piece model file.
    vocab_file_path: path to the vocabulary file.
  """
  sentence_piece_model_path: str
  vocab_file_path: Optional[str] = None


class Labels(object):
  """Simple container holding classification labels of a particular tensor.

  Example usage:
    # The first added label list can be used as category names as needed.
    labels = Labels()
      .add(['/m/011l78', '/m/031d23'])
      .add(['cat', 'dog], 'en')
      .add(['chat', 'chien], 'fr')
  """

  def __init__(self) -> None:
    self._labels = []  # [LabelItem]

  @property
  def labels(self) -> List[LabelItem]:
    return self._labels

  def add(self,
          labels: List[str],
          locale: Optional[str] = None,
          exported_filename: Optional[str] = None) -> 'Labels':
    """Adds labels in the container.

    Args:
      labels: A list of label names, e.g. ['apple', 'pear', 'banana'].
      locale: The specified locale for labels.
      exported_filename: The file name to export the labels. If not set,
        filename defaults to 'labels.txt'.

    Returns:
      The Labels instance, can be used for chained operation.
    """
    if not labels:
      raise ValueError('The list of labels is empty.')

    # Prepare the new item to be inserted
    if not exported_filename:
      exported_filename = 'labels'
      if locale:
        exported_filename += f'_{locale}'
      exported_filename += '.txt'
    item = LabelItem(filename=exported_filename, names=labels, locale=locale)

    # Insert the new element at the end of the list
    self._labels.append(item)
    return self

  def add_from_file(self,
                    label_filepath: str,
                    locale: Optional[str] = None,
                    exported_filename: Optional[str] = None) -> 'Labels':
    """Adds a label file in the container.

    Args:
      label_filepath: File path to read labels. Each line is a label name in the
        file.
      locale: The specified locale for labels.
      exported_filename: The file name to export the labels. If not set,
        filename defaults to 'labels.txt'.

    Returns:
      The Labels instance, can be used for chained operation.
    """

    with open(label_filepath, 'r') as f:
      labels = f.read().split('\n')
      return self.add(labels, locale, exported_filename)


class ScoreCalibration:
  """Simple container holding score calibration related parameters."""

  # A shortcut to avoid client side code importing metadata_fb
  transformation_types = metadata_fb.ScoreTransformationType

  def __init__(self,
               transformation_type: metadata_fb.ScoreTransformationType,
               parameters: List[Optional[CalibrationParameter]],
               default_score: int = 0):
    self.transformation_type = transformation_type
    self.parameters = parameters
    self.default_score = default_score

  @classmethod
  def create_from_file(cls,
                       transformation_type: metadata_fb.ScoreTransformationType,
                       file_path: str,
                       default_score: int = 0) -> 'ScoreCalibration':
    """Creates ScoreCalibration from the file.

    Args:
      transformation_type: type of the function used for transforming the
        uncalibrated score before applying score calibration.
      file_path: file_path of the score calibration file [1]. Contains
        sigmoid-based score calibration parameters, formatted as CSV. Lines
        contain for each index of an output tensor the scale, slope, offset and
        (optional) min_score parameters to be used for sigmoid fitting (in this
        order and in `strtof`-compatible [2] format). Scale should be a
        non-negative value. A line may be left empty to default calibrated
        scores for this index to default_score. In summary, each line should
        thus contain 0, 3 or 4 comma-separated values.
      default_score: the default calibrated score to apply if the uncalibrated
        score is below min_score or if no parameters were specified for a given
        index.
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L133
      [2]:
        https://en.cppreference.com/w/c/string/byte/strtof

    Returns:
      A ScoreCalibration object.
    Raises:
      ValueError: if the score_calibration file is malformed.
    """
    with open(file_path, 'r') as calibration_file:
      csv_reader = csv.reader(calibration_file, delimiter=',')
      parameters = []
      for row in csv_reader:
        if not row:
          parameters.append(None)
          continue

        if len(row) != 3 and len(row) != 4:
          raise ValueError(
              f'Expected empty lines or 3 or 4 parameters per line in score'
              f' calibration file, but got {len(row)}.')

        if float(row[0]) < 0:
          raise ValueError(
              f'Expected scale to be a non-negative value, but got '
              f'{float(row[0])}.')

        parameters.append(
            CalibrationParameter(
                scale=float(row[0]),
                slope=float(row[1]),
                offset=float(row[2]),
                min_score=None if len(row) == 3 else float(row[3])))

    return cls(transformation_type, parameters, default_score)


def _fill_default_tensor_names(
    tensor_metadata_list: List[metadata_fb.TensorMetadataT],
    tensor_names_from_model: List[str]):
  """Fills the default tensor names."""
  # If tensor name in metadata is empty, default to the tensor name saved in
  # the model.
  for tensor_metadata, name in zip(tensor_metadata_list,
                                   tensor_names_from_model):
    tensor_metadata.name = tensor_metadata.name or name


def _pair_tensor_metadata(
    tensor_md: List[metadata_info.TensorMd],
    tensor_names_from_model: List[str]) -> List[metadata_info.TensorMd]:
  """Pairs tensor_md according to the tensor names from the model."""
  tensor_names_from_arg = [
      md.tensor_name for md in tensor_md or [] if md.tensor_name is not None
  ]
  if not tensor_names_from_arg:
    return tensor_md

  if collections.Counter(tensor_names_from_arg) != collections.Counter(
      tensor_names_from_model):
    raise ValueError(
        'The tensor names from arguments ({}) do not match the tensor names'
        ' read from the model ({}).'.format(tensor_names_from_arg,
                                            tensor_names_from_model))
  pairs_tensor_md = []
  name_md_dict = dict(zip(tensor_names_from_arg, tensor_md))
  for name in tensor_names_from_model:
    pairs_tensor_md.append(name_md_dict[name])
  return pairs_tensor_md


def _create_metadata_buffer(
    model_buffer: bytearray,
    general_md: Optional[metadata_info.GeneralMd] = None,
    input_md: Optional[List[metadata_info.TensorMd]] = None,
    output_md: Optional[List[metadata_info.TensorMd]] = None,
    input_process_units: Optional[List[metadata_fb.ProcessUnitT]] = None,
    output_group_md: Optional[List[metadata_info.TensorGroupMd]] = None,
    custom_metadata_md: Optional[List[metadata_info.CustomMetadataMd]] = None,
) -> bytearray:
  """Creates a buffer of the metadata.

  Args:
    model_buffer: valid buffer of the model file.
    general_md: general information about the model.
    input_md: metadata information of the input tensors.
    output_md: metadata information of the output tensors.
    input_process_units: a lists of metadata of the input process units [1].
    output_group_md: a list of metadata of output tensor groups [2];
    custom_metadata_md: a lists of custom metadata.
    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L655
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L677

  Returns:
    A buffer of the metadata.

  Raises:
    ValueError: if the tensor names from `input_md` and `output_md` do not
    match the tensor names read from the model.
  """
  # Create input metadata from `input_md`.
  if input_md:
    input_md = _pair_tensor_metadata(
        input_md, writer_utils.get_input_tensor_names(model_buffer))
    input_metadata = [m.create_metadata() for m in input_md]
  else:
    num_input_tensors = writer_utils.get_subgraph(model_buffer).InputsLength()
    input_metadata = [metadata_fb.TensorMetadataT()] * num_input_tensors

  _fill_default_tensor_names(input_metadata,
                             writer_utils.get_input_tensor_names(model_buffer))

  # Create output metadata from `output_md`.
  if output_md:
    output_md = _pair_tensor_metadata(
        output_md, writer_utils.get_output_tensor_names(model_buffer))
    output_metadata = [m.create_metadata() for m in output_md]
  else:
    num_output_tensors = writer_utils.get_subgraph(model_buffer).OutputsLength()
    output_metadata = [metadata_fb.TensorMetadataT()] * num_output_tensors
  _fill_default_tensor_names(output_metadata,
                             writer_utils.get_output_tensor_names(model_buffer))

  # Create the subgraph metadata.
  subgraph_metadata = metadata_fb.SubGraphMetadataT()
  subgraph_metadata.inputTensorMetadata = input_metadata
  subgraph_metadata.outputTensorMetadata = output_metadata
  if input_process_units:
    subgraph_metadata.inputProcessUnits = input_process_units
  if custom_metadata_md:
    subgraph_metadata.customMetadata = [
        m.create_metadata() for m in custom_metadata_md
    ]
  if output_group_md:
    subgraph_metadata.outputTensorGroups = [
        m.create_metadata() for m in output_group_md
    ]

  # Create the whole model metadata.
  if general_md is None:
    general_md = metadata_info.GeneralMd()
  model_metadata = general_md.create_metadata()
  model_metadata.subgraphMetadata = [subgraph_metadata]

  # Get the metadata flatbuffer.
  b = flatbuffers.Builder(0)
  b.Finish(
      model_metadata.Pack(b),
      metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  return b.Output()


class MetadataWriter(object):
  """Generic Metadata writer.

  Example usage:

  For an example model which requires two inputs: image and general feature
  inputs, and generates one output: classification.

  with open(model_path, 'rb') as f:
    writer = MetadataWriter.create(f.read())
    model_content, metadata_json_content = writer
        .add_genernal_info('model_name', 'model description')
        .add_image_input()
        .add_feature_input()
        .add_classification_output(Labels().add(['A', 'B']))
        .populate()
  """

  @classmethod
  def create(cls, model_buffer: bytearray) -> 'MetadataWriter':
    return cls(model_buffer)

  def __init__(self, model_buffer: bytearray) -> None:
    self._model_buffer = model_buffer
    self._general_md = None
    self._input_mds = []
    self._input_process_units = []
    self._output_mds = []
    self._output_group_mds = []
    self._associated_files = []
    self._custom_metadata_mds = []
    self._temp_folder = tempfile.TemporaryDirectory()

  def __del__(self):
    if os.path.exists(self._temp_folder.name):
      self._temp_folder.cleanup()

  def add_general_info(
      self,
      model_name: str,
      model_description: Optional[str] = None) -> 'MetadataWriter':
    """Adds a general info metadata for the general metadata informantion."""
    # Will overwrite the previous `self._general_md` if exists.
    self._general_md = metadata_info.GeneralMd(
        name=model_name, description=model_description)
    return self

  color_space_types = metadata_fb.ColorSpaceType

  def add_feature_input(self,
                        name: Optional[str] = None,
                        description: Optional[str] = None) -> 'MetadataWriter':
    """Adds an input tensor metadata for the general basic feature input."""
    input_md = metadata_info.TensorMd(name=name, description=description)
    self._input_mds.append(input_md)
    return self

  def add_image_input(
      self,
      norm_mean: List[float],
      norm_std: List[float],
      color_space_type: Optional[int] = metadata_fb.ColorSpaceType.RGB,
      name: str = _INPUT_IMAGE_NAME,
      description: str = _INPUT_IMAGE_DESCRIPTION) -> 'MetadataWriter':
    """Adds an input image metadata for the image input.

    Args:
      norm_mean: The mean value used to normalize each input channel. If there
        is only one element in the list, its value will be broadcasted to all
        channels. Also note that norm_mean and norm_std should have the same
        number of elements. [1]
      norm_std: The std value used to normalize each input channel. If there is
        only one element in the list, its value will be broadcasted to all
        channels. [1]
      color_space_type: The color space type of the input image. [2]
      name: Name of the input tensor.
      description: Description of the input tensor.

    Returns:
      The MetadataWriter instance, can be used for chained operation.

    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L198
    """
    input_md = metadata_info.InputImageTensorMd(
        name=name,
        description=description,
        norm_mean=norm_mean,
        norm_std=norm_std,
        color_space_type=color_space_type,
        tensor_type=self._input_tensor_type(len(self._input_mds)))

    self._input_mds.append(input_md)
    return self

  def add_regex_text_input(
      self,
      regex_tokenizer: RegexTokenizer,
      name: str = _INPUT_REGEX_TEXT_NAME,
      description: str = _INPUT_REGEX_TEXT_DESCRIPTION) -> 'MetadataWriter':
    """Adds an input text metadata for the text input with regex tokenizer.

    Args:
      regex_tokenizer: information of the regex tokenizer [1] used to process
        the input string.
      name: Name of the input tensor.
      description: Description of the input tensor.

    Returns:
      The MetadataWriter instance, can be used for chained operation.

    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L500
    """
    tokenizer_md = metadata_info.RegexTokenizerMd(
        delim_regex_pattern=regex_tokenizer.delim_regex_pattern,
        vocab_file_path=regex_tokenizer.vocab_file_path)
    input_md = metadata_info.InputTextTensorMd(
        name=name, description=description, tokenizer_md=tokenizer_md)
    self._input_mds.append(input_md)
    self._associated_files.append(regex_tokenizer.vocab_file_path)
    return self

  def add_bert_text_input(self, tokenizer: Union[BertTokenizer,
                                                 SentencePieceTokenizer],
                          ids_name: str, mask_name: str,
                          segment_name: str) -> 'MetadataWriter':
    """Adds an metadata for the text input with bert / sentencepiece tokenizer.

    `ids_name`, `mask_name`, and `segment_name` correspond to the `Tensor.name`
    in the TFLite schema, which help to determine the tensor order when
    populating metadata.

    Args:
      tokenizer: information of the tokenizer used to process the input string,
        if any. Supported tokenziers are: `BertTokenizer` [1] and
        `SentencePieceTokenizer` [2].
      ids_name: name of the ids tensor, which represents the tokenized ids of
        the input text.
      mask_name: name of the mask tensor, which represents the mask with `1` for
        real tokens and `0` for padding tokens.
      segment_name: name of the segment ids tensor, where `0` stands for the
        first sequence, and `1` stands for the second sequence if exists.

    Returns:
      The MetadataWriter instance, can be used for chained operation.

    Raises:
      ValueError: if the type tokenizer is not BertTokenizer or
        SentencePieceTokenizer.

    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L477
    [2]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L485
    """
    if isinstance(tokenizer, BertTokenizer):
      tokenizer_md = metadata_info.BertTokenizerMd(
          vocab_file_path=tokenizer.vocab_file_path)
    elif isinstance(tokenizer, SentencePieceTokenizer):
      tokenizer_md = metadata_info.SentencePieceTokenizerMd(
          sentence_piece_model_path=tokenizer.sentence_piece_model_path,
          vocab_file_path=tokenizer.vocab_file_path)
    else:
      raise ValueError(
          f'The type of tokenizer, {type(tokenizer)}, is unsupported')
    bert_input_md = metadata_info.BertInputTensorsMd(
        self._model_buffer,
        ids_name,
        mask_name,
        segment_name,
        tokenizer_md=tokenizer_md)

    self._input_mds.extend(bert_input_md.input_md)
    self._associated_files.extend(
        bert_input_md.get_tokenizer_associated_files())
    self._input_process_units.extend(
        bert_input_md.create_input_process_unit_metadata())
    return self

  def add_classification_output(
      self,
      labels: Optional[Labels] = None,
      score_calibration: Optional[ScoreCalibration] = None,
      score_thresholding: Optional[ScoreThresholding] = None,
      name: str = _OUTPUT_CLASSIFICATION_NAME,
      description: str = _OUTPUT_CLASSIFICATION_DESCRIPTION
  ) -> 'MetadataWriter':
    """Add a classification head metadata for classification output tensor.

    Example usage:
      writer.add_classification_output(
        Labels()
          .add(['/m/011l78', '/m/031d23'])
          .add(['cat', 'dog], 'en')
          .add(['chat', 'chien], 'fr')
          )

    Args:
      labels: an instance of Labels helper class.
      score_calibration: an instance of ScoreCalibration helper class.
      score_thresholding: an instance of ScoreThresholding.
      name: Metadata name of the tensor. Note that this is different from tensor
        name in the flatbuffer.
      description: human readable description of what the output is.

    Returns:
      The current Writer instance to allow chained operation.
    """
    calibration_md = self._create_score_calibration_md(score_calibration)
    score_thresholding_md = None
    if score_thresholding:
      score_thresholding_md = metadata_info.ScoreThresholdingMd(
          score_thresholding.global_score_threshold)

    label_files = self._create_label_file_md(labels)
    output_md = metadata_info.ClassificationTensorMd(
        name=name,
        description=description,
        label_files=label_files,
        tensor_type=self._output_tensor_type(len(self._output_mds)),
        score_calibration_md=calibration_md,
        score_thresholding_md=score_thresholding_md,
    )
    self._output_mds.append(output_md)
    return self

  def add_detection_output(
      self,
      labels: Optional[Labels] = None,
      score_calibration: Optional[ScoreCalibration] = None,
      group_name: str = _DETECTION_GROUP_NAME,
  ) -> 'MetadataWriter':
    """Adds a detection head metadata for detection output tensor of models with postprocessing.

    Args:
      labels: an instance of Labels helper class.
      score_calibration: an instance of ScoreCalibration helper class.
      group_name: name of output tensor group.

    Returns:
      The current Writer instance to allow chained operation.
    """
    calibration_md = self._create_score_calibration_md(score_calibration)
    label_files = self._create_label_file_md(labels)
    detection_output_mds = metadata_info.DetectionOutputTensorsMd(
        self._model_buffer,
        label_files=label_files,
        score_calibration_md=calibration_md,
    ).output_mds
    self._output_mds.extend(detection_output_mds)
    # Outputs are location, category, score, number of detections.
    if len(detection_output_mds) != 4:
      raise ValueError('The size of detections output should be 4.')
    # The first 3 tensors (location, category, score) are grouped.
    group_md = metadata_info.TensorGroupMd(
        name=group_name,
        tensor_names=[output_md.name for output_md in detection_output_mds[:3]],
    )
    self._output_group_mds.append(group_md)
    return self

  def add_raw_detection_output(
      self,
      labels: Optional[Labels] = None,
      output_tensors_order: metadata_info.RawDetectionOutputTensorsOrder = metadata_info.RawDetectionOutputTensorsOrder.UNSPECIFIED,
  ) -> 'MetadataWriter':
    """Adds a detection head metadata for detection output tensor of models without postprocessing.

    Args:
      labels: an instance of Labels helper class.
      output_tensors_order: the order of the output tensors. For models of
        out-of-graph non-maximum-suppression only.

    Returns:
      The current Writer instance to allow chained operation.
    """
    label_files = self._create_label_file_md(labels)
    detection_output_mds = metadata_info.RawDetectionOutputTensorsMd(
        self._model_buffer,
        label_files=label_files,
        output_tensors_order=output_tensors_order,
    ).output_mds
    self._output_mds.extend(detection_output_mds)
    # Outputs are location, score.
    if len(detection_output_mds) != 2:
      raise ValueError('The size of detections output should be 2.')
    return self

  def add_segmentation_output(
      self,
      labels: Optional[Labels] = None,
      name: str = _OUTPUT_SEGMENTATION_MASKS_NAME,
      description: str = _OUTPUT_SEGMENTATION_MASKS_DESCRIPTION,
  ) -> 'MetadataWriter':
    """Adds a segmentation head metadata for segmentation output tensor.

    Args:
      labels: an instance of Labels helper class.
      name: Metadata name of the tensor. Note that this is different from tensor
        name in the flatbuffer.
      description: human readable description of what the output is.

    Returns:
      The current Writer instance to allow chained operation.
    """
    label_files = self._create_label_file_md(labels)
    output_md = metadata_info.SegmentationMaskMd(
        name=name,
        description=description,
        label_files=label_files,
    )
    self._output_mds.append(output_md)
    return self

  def add_feature_output(self,
                         name: Optional[str] = None,
                         description: Optional[str] = None) -> 'MetadataWriter':
    """Adds an output tensor metadata for the general basic feature output."""
    output_md = metadata_info.TensorMd(name=name, description=description)
    self._output_mds.append(output_md)
    return self

  def add_custom_metadata(
      self, custom_metadata_md: metadata_info.CustomMetadataMd
  ) -> 'MetadataWriter':
    self._custom_metadata_mds.append(custom_metadata_md)
    return self

  def populate(self) -> Tuple[bytearray, str]:
    """Populates metadata into the TFLite file.

    Note that only the output tflite is used for deployment. The output JSON
    content is used to interpret the metadata content.

    Returns:
      A tuple of (model_with_metadata_in_bytes, metdata_json_content)
    """
    # Populates metadata and associated files into TFLite model buffer.
    populator = metadata.MetadataPopulator.with_model_buffer(self._model_buffer)
    metadata_buffer = _create_metadata_buffer(
        model_buffer=self._model_buffer,
        general_md=self._general_md,
        input_md=self._input_mds,
        output_md=self._output_mds,
        input_process_units=self._input_process_units,
        custom_metadata_md=self._custom_metadata_mds,
        output_group_md=self._output_group_mds,
    )
    populator.load_metadata_buffer(metadata_buffer)
    if self._associated_files:
      populator.load_associated_files(self._associated_files)
    populator.populate()
    tflite_content = populator.get_model_buffer()

    displayer = metadata.MetadataDisplayer.with_model_buffer(tflite_content)
    metadata_json_content = displayer.get_metadata_json()

    return tflite_content, metadata_json_content

  def _input_tensor_type(self, idx):
    return writer_utils.get_input_tensor_types(self._model_buffer)[idx]

  def _output_tensor_type(self, idx):
    return writer_utils.get_output_tensor_types(self._model_buffer)[idx]

  def _export_labels(self, filename: str, index_to_label: List[str]) -> str:
    filepath = os.path.join(self._temp_folder.name, filename)
    with open(filepath, 'w') as f:
      f.write('\n'.join(index_to_label))
    self._associated_files.append(filepath)
    return filepath

  def _export_calibration_file(self, filename: str,
                               calibrations: List[CalibrationParameter]) -> str:
    """Stores calibration parameters in a csv file."""
    filepath = os.path.join(self._temp_folder.name, filename)
    with open(filepath, 'w') as f:
      for item in calibrations:
        if item:
          if item.scale is None or item.slope is None or item.offset is None:
            raise ValueError('scale, slope and offset values can not be set to '
                             'None.')
          elif item.min_score is not None:
            f.write(f'{item.scale},{item.slope},{item.offset},{item.min_score}')
          else:
            f.write(f'{item.scale},{item.slope},{item.offset}')
        f.write('\n')

    self._associated_files.append(filepath)
    return filepath

  def _create_score_calibration_md(
      self, score_calibration: ScoreCalibration
  ) -> Optional[metadata_info.ScoreCalibrationMd]:
    """Creates the ScoreCalibrationMd object."""
    if score_calibration is None:
      return None
    return metadata_info.ScoreCalibrationMd(
        score_transformation_type=score_calibration.transformation_type,
        default_score=score_calibration.default_score,
        file_path=self._export_calibration_file(
            _SCORE_CALIBATION_FILENAME, score_calibration.parameters
        ),
    )

  def _create_label_file_md(
      self, labels: Optional[Labels] = None
  ) -> Optional[List[metadata_info.LabelFileMd]]:
    """Creates a list of LabelFileMd objects."""
    label_files = None
    if labels:
      label_files = []
      for item in labels.labels:
        label_files.append(
            metadata_info.LabelFileMd(
                self._export_labels(item.filename, item.names),
                locale=item.locale,
            )
        )
    return label_files


class MetadataWriterBase:
  """Base MetadataWriter class which contains the apis exposed to users.

  MetadataWriter for Tasks e.g. image classifier / object detector will inherit
  this class for their own usage.
  """

  def __init__(self, writer: MetadataWriter) -> None:
    self.writer = writer

  def populate(self) -> Tuple[bytearray, str]:
    """Populates metadata into the TFLite file.

    Note that only the output tflite is used for deployment. The output JSON
    content is used to interpret the metadata content.

    Returns:
      A tuple of (model_with_metadata_in_bytes, metdata_json_content)
    """
    return self.writer.populate()
