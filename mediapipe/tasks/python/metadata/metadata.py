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
"""TensorFlow Lite metadata tools."""

import copy
import inspect
import io
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, Optional
import warnings
import zipfile

import flatbuffers
from mediapipe.tasks.cc.metadata.python import _pywrap_metadata_version
from mediapipe.tasks.metadata import metadata_schema_py_generated as _metadata_fb
from mediapipe.tasks.metadata import schema_py_generated as _schema_fb
from mediapipe.tasks.python.metadata.flatbuffers_lib import _pywrap_flatbuffers

try:
  # If exists, optionally use TensorFlow to open and check files. Used to
  # support more than local file systems.
  # In pip requirements, we doesn't necessarily need tensorflow as a dep.
  import tensorflow as tf
  _open_file = tf.io.gfile.GFile
  _exists_file = tf.io.gfile.exists
except ImportError as e:
  # If TensorFlow package doesn't exist, fall back to original open and exists.
  _open_file = open
  _exists_file = os.path.exists


def _maybe_open_as_binary(filename, mode):
  """Maybe open the binary file, and returns a file-like."""
  if hasattr(filename, "read"):  # A file-like has read().
    return filename
  openmode = mode if "b" in mode else mode + "b"  # Add binary explicitly.
  return _open_file(filename, openmode)


def _open_as_zipfile(filename, mode="r"):
  """Open file as a zipfile.

  Args:
    filename: str or file-like or path-like, to the zipfile.
    mode: str, common file mode for zip.
          (See: https://docs.python.org/3/library/zipfile.html)

  Returns:
    A ZipFile object.
  """
  file_like = _maybe_open_as_binary(filename, mode)
  return zipfile.ZipFile(file_like, mode)


def _is_zipfile(filename):
  """Checks whether it is a zipfile."""
  with _maybe_open_as_binary(filename, "r") as f:
    return zipfile.is_zipfile(f)


def get_path_to_datafile(path):
  """Gets the path to the specified file in the data dependencies.

  The path is relative to the file calling the function.

  It's a simple replacement of
  "tensorflow.python.platform.resource_loader.get_path_to_datafile".

  Args:
    path: a string resource path relative to the calling file.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.
  """
  data_files_path = os.path.dirname(inspect.getfile(sys._getframe(1)))  # pylint: disable=protected-access
  return os.path.join(data_files_path, path)


_FLATC_TFLITE_METADATA_SCHEMA_FILE = get_path_to_datafile(
    "../../metadata/metadata_schema.fbs")


# TODO: add delete method for associated files.
class MetadataPopulator(object):
  """Packs metadata and associated files into TensorFlow Lite model file.

  MetadataPopulator can be used to populate metadata and model associated files
  into a model file or a model buffer (in bytearray). It can also help to
  inspect list of files that have been packed into the model or are supposed to
  be packed into the model.

  The metadata file (or buffer) should be generated based on the metadata
  schema:
  mediapipe/tasks/metadata/metadata_schema.fbs

  Example usage:
  Populate metadata and label file into an image classifier model.

  First, based on metadata_schema.fbs, generate the metadata for this image
  classifier model using Flatbuffers API. Attach the label file onto the output
  tensor (the tensor of probabilities) in the metadata.

  Then, pack the metadata and label file into the model as follows.

    ```python
    # Populating a metadata file (or a metadata buffer) and associated files to
    a model file:
    populator = MetadataPopulator.with_model_file(model_file)
    # For metadata buffer (bytearray read from the metadata file), use:
    # populator.load_metadata_buffer(metadata_buf)
    populator.load_metadata_file(metadata_file)
    populator.load_associated_files([label.txt])
    # For associated file buffer (bytearray read from the file), use:
    # populator.load_associated_file_buffers({"label.txt": b"file content"})
    populator.populate()

    # Populating a metadata file (or a metadata buffer) and associated files to
    a model buffer:
    populator = MetadataPopulator.with_model_buffer(model_buf)
    populator.load_metadata_file(metadata_file)
    populator.load_associated_files([label.txt])
    populator.populate()
    # Writing the updated model buffer into a file.
    updated_model_buf = populator.get_model_buffer()
    with open("updated_model.tflite", "wb") as f:
      f.write(updated_model_buf)

    # Transferring metadata and associated files from another TFLite model:
    populator = MetadataPopulator.with_model_buffer(model_buf)
    populator_dst.load_metadata_and_associated_files(src_model_buf)
    populator_dst.populate()
    updated_model_buf = populator.get_model_buffer()
    with open("updated_model.tflite", "wb") as f:
      f.write(updated_model_buf)
    ```

  Note that existing metadata buffer (if applied) will be overridden by the new
  metadata buffer.
  """
  # As Zip API is used to concatenate associated files after tflite model file,
  # the populating operation is developed based on a model file. For in-memory
  # model buffer, we create a tempfile to serve the populating operation.
  # Creating the deleting such a tempfile is handled by the class,
  # _MetadataPopulatorWithBuffer.

  METADATA_FIELD_NAME = "TFLITE_METADATA"
  TFLITE_FILE_IDENTIFIER = b"TFL3"
  METADATA_FILE_IDENTIFIER = b"M001"

  def __init__(self, model_file):
    """Constructor for MetadataPopulator.

    Args:
      model_file: valid path to a TensorFlow Lite model file.

    Raises:
      IOError: File not found.
      ValueError: the model does not have the expected flatbuffer identifier.
    """
    _assert_model_file_identifier(model_file)
    self._model_file = model_file
    self._metadata_buf = None
    # _associated_files is a dict of file name and file buffer.
    self._associated_files = {}

  @classmethod
  def with_model_file(cls, model_file):
    """Creates a MetadataPopulator object that populates data to a model file.

    Args:
      model_file: valid path to a TensorFlow Lite model file.

    Returns:
      MetadataPopulator object.

    Raises:
      IOError: File not found.
      ValueError: the model does not have the expected flatbuffer identifier.
    """
    return cls(model_file)

  # TODO: investigate if type check can be applied to model_buf for
  # FB.
  @classmethod
  def with_model_buffer(cls, model_buf):
    """Creates a MetadataPopulator object that populates data to a model buffer.

    Args:
      model_buf: TensorFlow Lite model buffer in bytearray.

    Returns:
      A MetadataPopulator(_MetadataPopulatorWithBuffer) object.

    Raises:
      ValueError: the model does not have the expected flatbuffer identifier.
    """
    return _MetadataPopulatorWithBuffer(model_buf)

  def get_model_buffer(self):
    """Gets the buffer of the model with packed metadata and associated files.

    Returns:
      Model buffer (in bytearray).
    """
    with _open_file(self._model_file, "rb") as f:
      return f.read()

  def get_packed_associated_file_list(self):
    """Gets a list of associated files packed to the model file.

    Returns:
      List of packed associated files.
    """
    if not _is_zipfile(self._model_file):
      return []

    with _open_as_zipfile(self._model_file, "r") as zf:
      return zf.namelist()

  def get_recorded_associated_file_list(self):
    """Gets a list of associated files recorded in metadata of the model file.

    Associated files may be attached to a model, a subgraph, or an input/output
    tensor.

    Returns:
      List of recorded associated files.
    """
    if not self._metadata_buf:
      return []

    metadata = _metadata_fb.ModelMetadataT.InitFromObj(
        _metadata_fb.ModelMetadata.GetRootAsModelMetadata(
            self._metadata_buf, 0))

    return [
        file.name.decode("utf-8")
        for file in self._get_recorded_associated_file_object_list(metadata)
    ]

  def load_associated_file_buffers(self, associated_files):
    """Loads the associated file buffers (in bytearray) to be populated.

    Args:
      associated_files: a dictionary of associated file names and corresponding
        file buffers, such as {"file.txt": b"file content"}. If pass in file
          paths for the file name, only the basename will be populated.
    """

    self._associated_files.update({
        os.path.basename(name): buffers
        for name, buffers in associated_files.items()
    })

  def load_associated_files(self, associated_files):
    """Loads associated files that to be concatenated after the model file.

    Args:
      associated_files: list of file paths.

    Raises:
      IOError:
        File not found.
    """
    for af_name in associated_files:
      _assert_file_exist(af_name)
      with _open_file(af_name, "rb") as af:
        self.load_associated_file_buffers({af_name: af.read()})

  def load_metadata_buffer(self, metadata_buf):
    """Loads the metadata buffer (in bytearray) to be populated.

    Args:
      metadata_buf: metadata buffer (in bytearray) to be populated.

    Raises:
      ValueError: The metadata to be populated is empty.
      ValueError: The metadata does not have the expected flatbuffer identifier.
      ValueError: Cannot get minimum metadata parser version.
      ValueError: The number of SubgraphMetadata is not 1.
      ValueError: The number of input/output tensors does not match the number
        of input/output tensor metadata.
    """
    if not metadata_buf:
      raise ValueError("The metadata to be populated is empty.")

    self._validate_metadata(metadata_buf)

    # Gets the minimum metadata parser version of the metadata_buf.
    min_version = _pywrap_metadata_version.GetMinimumMetadataParserVersion(
        bytes(metadata_buf))

    # Inserts in the minimum metadata parser version into the metadata_buf.
    metadata = _metadata_fb.ModelMetadataT.InitFromObj(
        _metadata_fb.ModelMetadata.GetRootAsModelMetadata(metadata_buf, 0))
    metadata.minParserVersion = min_version

    # Remove local file directory in the `name` field of `AssociatedFileT`, and
    # make it consistent with the name of the actual file packed in the model.
    self._use_basename_for_associated_files_in_metadata(metadata)

    b = flatbuffers.Builder(0)
    b.Finish(metadata.Pack(b), self.METADATA_FILE_IDENTIFIER)
    metadata_buf_with_version = b.Output()

    self._metadata_buf = metadata_buf_with_version

  def load_metadata_file(self, metadata_file):
    """Loads the metadata file to be populated.

    Args:
      metadata_file: path to the metadata file to be populated.

    Raises:
      IOError: File not found.
      ValueError: The metadata to be populated is empty.
      ValueError: The metadata does not have the expected flatbuffer identifier.
      ValueError: Cannot get minimum metadata parser version.
      ValueError: The number of SubgraphMetadata is not 1.
      ValueError: The number of input/output tensors does not match the number
        of input/output tensor metadata.
    """
    _assert_file_exist(metadata_file)
    with _open_file(metadata_file, "rb") as f:
      metadata_buf = f.read()
    self.load_metadata_buffer(bytearray(metadata_buf))

  def load_metadata_and_associated_files(self, src_model_buf):
    """Loads the metadata and associated files from another model buffer.

    Args:
      src_model_buf: source model buffer (in bytearray) with metadata and
        associated files.
    """
    # Load the model metadata from src_model_buf if exist.
    metadata_buffer = get_metadata_buffer(src_model_buf)
    if metadata_buffer:
      self.load_metadata_buffer(metadata_buffer)

    # Load the associated files from src_model_buf if exist.
    if _is_zipfile(io.BytesIO(src_model_buf)):
      with _open_as_zipfile(io.BytesIO(src_model_buf)) as zf:
        self.load_associated_file_buffers(
            {f: zf.read(f) for f in zf.namelist()})

  def populate(self):
    """Populates loaded metadata and associated files into the model file."""
    self._assert_validate()
    self._populate_metadata_buffer()
    self._populate_associated_files()

  def _assert_validate(self):
    """Validates the metadata and associated files to be populated.

    Raises:
      ValueError:
        File is recorded in the metadata, but is not going to be populated.
        File has already been packed.
    """
    # Gets files that are recorded in metadata.
    recorded_files = self.get_recorded_associated_file_list()

    # Gets files that have been packed to self._model_file.
    packed_files = self.get_packed_associated_file_list()

    # Gets the file name of those associated files to be populated.
    to_be_populated_files = self._associated_files.keys()

    # Checks all files recorded in the metadata will be populated.
    for rf in recorded_files:
      if rf not in to_be_populated_files and rf not in packed_files:
        raise ValueError("File, '{0}', is recorded in the metadata, but has "
                         "not been loaded into the populator.".format(rf))

    for f in to_be_populated_files:
      if f in packed_files:
        raise ValueError("File, '{0}', has already been packed.".format(f))

      if f not in recorded_files:
        warnings.warn(
            "File, '{0}', does not exist in the metadata. But packing it to "
            "tflite model is still allowed.".format(f))

  def _copy_archived_files(self, src_zip, file_list, dst_zip):
    """Copy archieved files in file_list from src_zip ro dst_zip."""

    if not _is_zipfile(src_zip):
      raise ValueError("File, '{0}', is not a zipfile.".format(src_zip))

    with _open_as_zipfile(src_zip, "r") as src_zf, \
         _open_as_zipfile(dst_zip, "a") as dst_zf:
      src_list = src_zf.namelist()
      for f in file_list:
        if f not in src_list:
          raise ValueError(
              "File, '{0}', does not exist in the zipfile, {1}.".format(
                  f, src_zip))
        file_buffer = src_zf.read(f)
        dst_zf.writestr(f, file_buffer)

  def _get_associated_files_from_process_units(self, table, field_name):
    """Gets the files that are attached the process units field of a table.

    Args:
      table: a Flatbuffers table object that contains fields of an array of
        ProcessUnit, such as TensorMetadata and SubGraphMetadata.
      field_name: the name of the field in the table that represents an array of
        ProcessUnit. If the table is TensorMetadata, field_name can be
        "ProcessUnits". If the table is SubGraphMetadata, field_name can be
        either "InputProcessUnits" or "OutputProcessUnits".

    Returns:
      A list of AssociatedFileT objects.
    """

    if table is None:
      return []

    file_list = []
    process_units = getattr(table, field_name)
    # If the process_units field is not populated, it will be None. Use an
    # empty list to skip the check.
    for process_unit in process_units or []:
      options = process_unit.options
      if isinstance(options, (_metadata_fb.BertTokenizerOptionsT,
                              _metadata_fb.RegexTokenizerOptionsT)):
        file_list += self._get_associated_files_from_table(options, "vocabFile")
      elif isinstance(options, _metadata_fb.SentencePieceTokenizerOptionsT):
        file_list += self._get_associated_files_from_table(
            options, "sentencePieceModel")
        file_list += self._get_associated_files_from_table(options, "vocabFile")
    return file_list

  def _get_associated_files_from_table(self, table, field_name):
    """Gets the associated files that are attached a table directly.

    Args:
      table: a Flatbuffers table object that contains fields of an array of
        AssociatedFile, such as TensorMetadata and BertTokenizerOptions.
      field_name: the name of the field in the table that represents an array of
        ProcessUnit. If the table is TensorMetadata, field_name can be
        "AssociatedFiles". If the table is BertTokenizerOptions, field_name can
        be "VocabFile".

    Returns:
      A list of AssociatedFileT objects.
    """

    if table is None:
      return []

    # If the associated file field is not populated,
    # `getattr(table, field_name)` will be None. Return an empty list.
    return getattr(table, field_name) or []

  def _get_recorded_associated_file_object_list(self, metadata):
    """Gets a list of AssociatedFileT objects recorded in the metadata.

    Associated files may be attached to a model, a subgraph, or an input/output
    tensor.

    Args:
      metadata: the ModelMetadataT object.

    Returns:
      List of recorded AssociatedFileT objects.
    """
    recorded_files = []

    # Add associated files attached to ModelMetadata.
    recorded_files += self._get_associated_files_from_table(
        metadata, "associatedFiles")

    # Add associated files attached to each SubgraphMetadata.
    for subgraph in metadata.subgraphMetadata or []:
      recorded_files += self._get_associated_files_from_table(
          subgraph, "associatedFiles")

      # Add associated files attached to each input tensor.
      for tensor_metadata in subgraph.inputTensorMetadata or []:
        recorded_files += self._get_associated_files_from_table(
            tensor_metadata, "associatedFiles")
        recorded_files += self._get_associated_files_from_process_units(
            tensor_metadata, "processUnits")

      # Add associated files attached to each output tensor.
      for tensor_metadata in subgraph.outputTensorMetadata or []:
        recorded_files += self._get_associated_files_from_table(
            tensor_metadata, "associatedFiles")
        recorded_files += self._get_associated_files_from_process_units(
            tensor_metadata, "processUnits")

      # Add associated files attached to the input_process_units.
      recorded_files += self._get_associated_files_from_process_units(
          subgraph, "inputProcessUnits")

      # Add associated files attached to the output_process_units.
      recorded_files += self._get_associated_files_from_process_units(
          subgraph, "outputProcessUnits")

    return recorded_files

  def _populate_associated_files(self):
    """Concatenates associated files after TensorFlow Lite model file.

    If the MetadataPopulator object is created using the method,
    with_model_file(model_file), the model file will be updated.
    """
    # Opens up the model file in "appending" mode.
    # If self._model_file already has pack files, zipfile will concatenate
    # addition files after self._model_file. For example, suppose we have
    # self._model_file = old_tflite_file | label1.txt | label2.txt
    # Then after trigger populate() to add label3.txt, self._model_file becomes
    # self._model_file = old_tflite_file | label1.txt | label2.txt | label3.txt
    with tempfile.SpooledTemporaryFile() as temp:
      # (1) Copy content from model file of to temp file.
      with _open_file(self._model_file, "rb") as f:
        shutil.copyfileobj(f, temp)

      # (2) Append of to a temp file as a zip.
      with _open_as_zipfile(temp, "a") as zf:
        for file_name, file_buffer in self._associated_files.items():
          zf.writestr(file_name, file_buffer)

      # (3) Copy temp file to model file.
      temp.seek(0)
      with _open_file(self._model_file, "wb") as f:
        shutil.copyfileobj(temp, f)

  def _populate_metadata_buffer(self):
    """Populates the metadata buffer (in bytearray) into the model file.

    Inserts metadata_buf into the metadata field of schema.Model. If the
    MetadataPopulator object is created using the method,
    with_model_file(model_file), the model file will be updated.

    Existing metadata buffer (if applied) will be overridden by the new metadata
    buffer.
    """

    with _open_file(self._model_file, "rb") as f:
      model_buf = f.read()

    model = _schema_fb.ModelT.InitFromObj(
        _schema_fb.Model.GetRootAsModel(model_buf, 0))
    buffer_field = _schema_fb.BufferT()
    buffer_field.data = self._metadata_buf

    is_populated = False
    if not model.metadata:
      model.metadata = []
    else:
      # Check if metadata has already been populated.
      for meta in model.metadata:
        if meta.name.decode("utf-8") == self.METADATA_FIELD_NAME:
          is_populated = True
          model.buffers[meta.buffer] = buffer_field

    if not is_populated:
      if not model.buffers:
        model.buffers = []
      model.buffers.append(buffer_field)
      # Creates a new metadata field.
      metadata_field = _schema_fb.MetadataT()
      metadata_field.name = self.METADATA_FIELD_NAME
      metadata_field.buffer = len(model.buffers) - 1
      model.metadata.append(metadata_field)

    # Packs model back to a flatbuffer binaray file.
    b = flatbuffers.Builder(0)
    b.Finish(model.Pack(b), self.TFLITE_FILE_IDENTIFIER)
    model_buf = b.Output()

    # Saves the updated model buffer to model file.
    # Gets files that have been packed to self._model_file.
    packed_files = self.get_packed_associated_file_list()
    if packed_files:
      # Writes the updated model buffer and associated files into a new model
      # file (in memory). Then overwrites the original model file.
      with tempfile.SpooledTemporaryFile() as temp:
        temp.write(model_buf)
        self._copy_archived_files(self._model_file, packed_files, temp)
        temp.seek(0)
        with _open_file(self._model_file, "wb") as f:
          shutil.copyfileobj(temp, f)
    else:
      with _open_file(self._model_file, "wb") as f:
        f.write(model_buf)

  def _use_basename_for_associated_files_in_metadata(self, metadata):
    """Removes any associated file local directory (if exists)."""
    for file in self._get_recorded_associated_file_object_list(metadata):
      file.name = os.path.basename(file.name)

  def _validate_metadata(self, metadata_buf):
    """Validates the metadata to be populated."""
    _assert_metadata_buffer_identifier(metadata_buf)

    # Verify the number of SubgraphMetadata is exactly one.
    # TFLite currently only support one subgraph.
    model_meta = _metadata_fb.ModelMetadata.GetRootAsModelMetadata(
        metadata_buf, 0)
    if model_meta.SubgraphMetadataLength() != 1:
      raise ValueError("The number of SubgraphMetadata should be exactly one, "
                       "but got {0}.".format(
                           model_meta.SubgraphMetadataLength()))

    # Verify if the number of tensor metadata matches the number of tensors.
    with _open_file(self._model_file, "rb") as f:
      model_buf = f.read()
    model = _schema_fb.Model.GetRootAsModel(model_buf, 0)

    num_input_tensors = model.Subgraphs(0).InputsLength()
    num_input_meta = model_meta.SubgraphMetadata(0).InputTensorMetadataLength()
    if num_input_tensors != num_input_meta:
      raise ValueError(
          "The number of input tensors ({0}) should match the number of "
          "input tensor metadata ({1})".format(num_input_tensors,
                                               num_input_meta))
    num_output_tensors = model.Subgraphs(0).OutputsLength()
    num_output_meta = model_meta.SubgraphMetadata(
        0).OutputTensorMetadataLength()
    if num_output_tensors != num_output_meta:
      raise ValueError(
          "The number of output tensors ({0}) should match the number of "
          "output tensor metadata ({1})".format(num_output_tensors,
                                                num_output_meta))


class _MetadataPopulatorWithBuffer(MetadataPopulator):
  """Subclass of MetadataPopulator that populates metadata to a model buffer.

  This class is used to populate metadata into a in-memory model buffer. As we
  use Zip API to concatenate associated files after tflite model file, the
  populating operation is developed based on a model file. For in-memory model
  buffer, we create a tempfile to serve the populating operation. This class is
  then used to generate this tempfile, and delete the file when the
  MetadataPopulator object is deleted.
  """

  def __init__(self, model_buf):
    """Constructor for _MetadataPopulatorWithBuffer.

    Args:
      model_buf: TensorFlow Lite model buffer in bytearray.

    Raises:
      ValueError: model_buf is empty.
      ValueError: model_buf does not have the expected flatbuffer identifier.
    """
    if not model_buf:
      raise ValueError("model_buf cannot be empty.")

    with tempfile.NamedTemporaryFile() as temp:
      model_file = temp.name

    with _open_file(model_file, "wb") as f:
      f.write(model_buf)

    super().__init__(model_file)

  def __del__(self):
    """Destructor of _MetadataPopulatorWithBuffer.

    Deletes the tempfile.
    """
    if os.path.exists(self._model_file):
      os.remove(self._model_file)


class MetadataDisplayer(object):
  """Displays metadata and associated file info in human-readable format."""

  def __init__(self, model_buffer, metadata_buffer, associated_file_list):
    """Constructor for MetadataDisplayer.

    Args:
      model_buffer: valid buffer of the model file.
      metadata_buffer: valid buffer of the metadata file.
      associated_file_list: list of associate files in the model file.
    """
    _assert_model_buffer_identifier(model_buffer)
    _assert_metadata_buffer_identifier(metadata_buffer)
    self._model_buffer = model_buffer
    self._metadata_buffer = metadata_buffer
    self._associated_file_list = associated_file_list

  @classmethod
  def with_model_file(cls, model_file):
    """Creates a MetadataDisplayer object for the model file.

    Args:
      model_file: valid path to a TensorFlow Lite model file.

    Returns:
      MetadataDisplayer object.

    Raises:
      IOError: File not found.
      ValueError: The model does not have metadata.
    """
    _assert_file_exist(model_file)
    with _open_file(model_file, "rb") as f:
      return cls.with_model_buffer(f.read())

  @classmethod
  def with_model_buffer(cls, model_buffer):
    """Creates a MetadataDisplayer object for a file buffer.

    Args:
      model_buffer: TensorFlow Lite model buffer in bytearray.

    Returns:
      MetadataDisplayer object.
    """
    if not model_buffer:
      raise ValueError("model_buffer cannot be empty.")
    metadata_buffer = get_metadata_buffer(model_buffer)
    if not metadata_buffer:
      raise ValueError("The model does not have metadata.")
    associated_file_list = cls._parse_packed_associted_file_list(model_buffer)
    return cls(model_buffer, metadata_buffer, associated_file_list)

  def get_associated_file_buffer(self, filename):
    """Get the specified associated file content in bytearray.

    Args:
      filename: name of the file to be extracted.

    Returns:
      The file content in bytearray.

    Raises:
      ValueError: if the file does not exist in the model.
    """
    if filename not in self._associated_file_list:
      raise ValueError(
          "The file, {}, does not exist in the model.".format(filename))

    with _open_as_zipfile(io.BytesIO(self._model_buffer)) as zf:
      return zf.read(filename)

  def get_metadata_buffer(self):
    """Get the metadata buffer in bytearray out from the model."""
    return copy.deepcopy(self._metadata_buffer)

  def get_metadata_json(self):
    """Converts the metadata into a json string."""
    return convert_to_json(self._metadata_buffer)

  def get_packed_associated_file_list(self):
    """Returns a list of associated files that are packed in the model.

    Returns:
      A name list of associated files.
    """
    return copy.deepcopy(self._associated_file_list)

  @staticmethod
  def _parse_packed_associted_file_list(model_buf):
    """Gets a list of associated files packed to the model file.

    Args:
      model_buf: valid file buffer.

    Returns:
      List of packed associated files.
    """

    try:
      with _open_as_zipfile(io.BytesIO(model_buf)) as zf:
        return zf.namelist()
    except zipfile.BadZipFile:
      return []


def _get_custom_metadata(metadata_buffer: bytes, name: str):
  """Gets the custom metadata in metadata_buffer based on the name.

  Args:
    metadata_buffer: valid metadata buffer in bytes.
    name: custom metadata name.

  Returns:
    Index of custom metadata, custom metadata flatbuffer. Returns (None, None)
    if the custom metadata is not found.
  """
  model_metadata = _metadata_fb.ModelMetadata.GetRootAs(metadata_buffer)
  subgraph = model_metadata.SubgraphMetadata(0)
  if subgraph is None or subgraph.CustomMetadataIsNone():
    return None, None

  for i in range(subgraph.CustomMetadataLength()):
    custom_metadata = subgraph.CustomMetadata(i)
    if custom_metadata.Name().decode("utf-8") == name:
      return i, custom_metadata.DataAsNumpy().tobytes()
  return None, None


# Create an individual method for getting the metadata json file, so that it can
# be used as a standalone util.
def convert_to_json(
    metadata_buffer, custom_metadata_schema: Optional[Dict[str, str]] = None
) -> str:
  """Converts the metadata into a json string.

  Args:
    metadata_buffer: valid metadata buffer in bytes.
    custom_metadata_schema: A dict of custom metadata schema, in which key is
      custom metadata name [1], value is the filepath that defines custom
      metadata schema. For instance, custom_metadata_schema =
      {"SEGMENTER_METADATA": "metadata/vision_tasks_metadata_schema.fbs"}. [1]:
        https://github.com/google/mediapipe/blob/46b5c4012d2ef76c9d92bb0d88a6b107aee83814/mediapipe/tasks/metadata/metadata_schema.fbs#L612

  Returns:
    Metadata in JSON format.

  Raises:
    ValueError: error occurred when parsing the metadata schema file.
  """
  opt = _pywrap_flatbuffers.IDLOptions()
  opt.strict_json = True
  parser = _pywrap_flatbuffers.Parser(opt)
  with _open_file(_FLATC_TFLITE_METADATA_SCHEMA_FILE) as f:
    metadata_schema_content = f.read()
  if not parser.parse(metadata_schema_content):
    raise ValueError("Cannot parse metadata schema. Reason: " + parser.error)
  # Json content which may contain binary custom metadata.
  raw_json_content = _pywrap_flatbuffers.generate_text(parser, metadata_buffer)
  if not custom_metadata_schema:
    return raw_json_content

  json_data = json.loads(raw_json_content)
  # Gets the custom metadata by name and parse the binary custom metadata into
  # human readable json content.
  for name, schema_file in custom_metadata_schema.items():
    idx, custom_metadata = _get_custom_metadata(metadata_buffer, name)
    if not custom_metadata:
      logging.info(
          "No custom metadata with name %s in metadata flatbuffer.", name
      )
      continue
    _assert_file_exist(schema_file)
    with _open_file(schema_file, "rb") as f:
      custom_metadata_schema_content = f.read()
    if not parser.parse(custom_metadata_schema_content):
      raise ValueError(
          "Cannot parse custom metadata schema. Reason: " + parser.error
      )
    custom_metadata_json = _pywrap_flatbuffers.generate_text(
        parser, custom_metadata
    )
    json_meta = json_data["subgraph_metadata"][0]["custom_metadata"][idx]
    json_meta["name"] = name
    json_meta["data"] = json.loads(custom_metadata_json)
  return json.dumps(json_data, indent=2)


def _assert_file_exist(filename):
  """Checks if a file exists."""
  if not _exists_file(filename):
    raise IOError("File, '{0}', does not exist.".format(filename))


def _assert_model_file_identifier(model_file):
  """Checks if a model file has the expected TFLite schema identifier."""
  _assert_file_exist(model_file)
  with _open_file(model_file, "rb") as f:
    _assert_model_buffer_identifier(f.read())


def _assert_model_buffer_identifier(model_buf):
  if not _schema_fb.Model.ModelBufferHasIdentifier(model_buf, 0):
    raise ValueError(
        "The model provided does not have the expected identifier, and "
        "may not be a valid TFLite model.")


def _assert_metadata_buffer_identifier(metadata_buf):
  """Checks if a metadata buffer has the expected Metadata schema identifier."""
  if not _metadata_fb.ModelMetadata.ModelMetadataBufferHasIdentifier(
      metadata_buf, 0):
    raise ValueError(
        "The metadata buffer does not have the expected identifier, and may not"
        " be a valid TFLite Metadata.")


def get_metadata_buffer(model_buf):
  """Returns the metadata in the model file as a buffer.

  Args:
    model_buf: valid buffer of the model file.

  Returns:
    Metadata buffer. Returns `None` if the model does not have metadata.
  """
  tflite_model = _schema_fb.Model.GetRootAsModel(model_buf, 0)

  # Gets metadata from the model file.
  for i in range(tflite_model.MetadataLength()):
    meta = tflite_model.Metadata(i)
    if meta.Name().decode("utf-8") == MetadataPopulator.METADATA_FIELD_NAME:
      buffer_index = meta.Buffer()
      metadata = tflite_model.Buffers(buffer_index)
      if metadata.DataLength() == 0:
        continue
      return metadata.DataAsNumpy().tobytes()

  return None
