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
"""Tests for mediapipe.tasks.python.metadata.metadata."""

import enum
import os

from absl.testing import absltest
from absl.testing import parameterized
import flatbuffers
import six

from mediapipe.tasks.metadata import metadata_schema_py_generated as _metadata_fb
from mediapipe.tasks.metadata import schema_py_generated as _schema_fb
from mediapipe.tasks.python.metadata import metadata as _metadata
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/tasks/testdata/metadata"


class Tokenizer(enum.Enum):
  BERT_TOKENIZER = 0
  SENTENCE_PIECE = 1


class TensorType(enum.Enum):
  INPUT = 0
  OUTPUT = 1


def _read_file(file_name, mode="rb"):
  with open(file_name, mode) as f:
    return f.read()


class MetadataTest(parameterized.TestCase):

  def setUp(self):
    super(MetadataTest, self).setUp()
    self._invalid_model_buf = None
    self._invalid_file = "not_existed_file"
    self._model_buf = self._create_model_buf()
    self._model_file = self.create_tempfile().full_path
    with open(self._model_file, "wb") as f:
      f.write(self._model_buf)
    self._metadata_file = self._create_metadata_file()
    self._metadata_file_with_version = self._create_metadata_file_with_version(
        self._metadata_file, "1.0.0")
    self._file1 = self.create_tempfile("file1").full_path
    self._file2 = self.create_tempfile("file2").full_path
    self._file2_content = b"file2_content"
    with open(self._file2, "wb") as f:
      f.write(self._file2_content)
    self._file3 = self.create_tempfile("file3").full_path

  def _create_model_buf(self):
    # Create a model with two inputs and one output, which matches the metadata
    # created by _create_metadata_file().
    metadata_field = _schema_fb.MetadataT()
    subgraph = _schema_fb.SubGraphT()
    subgraph.inputs = [0, 1]
    subgraph.outputs = [2]

    metadata_field.name = "meta"
    buffer_field = _schema_fb.BufferT()
    model = _schema_fb.ModelT()
    model.subgraphs = [subgraph]
    # Creates the metadata and buffer fields for testing purposes.
    model.metadata = [metadata_field, metadata_field]
    model.buffers = [buffer_field, buffer_field, buffer_field]
    model_builder = flatbuffers.Builder(0)
    model_builder.Finish(
        model.Pack(model_builder),
        _metadata.MetadataPopulator.TFLITE_FILE_IDENTIFIER)
    return model_builder.Output()

  def _create_metadata_file(self):
    associated_file1 = _metadata_fb.AssociatedFileT()
    associated_file1.name = b"file1"
    associated_file2 = _metadata_fb.AssociatedFileT()
    associated_file2.name = b"file2"
    self.expected_recorded_files = [
        six.ensure_str(associated_file1.name),
        six.ensure_str(associated_file2.name)
    ]

    input_meta = _metadata_fb.TensorMetadataT()
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.associatedFiles = [associated_file2]
    subgraph = _metadata_fb.SubGraphMetadataT()
    # Create a model with two inputs and one output.
    subgraph.inputTensorMetadata = [input_meta, input_meta]
    subgraph.outputTensorMetadata = [output_meta]

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "Mobilenet_quantized"
    model_meta.associatedFiles = [associated_file1]
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

    metadata_file = self.create_tempfile().full_path
    with open(metadata_file, "wb") as f:
      f.write(b.Output())
    return metadata_file

  def _create_model_buffer_with_wrong_identifier(self):
    wrong_identifier = b"widn"
    model = _schema_fb.ModelT()
    model_builder = flatbuffers.Builder(0)
    model_builder.Finish(model.Pack(model_builder), wrong_identifier)
    return model_builder.Output()

  def _create_metadata_buffer_with_wrong_identifier(self):
    # Creates a metadata with wrong identifier
    wrong_identifier = b"widn"
    metadata = _metadata_fb.ModelMetadataT()
    metadata_builder = flatbuffers.Builder(0)
    metadata_builder.Finish(metadata.Pack(metadata_builder), wrong_identifier)
    return metadata_builder.Output()

  def _populate_metadata_with_identifier(self, model_buf, metadata_buf,
                                         identifier):
    # For testing purposes only. MetadataPopulator cannot populate metadata with
    # wrong identifiers.
    model = _schema_fb.ModelT.InitFromObj(
        _schema_fb.Model.GetRootAsModel(model_buf, 0))
    buffer_field = _schema_fb.BufferT()
    buffer_field.data = metadata_buf
    model.buffers = [buffer_field]
    # Creates a new metadata field.
    metadata_field = _schema_fb.MetadataT()
    metadata_field.name = _metadata.MetadataPopulator.METADATA_FIELD_NAME
    metadata_field.buffer = len(model.buffers) - 1
    model.metadata = [metadata_field]
    b = flatbuffers.Builder(0)
    b.Finish(model.Pack(b), identifier)
    return b.Output()

  def _create_metadata_file_with_version(self, metadata_file, min_version):
    # Creates a new metadata file with the specified min_version for testing
    # purposes.
    metadata_buf = bytearray(_read_file(metadata_file))

    metadata = _metadata_fb.ModelMetadataT.InitFromObj(
        _metadata_fb.ModelMetadata.GetRootAsModelMetadata(metadata_buf, 0))
    metadata.minParserVersion = min_version

    b = flatbuffers.Builder(0)
    b.Finish(
        metadata.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

    metadata_file_with_version = self.create_tempfile().full_path
    with open(metadata_file_with_version, "wb") as f:
      f.write(b.Output())
    return metadata_file_with_version


class MetadataPopulatorTest(MetadataTest):

  def _create_bert_tokenizer(self):
    vocab_file_name = "bert_vocab"
    vocab = _metadata_fb.AssociatedFileT()
    vocab.name = vocab_file_name
    vocab.type = _metadata_fb.AssociatedFileType.VOCABULARY
    tokenizer = _metadata_fb.ProcessUnitT()
    tokenizer.optionsType = _metadata_fb.ProcessUnitOptions.BertTokenizerOptions
    tokenizer.options = _metadata_fb.BertTokenizerOptionsT()
    tokenizer.options.vocabFile = [vocab]
    return tokenizer, [vocab_file_name]

  def _create_sentence_piece_tokenizer(self):
    sp_model_name = "sp_model"
    vocab_file_name = "sp_vocab"
    sp_model = _metadata_fb.AssociatedFileT()
    sp_model.name = sp_model_name
    vocab = _metadata_fb.AssociatedFileT()
    vocab.name = vocab_file_name
    vocab.type = _metadata_fb.AssociatedFileType.VOCABULARY
    tokenizer = _metadata_fb.ProcessUnitT()
    tokenizer.optionsType = (
        _metadata_fb.ProcessUnitOptions.SentencePieceTokenizerOptions)
    tokenizer.options = _metadata_fb.SentencePieceTokenizerOptionsT()
    tokenizer.options.sentencePieceModel = [sp_model]
    tokenizer.options.vocabFile = [vocab]
    return tokenizer, [sp_model_name, vocab_file_name]

  def _create_tokenizer(self, tokenizer_type):
    if tokenizer_type is Tokenizer.BERT_TOKENIZER:
      return self._create_bert_tokenizer()
    elif tokenizer_type is Tokenizer.SENTENCE_PIECE:
      return self._create_sentence_piece_tokenizer()
    else:
      raise ValueError(
          "The tokenizer type, {0}, is unsupported.".format(tokenizer_type))

  def _create_tempfiles(self, file_names):
    tempfiles = []
    for name in file_names:
      tempfiles.append(self.create_tempfile(name).full_path)
    return tempfiles

  def _create_model_meta_with_subgraph_meta(self, subgraph_meta):
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgraph_meta]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    return b.Output()

  def testToValidModelFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    self.assertIsInstance(populator, _metadata.MetadataPopulator)

  def testToInvalidModelFile(self):
    with self.assertRaises(IOError) as error:
      _metadata.MetadataPopulator.with_model_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testToValidModelBuffer(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    self.assertIsInstance(populator, _metadata.MetadataPopulator)

  def testToInvalidModelBuffer(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataPopulator.with_model_buffer(self._invalid_model_buf)
    self.assertEqual("model_buf cannot be empty.", str(error.exception))

  def testToModelBufferWithWrongIdentifier(self):
    model_buf = self._create_model_buffer_with_wrong_identifier()
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataPopulator.with_model_buffer(model_buf)
    self.assertEqual(
        "The model provided does not have the expected identifier, and "
        "may not be a valid TFLite model.", str(error.exception))

  def testSinglePopulateAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    populator.load_associated_files([self._file1])
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [os.path.basename(self._file1)]
    self.assertEqual(set(packed_files), set(expected_packed_files))

  def testRepeatedPopulateAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_associated_files([self._file1, self._file2])
    # Loads file2 multiple times.
    populator.load_associated_files([self._file2])
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertLen(packed_files, 2)
    self.assertEqual(set(packed_files), set(expected_packed_files))

    # Check if the model buffer read from file is the same as that read from
    # get_model_buffer().
    model_buf_from_file = _read_file(self._model_file)
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateInvalidAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(IOError) as error:
      populator.load_associated_files([self._invalid_file])
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testPopulatePackedAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    populator.load_associated_files([self._file1])
    populator.populate()
    with self.assertRaises(ValueError) as error:
      populator.load_associated_files([self._file1])
      populator.populate()
    self.assertEqual(
        "File, '{0}', has already been packed.".format(
            os.path.basename(self._file1)), str(error.exception))

  def testLoadAssociatedFileBuffers(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    file_buffer = _read_file(self._file1)
    populator.load_associated_file_buffers({self._file1: file_buffer})
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [os.path.basename(self._file1)]
    self.assertEqual(set(packed_files), set(expected_packed_files))

  def testRepeatedLoadAssociatedFileBuffers(self):
    file_buffer1 = _read_file(self._file1)
    file_buffer2 = _read_file(self._file2)
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)

    populator.load_associated_file_buffers({
        self._file1: file_buffer1,
        self._file2: file_buffer2
    })
    # Loads file2 multiple times.
    populator.load_associated_file_buffers({self._file2: file_buffer2})
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertEqual(set(packed_files), set(expected_packed_files))

    # Check if the model buffer read from file is the same as that read from
    # get_model_buffer().
    model_buf_from_file = _read_file(self._model_file)
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testLoadPackedAssociatedFileBuffersFails(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    file_buffer = _read_file(self._file1)
    populator.load_associated_file_buffers({self._file1: file_buffer})
    populator.populate()

    # Load file1 again should fail.
    with self.assertRaises(ValueError) as error:
      populator.load_associated_file_buffers({self._file1: file_buffer})
      populator.populate()
    self.assertEqual(
        "File, '{0}', has already been packed.".format(
            os.path.basename(self._file1)), str(error.exception))

  def testGetPackedAssociatedFileList(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    packed_files = populator.get_packed_associated_file_list()
    self.assertEqual(packed_files, [])

  def testPopulateMetadataFileToEmptyModelFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()

    model_buf_from_file = _read_file(self._model_file)
    model = _schema_fb.Model.GetRootAsModel(model_buf_from_file, 0)
    # self._model_file already has two elements in the metadata field, so the
    # populated TFLite metadata will be the third element.
    metadata_field = model.Metadata(2)
    self.assertEqual(
        six.ensure_str(metadata_field.Name()),
        six.ensure_str(_metadata.MetadataPopulator.METADATA_FIELD_NAME))

    buffer_index = metadata_field.Buffer()
    buffer_data = model.Buffers(buffer_index)
    metadata_buf_np = buffer_data.DataAsNumpy()
    metadata_buf = metadata_buf_np.tobytes()
    expected_metadata_buf = bytearray(
        _read_file(self._metadata_file_with_version))
    self.assertEqual(metadata_buf, expected_metadata_buf)

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

    # Up to now, we've proved the correctness of the model buffer that read from
    # file. Then we'll test if get_model_buffer() gives the same model buffer.
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateMetadataFileWithoutAssociatedFiles(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1])
    # Suppose to populate self._file2, because it is recorded in the metadata.
    with self.assertRaises(ValueError) as error:
      populator.populate()
    self.assertEqual(("File, '{0}', is recorded in the metadata, but has "
                      "not been loaded into the populator.").format(
                          os.path.basename(self._file2)), str(error.exception))

  def testPopulateMetadataBufferWithWrongIdentifier(self):
    metadata_buf = self._create_metadata_buffer_with_wrong_identifier()
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(metadata_buf)
    self.assertEqual(
        "The metadata buffer does not have the expected identifier, and may not"
        " be a valid TFLite Metadata.", str(error.exception))

  def _assert_golden_metadata(self, model_file):
    model_buf_from_file = _read_file(model_file)
    model = _schema_fb.Model.GetRootAsModel(model_buf_from_file, 0)
    # There are two elements in model.Metadata array before the population.
    # Metadata should be packed to the third element in the array.
    metadata_field = model.Metadata(2)
    self.assertEqual(
        six.ensure_str(metadata_field.Name()),
        six.ensure_str(_metadata.MetadataPopulator.METADATA_FIELD_NAME))

    buffer_index = metadata_field.Buffer()
    buffer_data = model.Buffers(buffer_index)
    metadata_buf_np = buffer_data.DataAsNumpy()
    metadata_buf = metadata_buf_np.tobytes()
    expected_metadata_buf = bytearray(
        _read_file(self._metadata_file_with_version))
    self.assertEqual(metadata_buf, expected_metadata_buf)

  def testPopulateMetadataFileToModelWithMetadataAndAssociatedFiles(self):
    # First, creates a dummy metadata different from self._metadata_file. It
    # needs to have the same input/output tensor numbers as self._model_file.
    # Populates it and the associated files into the model.
    input_meta = _metadata_fb.TensorMetadataT()
    output_meta = _metadata_fb.TensorMetadataT()
    subgraph = _metadata_fb.SubGraphMetadataT()
    # Create a model with two inputs and one output.
    subgraph.inputTensorMetadata = [input_meta, input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # Populate the metadata.
    populator1 = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator1.load_metadata_buffer(metadata_buf)
    populator1.load_associated_files([self._file1, self._file2])
    populator1.populate()

    # Then, populate the metadata again.
    populator2 = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator2.load_metadata_file(self._metadata_file)
    populator2.populate()

    # Test if the metadata is populated correctly.
    self._assert_golden_metadata(self._model_file)

  def testPopulateMetadataFileToModelFileWithMetadataAndBufFields(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()

    # Tests if the metadata is populated correctly.
    self._assert_golden_metadata(self._model_file)

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

    # Up to now, we've proved the correctness of the model buffer that read from
    # file. Then we'll test if get_model_buffer() gives the same model buffer.
    model_buf_from_file = _read_file(self._model_file)
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateInvalidMetadataFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(IOError) as error:
      populator.load_metadata_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testPopulateInvalidMetadataBuffer(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer([])
    self.assertEqual("The metadata to be populated is empty.",
                     str(error.exception))

  def testGetModelBufferBeforePopulatingData(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    model_buf = populator.get_model_buffer()
    expected_model_buf = self._model_buf
    self.assertEqual(model_buf, expected_model_buf)

  def testLoadMetadataBufferWithNoSubgraphMetadataThrowsException(self):
    # Create a dummy metadata without Subgraph.
    model_meta = _metadata_fb.ModelMetadataT()
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    meta_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(meta_buf)
    self.assertEqual(
        "The number of SubgraphMetadata should be exactly one, but got 0.",
        str(error.exception))

  def testLoadMetadataBufferWithWrongInputMetaNumberThrowsException(self):
    # Create a dummy metadata with no input tensor metadata, while the expected
    # number is 2.
    output_meta = _metadata_fb.TensorMetadataT()
    subgprah_meta = _metadata_fb.SubGraphMetadataT()
    subgprah_meta.outputTensorMetadata = [output_meta]
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgprah_meta]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    meta_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(meta_buf)
    self.assertEqual(
        ("The number of input tensors (2) should match the number of "
         "input tensor metadata (0)"), str(error.exception))

  def testLoadMetadataBufferWithWrongOutputMetaNumberThrowsException(self):
    # Create a dummy metadata with no output tensor metadata, while the expected
    # number is 1.
    input_meta = _metadata_fb.TensorMetadataT()
    subgprah_meta = _metadata_fb.SubGraphMetadataT()
    subgprah_meta.inputTensorMetadata = [input_meta, input_meta]
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgprah_meta]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    meta_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(meta_buf)
    self.assertEqual(
        ("The number of output tensors (1) should match the number of "
         "output tensor metadata (0)"), str(error.exception))

  def testLoadMetadataAndAssociatedFilesShouldSucceed(self):
    # Create a src model with metadata and two associated files.
    src_model_buf = self._create_model_buf()
    populator_src = _metadata.MetadataPopulator.with_model_buffer(src_model_buf)
    populator_src.load_metadata_file(self._metadata_file)
    populator_src.load_associated_files([self._file1, self._file2])
    populator_src.populate()

    # Create a model to be populated with the metadata and files from
    # src_model_buf.
    dst_model_buf = self._create_model_buf()
    populator_dst = _metadata.MetadataPopulator.with_model_buffer(dst_model_buf)
    populator_dst.load_metadata_and_associated_files(
        populator_src.get_model_buffer())
    populator_dst.populate()

    # Test if the metadata and associated files are populated correctly.
    dst_model_file = self.create_tempfile().full_path
    with open(dst_model_file, "wb") as f:
      f.write(populator_dst.get_model_buffer())
    self._assert_golden_metadata(dst_model_file)

    recorded_files = populator_dst.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

  def testLoadMetadataAndAssociatedFilesShouldSucceedOnEmptyMetadata(self):
    # When the user hasn't specified the metadata, but only the associated
    # files, an empty metadata buffer is created. Previously, it caused an
    # exception when reading.

    # Create a source model with two associated files but no metadata.
    src_model_buf = self._create_model_buf()
    populator_src = _metadata.MetadataPopulator.with_model_buffer(src_model_buf)
    populator_src.load_associated_files([self._file1, self._file2])
    populator_src.populate()

    # Create a model to be populated with the files from `src_model_buf`.
    dst_model_buf = self._create_model_buf()
    populator_dst = _metadata.MetadataPopulator.with_model_buffer(dst_model_buf)
    populator_dst.load_metadata_and_associated_files(
        populator_src.get_model_buffer())
    populator_dst.populate()

    # Test if the metadata and associated files are populated correctly.
    packed_files = populator_dst.get_packed_associated_file_list()
    self.assertEqual(set(packed_files), set(self.expected_recorded_files))

  @parameterized.named_parameters(
      {
          "testcase_name": "InputTensorWithBert",
          "tensor_type": TensorType.INPUT,
          "tokenizer_type": Tokenizer.BERT_TOKENIZER
      }, {
          "testcase_name": "OutputTensorWithBert",
          "tensor_type": TensorType.OUTPUT,
          "tokenizer_type": Tokenizer.BERT_TOKENIZER
      }, {
          "testcase_name": "InputTensorWithSentencePiece",
          "tensor_type": TensorType.INPUT,
          "tokenizer_type": Tokenizer.SENTENCE_PIECE
      }, {
          "testcase_name": "OutputTensorWithSentencePiece",
          "tensor_type": TensorType.OUTPUT,
          "tokenizer_type": Tokenizer.SENTENCE_PIECE
      })
  def testGetRecordedAssociatedFileListWithSubgraphTensor(
      self, tensor_type, tokenizer_type):
    # Creates a metadata with the tokenizer in the tensor process units.
    tokenizer, expected_files = self._create_tokenizer(tokenizer_type)

    # Create the tensor with process units.
    tensor = _metadata_fb.TensorMetadataT()
    tensor.processUnits = [tokenizer]

    # Create the subgrah with the tensor.
    subgraph = _metadata_fb.SubGraphMetadataT()
    dummy_tensor_meta = _metadata_fb.TensorMetadataT()
    subgraph.outputTensorMetadata = [dummy_tensor_meta]
    if tensor_type is TensorType.INPUT:
      subgraph.inputTensorMetadata = [tensor, dummy_tensor_meta]
      subgraph.outputTensorMetadata = [dummy_tensor_meta]
    elif tensor_type is TensorType.OUTPUT:
      subgraph.inputTensorMetadata = [dummy_tensor_meta, dummy_tensor_meta]
      subgraph.outputTensorMetadata = [tensor]
    else:
      raise ValueError(
          "The tensor type, {0}, is unsupported.".format(tensor_type))

    # Create a model metadata with the subgraph metadata
    meta_buffer = self._create_model_meta_with_subgraph_meta(subgraph)

    # Creates the tempfiles.
    tempfiles = self._create_tempfiles(expected_files)

    # Creates the MetadataPopulator object.
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_buffer(meta_buffer)
    populator.load_associated_files(tempfiles)
    populator.populate()

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(expected_files))

  @parameterized.named_parameters(
      {
          "testcase_name": "InputTensorWithBert",
          "tensor_type": TensorType.INPUT,
          "tokenizer_type": Tokenizer.BERT_TOKENIZER
      }, {
          "testcase_name": "OutputTensorWithBert",
          "tensor_type": TensorType.OUTPUT,
          "tokenizer_type": Tokenizer.BERT_TOKENIZER
      }, {
          "testcase_name": "InputTensorWithSentencePiece",
          "tensor_type": TensorType.INPUT,
          "tokenizer_type": Tokenizer.SENTENCE_PIECE
      }, {
          "testcase_name": "OutputTensorWithSentencePiece",
          "tensor_type": TensorType.OUTPUT,
          "tokenizer_type": Tokenizer.SENTENCE_PIECE
      })
  def testGetRecordedAssociatedFileListWithSubgraphProcessUnits(
      self, tensor_type, tokenizer_type):
    # Creates a metadata with the tokenizer in the subgraph process units.
    tokenizer, expected_files = self._create_tokenizer(tokenizer_type)

    # Create the subgraph with process units.
    subgraph = _metadata_fb.SubGraphMetadataT()
    if tensor_type is TensorType.INPUT:
      subgraph.inputProcessUnits = [tokenizer]
    elif tensor_type is TensorType.OUTPUT:
      subgraph.outputProcessUnits = [tokenizer]
    else:
      raise ValueError(
          "The tensor type, {0}, is unsupported.".format(tensor_type))

    # Creates the input and output tensor meta to match self._model_file.
    dummy_tensor_meta = _metadata_fb.TensorMetadataT()
    subgraph.inputTensorMetadata = [dummy_tensor_meta, dummy_tensor_meta]
    subgraph.outputTensorMetadata = [dummy_tensor_meta]

    # Create a model metadata with the subgraph metadata
    meta_buffer = self._create_model_meta_with_subgraph_meta(subgraph)

    # Creates the tempfiles.
    tempfiles = self._create_tempfiles(expected_files)

    # Creates the MetadataPopulator object.
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_buffer(meta_buffer)
    populator.load_associated_files(tempfiles)
    populator.populate()

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(expected_files))

  def testPopulatedFullPathAssociatedFileShouldSucceed(self):
    # Create AssociatedFileT using the full path file name.
    associated_file = _metadata_fb.AssociatedFileT()
    associated_file.name = self._file1

    # Create model metadata with the associated file.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.associatedFiles = [associated_file]
    # Creates the input and output tensor metadata to match self._model_file.
    dummy_tensor = _metadata_fb.TensorMetadataT()
    subgraph.inputTensorMetadata = [dummy_tensor, dummy_tensor]
    subgraph.outputTensorMetadata = [dummy_tensor]
    md_buffer = self._create_model_meta_with_subgraph_meta(subgraph)

    # Populate the metadata to a model.
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_buffer(md_buffer)
    populator.load_associated_files([self._file1])
    populator.populate()

    # The recorded file name in metadata should only contain file basename; file
    # directory should not be included.
    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set([os.path.basename(self._file1)]))


class MetadataDisplayerTest(MetadataTest):

  def setUp(self):
    super(MetadataDisplayerTest, self).setUp()
    self._model_with_meta_file = (
        self._create_model_with_metadata_and_associated_files())

  def _create_model_with_metadata_and_associated_files(self):
    model_buf = self._create_model_buf()
    model_file = self.create_tempfile().full_path
    with open(model_file, "wb") as f:
      f.write(model_buf)

    populator = _metadata.MetadataPopulator.with_model_file(model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()
    return model_file

  def testLoadModelBufferMetadataBufferWithWrongIdentifierThrowsException(self):
    model_buf = self._create_model_buffer_with_wrong_identifier()
    metadata_buf = self._create_metadata_buffer_with_wrong_identifier()
    model_buf = self._populate_metadata_with_identifier(
        model_buf, metadata_buf,
        _metadata.MetadataPopulator.TFLITE_FILE_IDENTIFIER)
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(model_buf)
    self.assertEqual(
        "The metadata buffer does not have the expected identifier, and may not"
        " be a valid TFLite Metadata.", str(error.exception))

  def testLoadModelBufferModelBufferWithWrongIdentifierThrowsException(self):
    model_buf = self._create_model_buffer_with_wrong_identifier()
    metadata_file = self._create_metadata_file()
    wrong_identifier = b"widn"
    metadata_buf = bytearray(_read_file(metadata_file))
    model_buf = self._populate_metadata_with_identifier(model_buf, metadata_buf,
                                                        wrong_identifier)
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(model_buf)
    self.assertEqual(
        "The model provided does not have the expected identifier, and "
        "may not be a valid TFLite model.", str(error.exception))

  def testLoadModelFileInvalidModelFileThrowsException(self):
    with self.assertRaises(IOError) as error:
      _metadata.MetadataDisplayer.with_model_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testLoadModelFileModelWithoutMetadataThrowsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_file(self._model_file)
    self.assertEqual("The model does not have metadata.", str(error.exception))

  def testLoadModelFileModelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    self.assertIsInstance(displayer, _metadata.MetadataDisplayer)

  def testLoadModelBufferInvalidModelBufferThrowsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(_read_file(self._file1))
    self.assertEqual("model_buffer cannot be empty.", str(error.exception))

  def testLoadModelBufferModelWithOutMetadataThrowsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(self._create_model_buf())
    self.assertEqual("The model does not have metadata.", str(error.exception))

  def testLoadModelBufferModelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_buffer(
        _read_file(self._model_with_meta_file))
    self.assertIsInstance(displayer, _metadata.MetadataDisplayer)

  def testGetAssociatedFileBufferShouldSucceed(self):
    # _model_with_meta_file contains file1 and file2.
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)

    actual_content = displayer.get_associated_file_buffer("file2")
    self.assertEqual(actual_content, self._file2_content)

  def testGetAssociatedFileBufferFailsWithNonExistentFile(self):
    # _model_with_meta_file contains file1 and file2.
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)

    non_existent_file = "non_existent_file"
    with self.assertRaises(ValueError) as error:
      displayer.get_associated_file_buffer(non_existent_file)
    self.assertEqual(
        "The file, {}, does not exist in the model.".format(non_existent_file),
        str(error.exception))

  def testGetMetadataBufferShouldSucceed(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    actual_buffer = displayer.get_metadata_buffer()
    actual_json = _metadata.convert_to_json(actual_buffer)

    # Verifies the generated json file.
    golden_json_file_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, "golden_json.json"))
    with open(golden_json_file_path, "r") as f:
      expected = f.read()
    self.assertEqual(actual_json, expected)

  def testGetMetadataJsonModelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    actual = displayer.get_metadata_json()

    # Verifies the generated json file.
    golden_json_file_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, "golden_json.json"))
    expected = _read_file(golden_json_file_path, "r")
    self.assertEqual(actual, expected)

  def testGetPackedAssociatedFileListModelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    packed_files = displayer.get_packed_associated_file_list()

    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertLen(
        packed_files, 2,
        "The following two associated files packed to the model: {0}; {1}"
        .format(expected_packed_files[0], expected_packed_files[1]))
    self.assertEqual(set(packed_files), set(expected_packed_files))


class MetadataUtilTest(MetadataTest):

  def test_convert_to_json_should_succeed(self):
    metadata_buf = _read_file(self._metadata_file_with_version)
    metadata_json = _metadata.convert_to_json(metadata_buf)

    # Verifies the generated json file.
    golden_json_file_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, "golden_json.json"))
    expected = _read_file(golden_json_file_path, "r")
    self.assertEqual(metadata_json, expected)


if __name__ == "__main__":
  absltest.main()
