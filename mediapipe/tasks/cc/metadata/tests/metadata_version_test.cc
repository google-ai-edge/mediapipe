/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/cc/metadata/metadata_version.h"

#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace metadata {
namespace {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::Offset;
using ::flatbuffers::Vector;
using ::testing::MatchesRegex;
using ::testing::StrEq;
using ::tflite::AssociatedFile;
using ::tflite::AssociatedFileBuilder;
using ::tflite::AudioPropertiesBuilder;
using ::tflite::BertTokenizerOptionsBuilder;
using ::tflite::ContentBuilder;
using ::tflite::ContentProperties_AudioProperties;
using ::tflite::ModelMetadataBuilder;
using ::tflite::NormalizationOptionsBuilder;
using ::tflite::ProcessUnit;
using ::tflite::ProcessUnitBuilder;
using ::tflite::ProcessUnitOptions_BertTokenizerOptions;
using ::tflite::ProcessUnitOptions_NormalizationOptions;
using ::tflite::ProcessUnitOptions_RegexTokenizerOptions;
using ::tflite::ProcessUnitOptions_SentencePieceTokenizerOptions;
using ::tflite::RegexTokenizerOptionsBuilder;
using ::tflite::SentencePieceTokenizerOptionsBuilder;
using ::tflite::SubGraphMetadata;
using ::tflite::SubGraphMetadataBuilder;
using ::tflite::TensorGroup;
using ::tflite::TensorGroupBuilder;
using ::tflite::TensorMetadata;
using ::tflite::TensorMetadataBuilder;

// Creates Model with metadata with input tensor metadata.
void CreateModelWithMetadata(
    const Offset<Vector<Offset<tflite::TensorMetadata>>>& tensors,
    FlatBufferBuilder& builder) {
  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_input_tensor_metadata(tensors);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionSucceedsWithValidMetadata) {
  // Creates a dummy metadata flatbuffer for test.
  FlatBufferBuilder builder(1024);
  auto name = builder.CreateString("Foo");
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_name(name);
  auto metadata = metadata_builder.Finish();
  FinishModelMetadataBuffer(builder, metadata);

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is well-formed (x.y.z).
  EXPECT_THAT(min_version, MatchesRegex("[0-9]+\\.[0-9]+\\.[0-9]+"));
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionFailsWithInvalidIdentifier) {
  // Creates a dummy metadata flatbuffer without identifier.
  FlatBufferBuilder builder(1024);
  ModelMetadataBuilder metadata_builder(builder);
  auto metadata = metadata_builder.Finish();
  builder.Finish(metadata);

  // Gets the mimimum metadata parser version and triggers error.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteError);
  EXPECT_TRUE(min_version.empty());
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForModelMetadataVocabAssociatedFiles) {
  // Creates a metadata flatbuffer with the field,
  // ModelMetadata.associated_fiels, populated with the vocabulary file type.
  FlatBufferBuilder builder(1024);
  AssociatedFileBuilder associated_file_builder(builder);
  associated_file_builder.add_type(tflite::AssociatedFileType_VOCABULARY);
  auto associated_files = builder.CreateVector(
      std::vector<Offset<AssociatedFile>>{associated_file_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_associated_files(associated_files);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.0.1.
  EXPECT_THAT(min_version, StrEq("1.0.1"));
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForSubGraphMetadataVocabAssociatedFiles) {
  // Creates a metadata flatbuffer with the field,
  // SubGraphMetadata.associated_files, populated with the vocabulary file type.
  FlatBufferBuilder builder(1024);
  AssociatedFileBuilder associated_file_builder(builder);
  associated_file_builder.add_type(tflite::AssociatedFileType_VOCABULARY);
  auto associated_files = builder.CreateVector(
      std::vector<Offset<AssociatedFile>>{associated_file_builder.Finish()});
  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_associated_files(associated_files);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.0.1.
  EXPECT_THAT(min_version, StrEq("1.0.1"));
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForInputMetadataVocabAssociatedFiles) {
  // Creates a metadata flatbuffer with the field,
  // SubGraphMetadata.input_tensor_metadata.associated_fiels, populated with the
  // vocabulary file type.
  FlatBufferBuilder builder(1024);
  AssociatedFileBuilder associated_file_builder(builder);
  associated_file_builder.add_type(tflite::AssociatedFileType_VOCABULARY);
  auto associated_files = builder.CreateVector(
      std::vector<Offset<AssociatedFile>>{associated_file_builder.Finish()});
  TensorMetadataBuilder tensor_builder(builder);
  tensor_builder.add_associated_files(associated_files);
  auto tensors = builder.CreateVector(
      std::vector<Offset<TensorMetadata>>{tensor_builder.Finish()});
  CreateModelWithMetadata(tensors, builder);

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.0.1.
  EXPECT_THAT(min_version, StrEq("1.0.1"));
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForOutputMetadataVocabAssociatedFiles) {
  // Creates a metadata flatbuffer with the field,
  // SubGraphMetadata.output_tensor_metadata.associated_fiels, populated with
  // the vocabulary file type.
  FlatBufferBuilder builder(1024);
  AssociatedFileBuilder associated_file_builder(builder);
  associated_file_builder.add_type(tflite::AssociatedFileType_VOCABULARY);
  auto associated_files = builder.CreateVector(
      std::vector<Offset<AssociatedFile>>{associated_file_builder.Finish()});
  TensorMetadataBuilder tensor_builder(builder);
  tensor_builder.add_associated_files(associated_files);
  auto tensors = builder.CreateVector(
      std::vector<Offset<TensorMetadata>>{tensor_builder.Finish()});
  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_output_tensor_metadata(tensors);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.0.1.
  EXPECT_EQ(min_version, "1.0.1");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForSubGraphMetadataInputProcessUnits) {
  // Creates a metadata flatbuffer with the field,
  // SubGraphMetadata.input_process_units
  FlatBufferBuilder builder(1024);
  NormalizationOptionsBuilder normalization_builder(builder);
  auto normalization = normalization_builder.Finish();
  ProcessUnitBuilder process_unit_builder(builder);
  process_unit_builder.add_options_type(
      ProcessUnitOptions_NormalizationOptions);
  process_unit_builder.add_options(normalization.Union());
  auto process_units = builder.CreateVector(
      std::vector<Offset<ProcessUnit>>{process_unit_builder.Finish()});

  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_input_process_units(process_units);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.1.0.
  EXPECT_EQ(min_version, "1.1.0");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForSubGraphMetadataOutputProcessUnits) {
  // Creates a metadata flatbuffer with the field,
  // SubGraphMetadata.output_process_units
  FlatBufferBuilder builder(1024);
  NormalizationOptionsBuilder normalization_builder(builder);
  auto normalization = normalization_builder.Finish();
  ProcessUnitBuilder process_unit_builder(builder);
  process_unit_builder.add_options_type(
      ProcessUnitOptions_NormalizationOptions);
  process_unit_builder.add_options(normalization.Union());
  auto process_units = builder.CreateVector(
      std::vector<Offset<ProcessUnit>>{process_unit_builder.Finish()});

  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_output_process_units(process_units);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.1.0.
  EXPECT_EQ(min_version, "1.1.0");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForProcessUnitBertTokenizerOptions) {
  // Creates a metadata flatbuffer with the field,
  // ProcessUnitOptions.BertTokenizerOptions
  FlatBufferBuilder builder(1024);
  BertTokenizerOptionsBuilder bert_tokenizer_builder(builder);
  auto bert_tokenizer = bert_tokenizer_builder.Finish();
  ProcessUnitBuilder process_unit_builder(builder);
  process_unit_builder.add_options_type(
      ProcessUnitOptions_BertTokenizerOptions);
  process_unit_builder.add_options(bert_tokenizer.Union());
  auto process_units = builder.CreateVector(
      std::vector<Offset<ProcessUnit>>{process_unit_builder.Finish()});

  TensorMetadataBuilder tensor_builder(builder);
  tensor_builder.add_process_units(process_units);
  auto tensors = builder.CreateVector(
      std::vector<Offset<TensorMetadata>>{tensor_builder.Finish()});
  CreateModelWithMetadata(tensors, builder);

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.1.0.
  EXPECT_EQ(min_version, "1.1.0");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForProcessUnitSentencePieceTokenizer) {
  // Creates a metadata flatbuffer with the field,
  // ProcessUnitOptions.SentencePieceTokenizerOptions
  FlatBufferBuilder builder(1024);
  SentencePieceTokenizerOptionsBuilder sentence_piece_builder(builder);
  auto sentence_piece = sentence_piece_builder.Finish();
  ProcessUnitBuilder process_unit_builder(builder);
  process_unit_builder.add_options_type(
      ProcessUnitOptions_SentencePieceTokenizerOptions);
  process_unit_builder.add_options(sentence_piece.Union());
  auto process_units = builder.CreateVector(
      std::vector<Offset<ProcessUnit>>{process_unit_builder.Finish()});

  TensorMetadataBuilder tensor_builder(builder);
  tensor_builder.add_process_units(process_units);
  auto tensors = builder.CreateVector(
      std::vector<Offset<TensorMetadata>>{tensor_builder.Finish()});
  CreateModelWithMetadata(tensors, builder);

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.1.0.
  EXPECT_EQ(min_version, "1.1.0");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForSubgraphMetadataInputTensorGroup) {
  // Creates a metadata flatbuffer with the field,
  // SubgraphMetadata.input_tensor_group.
  FlatBufferBuilder builder(1024);
  TensorGroupBuilder tensor_group_builder(builder);
  auto tensor_groups = builder.CreateVector(
      std::vector<Offset<TensorGroup>>{tensor_group_builder.Finish()});
  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_input_tensor_groups(tensor_groups);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.2.0.
  EXPECT_EQ(min_version, "1.2.0");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForSubgraphMetadataOutputTensorGroup) {
  // Creates a metadata flatbuffer with the field,
  // SubgraphMetadata.output_tensor_group.
  FlatBufferBuilder builder(1024);
  TensorGroupBuilder tensor_group_builder(builder);
  auto tensor_groups = builder.CreateVector(
      std::vector<Offset<TensorGroup>>{tensor_group_builder.Finish()});
  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_output_tensor_groups(tensor_groups);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.2.0.
  EXPECT_EQ(min_version, "1.2.0");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForProcessUnitRegexTokenizer) {
  // Creates a metadata flatbuffer with the field,
  // ProcessUnitOptions.RegexTokenizerOptions
  FlatBufferBuilder builder(1024);
  RegexTokenizerOptionsBuilder regex_builder(builder);
  auto regex = regex_builder.Finish();
  ProcessUnitBuilder process_unit_builder(builder);
  process_unit_builder.add_options_type(
      ProcessUnitOptions_RegexTokenizerOptions);
  process_unit_builder.add_options(regex.Union());
  auto process_units = builder.CreateVector(
      std::vector<Offset<ProcessUnit>>{process_unit_builder.Finish()});

  SubGraphMetadataBuilder subgraph_builder(builder);
  subgraph_builder.add_input_process_units(process_units);
  auto subgraphs = builder.CreateVector(
      std::vector<Offset<SubGraphMetadata>>{subgraph_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_subgraph_metadata(subgraphs);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.2.1.
  EXPECT_EQ(min_version, "1.2.1");
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForContentPropertiesAudioProperties) {
  // Creates a metadata flatbuffer with the field,
  // ContentProperties.AudioProperties.
  FlatBufferBuilder builder(1024);
  AudioPropertiesBuilder audio_builder(builder);
  auto audio = audio_builder.Finish();
  ContentBuilder content_builder(builder);
  content_builder.add_content_properties_type(
      ContentProperties_AudioProperties);
  content_builder.add_content_properties(audio.Union());
  auto content = content_builder.Finish();
  TensorMetadataBuilder tensor_builder(builder);
  tensor_builder.add_content(content);
  auto tensors = builder.CreateVector(
      std::vector<Offset<TensorMetadata>>{tensor_builder.Finish()});
  CreateModelWithMetadata(tensors, builder);

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.3.0.
  EXPECT_THAT(min_version, StrEq("1.3.0"));
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForModelMetadataScannAssociatedFiles) {
  // Creates a metadata flatbuffer with the field,
  // ModelMetadata.associated_files, populated with the scann file type.
  FlatBufferBuilder builder(1024);
  AssociatedFileBuilder associated_file_builder(builder);
  associated_file_builder.add_type(tflite::AssociatedFileType_SCANN_INDEX_FILE);
  auto associated_files = builder.CreateVector(
      std::vector<Offset<AssociatedFile>>{associated_file_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_associated_files(associated_files);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.4.0.
  EXPECT_THAT(min_version, StrEq("1.4.0"));
}

TEST(MetadataVersionTest,
     GetMinimumMetadataParserVersionForAssociatedFileVersion) {
  // Creates a metadata flatbuffer with the field,
  // AssociatedFile.version.
  FlatBufferBuilder builder(1024);
  auto version = builder.CreateString("v1");
  AssociatedFileBuilder associated_file_builder(builder);
  associated_file_builder.add_version(version);
  auto associated_files = builder.CreateVector(
      std::vector<Offset<AssociatedFile>>{associated_file_builder.Finish()});
  ModelMetadataBuilder metadata_builder(builder);
  metadata_builder.add_associated_files(associated_files);
  FinishModelMetadataBuffer(builder, metadata_builder.Finish());

  // Gets the mimimum metadata parser version.
  std::string min_version;
  EXPECT_EQ(GetMinimumMetadataParserVersion(builder.GetBufferPointer(),
                                            builder.GetSize(), &min_version),
            kTfLiteOk);
  // Validates that the version is exactly 1.4.1.
  EXPECT_THAT(min_version, StrEq("1.4.1"));
}

}  // namespace
}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
