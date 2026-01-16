/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/model_converter.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/double_array_trie_builder.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/encoder_config_generated.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/sentencepiece_constants.h"
#include "sentencepiece/src/sentencepiece_model.pb.h"  // from @com_google_sentencepiece

namespace mediapipe::tflite_operations::sentencepiece {

std::tuple<std::vector<uint32_t>, std::vector<int8_t>>
DecodePrecompiledCharsmap(
    const ::sentencepiece::NormalizerSpec& normalizer_spec) {
  // This function "undoes" encoding done by
  // sentencepiece::normalizer::Normalizer::EncodePrecompiledCharsMap.
  const char* precompiled_map = normalizer_spec.precompiled_charsmap().data();
  const uint32_t trie_size =
      *reinterpret_cast<const uint32_t*>(precompiled_map);
  const uint32_t* trie_ptr =
      reinterpret_cast<const uint32_t*>(precompiled_map + sizeof(uint32_t));
  const int8_t* normalized_ptr = reinterpret_cast<const int8_t*>(
      precompiled_map + sizeof(uint32_t) + trie_size);
  const int normalized_size = normalizer_spec.precompiled_charsmap().length() -
                              sizeof(uint32_t) - trie_size;
  return std::make_tuple(
      std::vector<uint32_t>(trie_ptr, trie_ptr + trie_size / sizeof(uint32_t)),
      std::vector<int8_t>(normalized_ptr, normalized_ptr + normalized_size));
}

absl::StatusOr<std::string> ConvertSentencepieceModelToFlatBuffer(
    const std::string& model_config_str, int encoding_offset) {
  ::sentencepiece::ModelProto model_config;
  if (!model_config.ParseFromString(model_config_str)) {
    return absl::InvalidArgumentError(
        "Invalid configuration, can't parse SentencePiece model config " +
        model_config.InitializationErrorString());
  }
  // Convert sentencepieces.
  std::vector<std::string> pieces;
  pieces.reserve(model_config.pieces_size());
  std::vector<float> scores;
  scores.reserve(model_config.pieces_size());
  std::vector<int> ids;
  ids.reserve(model_config.pieces_size());
  float min_score = 0.0;
  int index = 0;
  for (const auto& piece : model_config.pieces()) {
    switch (piece.type()) {
      case ::sentencepiece::ModelProto::SentencePiece::NORMAL:
      case ::sentencepiece::ModelProto::SentencePiece::USER_DEFINED:
        pieces.push_back(piece.piece());
        ids.push_back(index);
        if (piece.score() < min_score) {
          min_score = piece.score();
        }
        break;
      case ::sentencepiece::ModelProto::SentencePiece::UNKNOWN:
      case ::sentencepiece::ModelProto::SentencePiece::CONTROL:
        // Ignore unknown and control codes.
        break;
      default:
        return absl::InvalidArgumentError("Invalid SentencePiece piece type " +
                                          piece.piece());
    }
    scores.push_back(piece.score());
    ++index;
  }
  flatbuffers::FlatBufferBuilder builder(1024);
  const auto pieces_trie_vector = builder.CreateVector(BuildTrie(pieces, ids));
  const auto pieces_score_vector = builder.CreateVector(scores);
  TrieBuilder pieces_trie_builder(builder);
  pieces_trie_builder.add_nodes(pieces_trie_vector);
  const auto pieces_trie_fbs = pieces_trie_builder.Finish();

  // Converting normalization.
  const auto normalization =
      DecodePrecompiledCharsmap(model_config.normalizer_spec());
  const auto normalization_trie = std::get<0>(normalization);
  const auto normalization_strings = std::get<1>(normalization);
  const auto normalization_trie_vector =
      builder.CreateVector(normalization_trie);
  TrieBuilder normalization_trie_builder(builder);
  normalization_trie_builder.add_nodes(normalization_trie_vector);
  const auto normalization_trie_fbs = normalization_trie_builder.Finish();
  const auto normalization_strings_fbs =
      builder.CreateVector(normalization_strings);

  EncoderConfigBuilder ecb(builder);
  ecb.add_version(EncoderVersion::EncoderVersion_SENTENCE_PIECE);
  ecb.add_start_code(model_config.trainer_spec().bos_id());
  ecb.add_end_code(model_config.trainer_spec().eos_id());
  ecb.add_unknown_code(model_config.trainer_spec().unk_id());
  ecb.add_unknown_penalty(min_score - kUnkPenalty);
  ecb.add_encoding_offset(encoding_offset);
  ecb.add_pieces(pieces_trie_fbs);
  ecb.add_pieces_scores(pieces_score_vector);
  ecb.add_remove_extra_whitespaces(
      model_config.normalizer_spec().remove_extra_whitespaces());
  ecb.add_add_dummy_prefix(model_config.normalizer_spec().add_dummy_prefix());
  ecb.add_escape_whitespaces(
      model_config.normalizer_spec().escape_whitespaces());
  ecb.add_normalized_prefixes(normalization_trie_fbs);
  ecb.add_normalized_replacements(normalization_strings_fbs);
  FinishEncoderConfigBuffer(builder, ecb.Finish());
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

std::string ConvertSentencepieceModel(const std::string& model_string) {
  const auto result = ConvertSentencepieceModelToFlatBuffer(model_string);
  assert(result.status().ok());
  return result.value();
}

}  // namespace mediapipe::tflite_operations::sentencepiece
