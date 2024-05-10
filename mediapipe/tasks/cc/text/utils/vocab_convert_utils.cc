#include "mediapipe/tasks/cc/text/utils/vocab_convert_utils.h"

#include <fstream>
#include <istream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/resource_util.h"
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "nlohmann/json_fwd.hpp"
#include "sentencepiece/src/builder.h"  // from @com_google_sentencepiece
#include "sentencepiece/src/sentencepiece_model.pb.h"  // from @com_google_sentencepiece

namespace mediapipe {
namespace tasks {
namespace text {
namespace {

using ::nlohmann::json;
using ::sentencepiece::ModelProto;
using ::sentencepiece::NormalizerSpec;
using ::sentencepiece::TrainerSpec;
using ::sentencepiece::normalizer::Builder;

// Loads Hugging Face's `tokenizer_config.json` and `tokenizer.json`. The
// files include the preprocessing and postprocessing steps and the token
// mappings. The loaded jsons are returned as a pair containing
// `tokenizer_config.json` and `tokenizer.json` in the same order.
absl::StatusOr<std::pair<json, json>> LoadHFTokenizerConfigs(
    absl::string_view path) {
  std::string contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::StrCat(path, "/tokenizer_config.json"), &contents));
  auto config_json = json::parse(contents, nullptr, false);
  if (config_json.is_discarded()) {
    return absl::InternalError("Failed to parse tokenizer_config.json");
  }
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::StrCat(path, "/tokenizer.json"), &contents));
  auto tokenizer_json = json::parse(contents);
  if (tokenizer_json.is_discarded()) {
    return absl::InternalError("Failed to parse tokenizer.json");
  }
  return std::make_pair(config_json, tokenizer_json);
}

absl::Status ConfigureNormalizerSpecs(NormalizerSpec* spec) {
  spec->set_add_dummy_prefix(false);
  spec->set_remove_extra_whitespaces(false);
  spec->set_escape_whitespaces(false);
  return absl::OkStatus();
}

absl::Status ConfigureDenormalizerSpecs(NormalizerSpec* spec) {
  spec->set_add_dummy_prefix(false);
  spec->set_remove_extra_whitespaces(false);
  spec->set_escape_whitespaces(false);
  return absl::OkStatus();
}
}  // namespace
absl::Status ConvertHfTokenizer(const std::string& hf_tokenizer,
                                const std::string& output_vocab_path) {
  MP_ASSIGN_OR_RETURN(auto configs, LoadHFTokenizerConfigs(hf_tokenizer));

  ModelProto model_proto;

  MP_RETURN_IF_ERROR(
      ConfigureNormalizerSpecs(model_proto.mutable_normalizer_spec()));
  MP_RETURN_IF_ERROR(
      ConfigureDenormalizerSpecs(model_proto.mutable_denormalizer_spec()));

  // The scores assigned here are heuristic based and only captures the ordering
  // of elements within HF configs. This may not be optimal.
  std::vector<std::string> normal_vocabs(
      configs.second["model"]["vocab"].size());
  for (const auto& [vocab, id] : configs.second["model"]["vocab"].items()) {
    normal_vocabs[id] = vocab;
  }
  std::string unk_token = configs.first.at("unk_token").get<std::string>();
  for (int i = 0; i < normal_vocabs.size(); ++i) {
    auto* sp = model_proto.add_pieces();
    auto vocab = normal_vocabs[i];
    sp->set_type(unk_token == vocab ? ModelProto::SentencePiece::UNKNOWN
                                    : ModelProto::SentencePiece::NORMAL);
    sp->set_piece(vocab);
    sp->set_score(-i);
  }
  const auto& added_tokens = configs.second["added_tokens"];
  for (int i = 0; i < added_tokens.size(); ++i) {
    if (added_tokens[i]["normalized"]) {
      auto vocab = added_tokens[i]["content"];
      auto* sp = model_proto.add_pieces();
      sp->set_type(ModelProto::SentencePiece::USER_DEFINED);
      sp->set_piece(vocab);
      sp->set_score(-(normal_vocabs.size() + i));
    }
  }

  auto* trainer_spec = model_proto.mutable_trainer_spec();
  trainer_spec->set_model_type(TrainerSpec::BPE);
  trainer_spec->set_vocab_size(model_proto.pieces_size());

  absl::string_view output_dir = ::mediapipe::file::Dirname(output_vocab_path);
  if (!::mediapipe::file::IsDirectory(output_dir).ok()) {
    MP_RETURN_IF_ERROR(::mediapipe::file::RecursivelyCreateDir(output_dir));
  }

  MP_RETURN_IF_ERROR(mediapipe::file::SetContents(
      output_vocab_path, model_proto.SerializeAsString()));

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
