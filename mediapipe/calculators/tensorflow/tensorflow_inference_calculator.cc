// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/calculators/tensorflow/tensorflow_inference_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/deps/monotonic_clock.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/status_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"

#if !defined(MEDIAPIPE_MOBILE) && !defined(__APPLE__)
#include "tensorflow/core/profiler/lib/traceme.h"
#endif

namespace tf = ::tensorflow;

namespace mediapipe {

namespace {

constexpr char kRecurrentInitTensorsTag[] = "RECURRENT_INIT_TENSORS";
constexpr char kSessionTag[] = "SESSION";
constexpr char kSessionBundleTag[] = "SESSION_BUNDLE";

// This is a simple implementation of a semaphore using standard C++ libraries.
// It is supposed to be used only by TensorflowInferenceCalculator to throttle
// the concurrent calls of Tensorflow Session::Run. This is useful when multiple
// threads execute the graph (e.g. in a mapreduce type of job) but not to
// overload GPU/TPU/...
class SimpleSemaphore {
 public:
  explicit SimpleSemaphore(uint32_t initial_count) : count_(initial_count) {}
  SimpleSemaphore(const SimpleSemaphore&) = delete;
  SimpleSemaphore(SimpleSemaphore&&) = delete;

  // Acquires the semaphore by certain amount.
  void Acquire(uint32_t amount) {
    mutex_.Lock();
    while (count_ < amount) {
      cond_.Wait(&mutex_);
    }
    count_ -= amount;
    mutex_.Unlock();
  }

  // Releases the semaphore by certain amount.
  void Release(uint32_t amount) {
    mutex_.Lock();
    count_ += amount;
    cond_.SignalAll();
    mutex_.Unlock();
  }

 private:
  uint32_t count_;
  absl::Mutex mutex_;
  absl::CondVar cond_;
};

class InferenceState {
 public:
  InferenceState() : input_tensor_batches_(), batch_timestamps_() {}
  // A mapping between stream tags and the tensors we are collecting as a
  // batch.
  std::map<std::string, std::vector<tf::Tensor>> input_tensor_batches_;
  // The timestamps that go into a batch.
  std::vector<Timestamp> batch_timestamps_;
};

}  // namespace

// This calculator performs inference on a trained TensorFlow model.
//
// TensorFlow Sessions can be created from checkpoint paths, frozen models, or
// the SavedModel system. See the TensorFlowSessionFrom* packet generators for
// details. Each of these methods defines a mapping between MediaPipe streams
// and TensorFlow tensors. All of this information is passed in as an
// input_side_packet.
//
// The input and output streams are TensorFlow tensors labeled by tags. The tags
// for the streams are matched to feeds and fetches in a TensorFlow session
// using a named_signature.generic_signature in the ModelManifest. The
// generic_signature is used as key-value pairs between the MediaPipe tag and
// the TensorFlow tensor. The signature_name in the options proto determines
// which named_signature is used. The keys in the generic_signature must be
// valid MediaPipe tags ([A-Z0-9_]*, no lowercase or special characters). All of
// the tensors corresponding to tags in the signature for input_streams are fed
// to the model and for output_streams the tensors are fetched from the model.
//
// Other calculators are used to convert data to and from tensors, this op only
// handles the TensorFlow session and batching. Batching occurs by concatenating
// input tensors along the 0th dimension across timestamps. If the 0th dimension
// is not a batch dimension, this calculator will add a 0th dimension by
// default. Setting add_batch_dim_to_tensors to false disables the dimension
// addition. Once batch_size inputs have been provided, the batch will be run
// and the output tensors sent out on the output streams with timestamps
// corresponding to the input stream packets. Setting the batch_size to 1
// completely disables batching, but is independent of add_batch_dim_to_tensors.
//
// The TensorFlowInferenceCalculator also support feeding states recurrently for
// RNNs and LSTMs. Simply set the recurrent_tag_pair options to define the
// recurrent tensors. Initializing the recurrent state can be handled by the
// GraphTensorsPacketGenerator.
//
// The calculator updates two Counters to report timing information:
//   --<name>-TotalTimeUsecs = Total time spent running inference (in usecs),
//   --<name>-TotalProcessedTimestamps = # of instances processed
//         (approximately batches processed  * batch_size),
// where <name> is replaced with CalculatorGraphConfig::Node::name() if it
// exists, or with TensorFlowInferenceCalculator if the name is not set. The
// name must be set for timing information to be instance-specific in graphs
// with multiple TensorFlowInferenceCalculators.
//
// Example config:
//   packet_generator {
//     packet_generator: "TensorFlowSessionFromSavedModelGenerator"
//     output_side_packet: "tensorflow_session"
//     options {
//       [mediapipe.TensorFlowSessionFromSavedModelGeneratorOptions.ext]: {
//         saved_model_path: "/path/to/saved/model"
//         signature_name: "mediapipe"
//       }
//     }
//   }
//   node {
//     calculator: "TensorFlowInferenceCalculator"
//     input_stream: "IMAGES:image_tensors_keyed_in_signature_by_tag"
//     input_stream: "AUDIO:audio_tensors_keyed_in_signature_by_tag"
//     output_stream: "LABELS:softmax_tensor_keyed_in_signature_by_tag"
//     input_side_packet: "SESSION:tensorflow_session"
//   }
//
//   Where the input and output streams are treated as Packet<tf::Tensor> and
//   the mediapipe_signature has tensor bindings between "IMAGES", "AUDIO", and
//   "LABELS" and their respective tensors exported to /path/to/bundle. For an
//   example of how this model was exported, see
//   tensorflow_inference_test_graph_generator.py
//
// It is possible to use a GraphDef proto that was not exported by exporter (i.e
// without MetaGraph with bindings). Such GraphDef could contain all of its
// parameters in-lined (for example, it can be the output of freeze_graph.py).
// To instantiate a TensorFlow model from a GraphDef file, replace the
// packet_factory above with TensorFlowSessionFromFrozenGraphGenerator:
//
//   packet_generator {
//     packet_generator: "TensorFlowSessionFromFrozenGraphGenerator"
//     output_side_packet: "SESSION:tensorflow_session"
//     options {
//       [mediapipe.TensorFlowSessionFromFrozenGraphGeneratorOptions.ext]: {
//         graph_proto_path: "[PATH]"
//         tag_to_tensor_names {
//           key: "JPG_STRING"
//           value: "input:0"
//         }
//         tag_to_tensor_names {
//           key: "SOFTMAX"
//           value: "softmax:0"
//         }
//       }
//     }
//   }
//
// It is also possible to use a GraphDef proto and checkpoint file that have not
// been frozen. This can be used to load graphs directly as they have been
// written from training. However, it is more brittle and you are encouraged to
// use a one of the more perminent formats described above. To instantiate a
// TensorFlow model from a GraphDef file and checkpoint, replace the
// packet_factory above with TensorFlowSessionFromModelCheckpointGenerator:
//
//   packet_generator {
//     packet_generator: "TensorFlowSessionFromModelCheckpointGenerator"
//     output_side_packet: "SESSION:tensorflow_session"
//     options {
//       [mediapipe.TensorFlowSessionFromModelCheckpointGeneratorOptions.ext]: {
//         graph_proto_path: "[PATH]"
//         model_options {
//           checkpoint_path: "[PATH2]"
//         }
//         tag_to_tensor_names {
//           key: "JPG_STRING"
//           value: "input:0"
//         }
//         tag_to_tensor_names {
//           key: "SOFTMAX"
//           value: "softmax:0"
//         }
//       }
//     }
//   }
class TensorFlowInferenceCalculator : public CalculatorBase {
 public:
  // Counters for recording timing information. The actual names have the value
  // of CalculatorGraphConfig::Node::name() prepended.
  static constexpr char kTotalUsecsCounterSuffix[] = "TotalTimeUsecs";
  static constexpr char kTotalProcessedTimestampsCounterSuffix[] =
      "TotalProcessedTimestamps";
  static constexpr char kTotalSessionRunsTimeUsecsCounterSuffix[] =
      "TotalSessionRunsTimeUsecs";
  static constexpr char kTotalNumSessionRunsCounterSuffix[] =
      "TotalNumSessionRuns";

  TensorFlowInferenceCalculator() : session_(nullptr) {
    clock_ = std::unique_ptr<mediapipe::Clock>(
        mediapipe::MonotonicClock::CreateSynchronizedMonotonicClock());
  }

  static absl::Status GetContract(CalculatorContract* cc) {
    const auto& options = cc->Options<TensorFlowInferenceCalculatorOptions>();
    RET_CHECK(!cc->Inputs().GetTags().empty());
    for (const std::string& tag : cc->Inputs().GetTags()) {
      // The tensorflow::Tensor with the tag equal to the graph node. May
      // have a TimeSeriesHeader if all present TimeSeriesHeaders match.
      if (!options.batched_input()) {
        cc->Inputs().Tag(tag).Set<tf::Tensor>();
      } else {
        cc->Inputs().Tag(tag).Set<std::vector<mediapipe::Packet>>();
      }
    }
    RET_CHECK(!cc->Outputs().GetTags().empty());
    for (const std::string& tag : cc->Outputs().GetTags()) {
      // The tensorflow::Tensor with tag equal to the graph node to
      // output.  Any TimeSeriesHeader from the inputs will be forwarded
      // with channels set to 0.
      cc->Outputs().Tag(tag).Set<tf::Tensor>();
    }
    // A mediapipe::TensorFlowSession with a model loaded and ready for use.
    // For this calculator it must include a tag_to_tensor_map.
    cc->InputSidePackets().Tag(kSessionTag).Set<TensorFlowSession>();
    if (cc->InputSidePackets().HasTag(kRecurrentInitTensorsTag)) {
      cc->InputSidePackets()
          .Tag(kRecurrentInitTensorsTag)
          .Set<std::unique_ptr<std::map<std::string, tf::Tensor>>>();
    }
    return absl::OkStatus();
  }

  std::unique_ptr<InferenceState> CreateInferenceState(CalculatorContext* cc)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    std::unique_ptr<InferenceState> inference_state =
        absl::make_unique<InferenceState>();
    if (cc->InputSidePackets().HasTag(kRecurrentInitTensorsTag) &&
        !cc->InputSidePackets().Tag(kRecurrentInitTensorsTag).IsEmpty()) {
      std::map<std::string, tf::Tensor>* init_tensor_map;
      init_tensor_map = GetFromUniquePtr<std::map<std::string, tf::Tensor>>(
          cc->InputSidePackets().Tag(kRecurrentInitTensorsTag));
      for (const auto& p : *init_tensor_map) {
        inference_state->input_tensor_batches_[p.first].emplace_back(p.second);
      }
    }
    return inference_state;
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<TensorFlowInferenceCalculatorOptions>();

    RET_CHECK(cc->InputSidePackets().HasTag(kSessionTag));
    session_ = cc->InputSidePackets()
                   .Tag(kSessionTag)
                   .Get<TensorFlowSession>()
                   .session.get();
    tag_to_tensor_map_ = cc->InputSidePackets()
                             .Tag(kSessionTag)
                             .Get<TensorFlowSession>()
                             .tag_to_tensor_map;

    // Validate and store the recurrent tags
    RET_CHECK(options_.has_batch_size());
    RET_CHECK(options_.batch_size() == 1 ||
              options_.recurrent_tag_pair().empty())
        << "To use recurrent_tag_pairs, batch_size must be 1.";

    // Helper for StrJoin. Prints key (tag) of map<string, string>.
    auto TagFormatter =
        absl::PairFormatter(absl::StreamFormatter(), "",
                            [](std::string* out, const std::string& second) {});

    for (const auto& tag_pair : options_.recurrent_tag_pair()) {
      const std::vector<std::string> tags = absl::StrSplit(tag_pair, ':');
      RET_CHECK_EQ(tags.size(), 2) << "recurrent_tag_pair must be a colon "
                                      "separated string with two components: "
                                   << tag_pair;

      RET_CHECK(mediapipe::ContainsKey(tag_to_tensor_map_, tags[0]))
          << "Can't find tag '" << tags[0] << "' in signature "
          << options_.signature_name() << "; instead found tags "
          << absl::StrJoin(tag_to_tensor_map_, ", ", TagFormatter);
      RET_CHECK(mediapipe::ContainsKey(tag_to_tensor_map_, tags[1]))
          << "Can't find tag '" << tags[1] << "' in signature "
          << options_.signature_name() << " ; instead found tags "
          << absl::StrJoin(tag_to_tensor_map_, ", ", TagFormatter);
      recurrent_feed_tags_.insert(tags[0]);
      recurrent_fetch_tags_to_feed_tags_[tags[1]] = tags[0];
    }

    // Check that all tags are present in this signature bound to tensors.
    for (const std::string& tag : cc->Inputs().GetTags()) {
      RET_CHECK(mediapipe::ContainsKey(tag_to_tensor_map_, tag))
          << "Can't find tag '" << tag << "' in signature "
          << options_.signature_name() << "; instead found tags "
          << absl::StrJoin(tag_to_tensor_map_, ", ", TagFormatter);
    }
    for (const std::string& tag : cc->Outputs().GetTags()) {
      RET_CHECK(mediapipe::ContainsKey(tag_to_tensor_map_, tag))
          << "Can't find tag '" << tag << "' in signature "
          << options_.signature_name() << "; instead found tags "
          << absl::StrJoin(tag_to_tensor_map_, ", ", TagFormatter);
    }

    {
      absl::WriterMutexLock l(&mutex_);
      inference_state_ = std::unique_ptr<InferenceState>();
    }

    if (options_.batch_size() == 1 || options_.batched_input()) {
      cc->SetOffset(0);
    }

    return absl::OkStatus();
  }

  // Adds a batch dimension to the input tensor if specified in the calculator
  // options.
  absl::Status AddBatchDimension(tf::Tensor* input_tensor) {
    if (options_.add_batch_dim_to_tensors()) {
      tf::TensorShape new_shape(input_tensor->shape());
      new_shape.InsertDim(0, 1);
      RET_CHECK(input_tensor->CopyFrom(*input_tensor, new_shape))
          << "Could not add 0th dimension to tensor without changing its shape."
          << " Current shape: " << input_tensor->shape().DebugString();
    }
    return absl::OkStatus();
  }

  absl::Status AggregateTensorPacket(
      const std::string& tag_name, const Packet& packet,
      std::map<Timestamp, std::map<std::string, tf::Tensor>>*
          input_tensors_by_tag_by_timestamp,
      InferenceState* inference_state) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    tf::Tensor input_tensor(packet.Get<tf::Tensor>());
    RET_CHECK_OK(AddBatchDimension(&input_tensor));
    if (mediapipe::ContainsKey(recurrent_feed_tags_, tag_name)) {
      // If we receive an input on a recurrent tag, override the state.
      // It's OK to override the global state because there is just one
      // input stream allowed for recurrent tensors.
      inference_state_->input_tensor_batches_[tag_name].clear();
    }
    (*input_tensors_by_tag_by_timestamp)[packet.Timestamp()].insert(
        std::make_pair(tag_name, input_tensor));
    return absl::OkStatus();
  }

  // Removes the batch dimension of the output tensor if specified in the
  // calculator options.
  absl::Status RemoveBatchDimension(tf::Tensor* output_tensor) {
    if (options_.add_batch_dim_to_tensors()) {
      tf::TensorShape new_shape(output_tensor->shape());
      new_shape.RemoveDim(0);
      RET_CHECK(output_tensor->CopyFrom(*output_tensor, new_shape))
          << "Could not remove 0th dimension from tensor without changing its "
          << "shape. Current shape: " << output_tensor->shape().DebugString()
          << " (The expected first dimension is 1 for a batch element.)";
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::unique_ptr<InferenceState> inference_state_to_process;
    {
      absl::WriterMutexLock l(&mutex_);
      if (inference_state_ == nullptr) {
        inference_state_ = CreateInferenceState(cc);
      }
      std::map<Timestamp, std::map<std::string, tf::Tensor>>
          input_tensors_by_tag_by_timestamp;
      for (const std::string& tag_as_node_name : cc->Inputs().GetTags()) {
        if (cc->Inputs().Tag(tag_as_node_name).IsEmpty()) {
          // Recurrent tensors can be empty.
          if (!mediapipe::ContainsKey(recurrent_feed_tags_, tag_as_node_name)) {
            if (options_.skip_on_missing_features()) {
              return absl::OkStatus();
            } else {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Tag ", tag_as_node_name,
                  " not present at timestamp: ", cc->InputTimestamp().Value()));
            }
          }
        } else if (options_.batched_input()) {
          const auto& tensor_packets =
              cc->Inputs().Tag(tag_as_node_name).Get<std::vector<Packet>>();
          if (tensor_packets.size() > options_.batch_size()) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Batch for tag ", tag_as_node_name,
                " has more packets than batch capacity. batch_size: ",
                options_.batch_size(), " packets: ", tensor_packets.size()));
          }
          for (const auto& packet : tensor_packets) {
            RET_CHECK_OK(AggregateTensorPacket(
                tag_as_node_name, packet, &input_tensors_by_tag_by_timestamp,
                inference_state_.get()));
          }
        } else {
          RET_CHECK_OK(AggregateTensorPacket(
              tag_as_node_name, cc->Inputs().Tag(tag_as_node_name).Value(),
              &input_tensors_by_tag_by_timestamp, inference_state_.get()));
        }
      }
      for (const auto& timestamp_and_input_tensors_by_tag :
           input_tensors_by_tag_by_timestamp) {
        inference_state_->batch_timestamps_.emplace_back(
            timestamp_and_input_tensors_by_tag.first);
        for (const auto& input_tensor_and_tag :
             timestamp_and_input_tensors_by_tag.second) {
          inference_state_->input_tensor_batches_[input_tensor_and_tag.first]
              .emplace_back(input_tensor_and_tag.second);
        }
      }
      if (inference_state_->batch_timestamps_.size() == options_.batch_size() ||
          options_.batched_input()) {
        inference_state_to_process = std::move(inference_state_);
        inference_state_ = std::unique_ptr<InferenceState>();
      }
    }

    if (inference_state_to_process) {
      MP_RETURN_IF_ERROR(
          OutputBatch(cc, std::move(inference_state_to_process)));
    }

    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    std::unique_ptr<InferenceState> inference_state_to_process = nullptr;
    {
      absl::WriterMutexLock l(&mutex_);
      if (cc->GraphStatus().ok() && inference_state_ != nullptr &&
          !inference_state_->batch_timestamps_.empty()) {
        inference_state_to_process = std::move(inference_state_);
        inference_state_ = std::unique_ptr<InferenceState>();
      }
    }
    if (inference_state_to_process) {
      MP_RETURN_IF_ERROR(
          OutputBatch(cc, std::move(inference_state_to_process)));
    }
    return absl::OkStatus();
  }

  // When a batch of input tensors is ready to be run, runs TensorFlow and
  // outputs the output tensors. The output tensors have timestamps matching
  // the input tensor that formed that batch element. Any requested
  // batch_dimension is added and removed. This code takes advantage of the fact
  // that copying a tensor shares the same reference-counted, heap allocated
  // memory buffer. Therefore, copies are cheap and should not cause the memory
  // buffer to fall out of scope. In contrast, concat is only used where
  // necessary.
  absl::Status OutputBatch(CalculatorContext* cc,
                           std::unique_ptr<InferenceState> inference_state) {
    const int64_t start_time = absl::ToUnixMicros(clock_->TimeNow());
    std::vector<std::pair<mediapipe::ProtoString, tf::Tensor>> input_tensors;

    for (auto& keyed_tensors : inference_state->input_tensor_batches_) {
      if (options_.batch_size() == 1) {
        // Short circuit to avoid the cost of deep copying tensors in concat.
        if (!keyed_tensors.second.empty()) {
          input_tensors.emplace_back(tag_to_tensor_map_[keyed_tensors.first],
                                     keyed_tensors.second[0]);
        } else {
          // The input buffer can be empty for recurrent tensors.
          RET_CHECK(
              mediapipe::ContainsKey(recurrent_feed_tags_, keyed_tensors.first))
              << "A non-recurrent tensor does not have an input: "
              << keyed_tensors.first;
        }
      } else {
        if (options_.pad_to_batch_size()) {
          // Pad by replicating the first tensor, then ignore the values.
          keyed_tensors.second.resize(options_.batch_size());
          std::fill(keyed_tensors.second.begin() +
                        inference_state->batch_timestamps_.size(),
                    keyed_tensors.second.end(), keyed_tensors.second[0]);
        }
        tf::Tensor concated;
        const absl::Status concat_status =
            tf::tensor::Concat(keyed_tensors.second, &concated);
        ABSL_CHECK(concat_status.ok()) << concat_status.ToString();
        input_tensors.emplace_back(tag_to_tensor_map_[keyed_tensors.first],
                                   concated);
      }
    }
    inference_state->input_tensor_batches_.clear();
    std::vector<mediapipe::ProtoString> output_tensor_names;
    std::vector<std::string> output_name_in_signature;
    for (const std::string& tag : cc->Outputs().GetTags()) {
      output_tensor_names.emplace_back(tag_to_tensor_map_[tag]);
      output_name_in_signature.emplace_back(tag);
    }
    for (const auto& tag_pair : recurrent_fetch_tags_to_feed_tags_) {
      // Ensure that we always fetch the recurrent state tensors.
      if (std::find(output_name_in_signature.begin(),
                    output_name_in_signature.end(),
                    tag_pair.first) == output_name_in_signature.end()) {
        output_tensor_names.emplace_back(tag_to_tensor_map_[tag_pair.first]);
        output_name_in_signature.emplace_back(tag_pair.first);
      }
    }
    std::vector<tf::Tensor> outputs;

    SimpleSemaphore* session_run_throttle = nullptr;
    if (options_.max_concurrent_session_runs() > 0) {
      session_run_throttle =
          get_session_run_throttle(options_.max_concurrent_session_runs());
      session_run_throttle->Acquire(1);
    }
    const int64_t run_start_time = absl::ToUnixMicros(clock_->TimeNow());
    absl::Status tf_status;
    {
#if !defined(MEDIAPIPE_MOBILE) && !defined(__APPLE__)
      tsl::profiler::TraceMe trace(absl::string_view(cc->NodeName()));
#endif
      tf_status = session_->Run(input_tensors, output_tensor_names,
                                {} /* target_node_names */, &outputs);
    }

    if (session_run_throttle != nullptr) {
      session_run_throttle->Release(1);
    }

    // RET_CHECK on the tf::Status object itself in order to print an
    // informative error message.
    RET_CHECK(tf_status.ok()) << "Run failed: " << tf_status.ToString();

    const int64_t run_end_time = absl::ToUnixMicros(clock_->TimeNow());
    cc->GetCounter(kTotalSessionRunsTimeUsecsCounterSuffix)
        ->IncrementBy(run_end_time - run_start_time);
    cc->GetCounter(kTotalNumSessionRunsCounterSuffix)->Increment();

    // Feed back the recurrent state.
    for (const auto& tag_pair : recurrent_fetch_tags_to_feed_tags_) {
      int pos = std::find(output_name_in_signature.begin(),
                          output_name_in_signature.end(), tag_pair.first) -
                output_name_in_signature.begin();
      inference_state->input_tensor_batches_[tag_pair.second].emplace_back(
          outputs[pos]);
    }

    absl::WriterMutexLock l(&mutex_);
    // Set that we want to split on each index of the 0th dimension.
    std::vector<int64_t> split_vector(
        options_.pad_to_batch_size()
            ? options_.batch_size()
            : inference_state->batch_timestamps_.size(),
        1);
    for (int i = 0; i < output_tensor_names.size(); ++i) {
      if (options_.batch_size() == 1) {
        if (cc->Outputs().HasTag(output_name_in_signature[i])) {
          tf::Tensor output_tensor(outputs[i]);
          RET_CHECK_OK(RemoveBatchDimension(&output_tensor));
          cc->Outputs()
              .Tag(output_name_in_signature[i])
              .Add(new tf::Tensor(output_tensor),
                   inference_state->batch_timestamps_[0]);
        }
      } else {
        std::vector<tf::Tensor> split_tensors;
        const absl::Status split_status =
            tf::tensor::Split(outputs[i], split_vector, &split_tensors);
        ABSL_CHECK(split_status.ok()) << split_status.ToString();
        // Loop over timestamps so that we don't copy the padding.
        for (int j = 0; j < inference_state->batch_timestamps_.size(); ++j) {
          tf::Tensor output_tensor(split_tensors[j]);
          RET_CHECK_OK(RemoveBatchDimension(&output_tensor));
          cc->Outputs()
              .Tag(output_name_in_signature[i])
              .Add(new tf::Tensor(output_tensor),
                   inference_state->batch_timestamps_[j]);
        }
      }
    }

    // Get end time and report.
    const int64_t end_time = absl::ToUnixMicros(clock_->TimeNow());
    cc->GetCounter(kTotalUsecsCounterSuffix)
        ->IncrementBy(end_time - start_time);
    cc->GetCounter(kTotalProcessedTimestampsCounterSuffix)
        ->IncrementBy(inference_state->batch_timestamps_.size());

    // Make sure we hold on to the recursive state.
    if (!options_.recurrent_tag_pair().empty()) {
      inference_state_ = std::move(inference_state);
      inference_state_->batch_timestamps_.clear();
    }

    return absl::OkStatus();
  }

 private:
  // The Session object is provided by a packet factory and is owned by the
  // MediaPipe framework. Individual calls are thread-safe, but session state
  // may be shared across threads.
  tf::Session* session_;

  // A mapping between stream tags and the tensor names they are bound to.
  std::map<std::string, std::string> tag_to_tensor_map_;

  absl::Mutex mutex_;
  std::unique_ptr<InferenceState> inference_state_ ABSL_GUARDED_BY(mutex_);

  // The options for the calculator.
  TensorFlowInferenceCalculatorOptions options_;

  // Store the feed and fetch tags for feed/fetch recurrent networks.
  std::set<std::string> recurrent_feed_tags_;
  std::map<std::string, std::string> recurrent_fetch_tags_to_feed_tags_;

  // Clock used to measure the computation time in OutputBatch().
  std::unique_ptr<mediapipe::Clock> clock_;

  // The static singleton semaphore to throttle concurrent session runs.
  static SimpleSemaphore* get_session_run_throttle(
      int32_t max_concurrent_session_runs) {
    static SimpleSemaphore* session_run_throttle =
        new SimpleSemaphore(max_concurrent_session_runs);
    return session_run_throttle;
  }
};
REGISTER_CALCULATOR(TensorFlowInferenceCalculator);

constexpr char TensorFlowInferenceCalculator::kTotalUsecsCounterSuffix[];
constexpr char
    TensorFlowInferenceCalculator::kTotalProcessedTimestampsCounterSuffix[];
constexpr char
    TensorFlowInferenceCalculator::kTotalSessionRunsTimeUsecsCounterSuffix[];
constexpr char
    TensorFlowInferenceCalculator::kTotalNumSessionRunsCounterSuffix[];
}  // namespace mediapipe
