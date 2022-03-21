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
//
// Simple calculators that are useful for test cases.

#include <memory>
#include <random>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/mathutil.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/test_calculators.pb.h"

namespace mediapipe {

constexpr char kOutTag[] = "OUT";
constexpr char kClockTag[] = "CLOCK";
constexpr char kSleepMicrosTag[] = "SLEEP_MICROS";
constexpr char kCloseTag[] = "CLOSE";
constexpr char kProcessTag[] = "PROCESS";
constexpr char kOpenTag[] = "OPEN";
constexpr char kTag[] = "";
constexpr char kMeanTag[] = "MEAN";
constexpr char kDataTag[] = "DATA";
constexpr char kPairTag[] = "PAIR";
constexpr char kLowTag[] = "LOW";
constexpr char kHighTag[] = "HIGH";

using RandomEngine = std::mt19937_64;

// A Calculator that outputs twice the value of its input packet (an int).
class DoubleIntCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    int value = cc->Inputs().Index(0).Value().Get<int>();
    cc->Outputs().Index(0).Add(new int(2 * value), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(DoubleIntCalculator);

// Splits a uint64 into a pair of two uint32, the first element of which
// holds the high order bits and the second the low order ones.
class IntSplitterPacketGenerator : public PacketGenerator {
 public:
  static absl::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,  //
      PacketTypeSet* input_side_packets,                 //
      PacketTypeSet* output_side_packets) {
    input_side_packets->Index(0).Set<uint64>();
    output_side_packets->Index(0).Set<std::pair<uint32, uint32>>();
    return absl::OkStatus();
  }

  static absl::Status Generate(
      const PacketGeneratorOptions& extendable_options,  //
      const PacketSet& input_side_packets,               //
      PacketSet* output_side_packets) {
    uint64 value = input_side_packets.Index(0).Get<uint64>();
    uint32 high = value >> 32;
    uint32 low = value & 0xFFFFFFFF;
    output_side_packets->Index(0) =
        Adopt(new std::pair<uint32, uint32>(high, low));
    return absl::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(IntSplitterPacketGenerator);

// Takes a uint64 and produces three input side packets, a uint32 of the
// high order bits, a uint32 of the low order bits and a pair of uint32
// with both the high and low order bits.
class TaggedIntSplitterPacketGenerator : public PacketGenerator {
 public:
  static absl::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,  //
      PacketTypeSet* input_side_packets,                 //
      PacketTypeSet* output_side_packets) {
    input_side_packets->Index(0).Set<uint64>();
    output_side_packets->Tag(kHighTag).Set<uint32>();
    output_side_packets->Tag(kLowTag).Set<uint32>();
    output_side_packets->Tag(kPairTag).Set<std::pair<uint32, uint32>>();
    return absl::OkStatus();
  }

  static absl::Status Generate(
      const PacketGeneratorOptions& extendable_options,  //
      const PacketSet& input_side_packets,               //
      PacketSet* output_side_packets) {
    uint64 value = input_side_packets.Index(0).Get<uint64>();
    uint32 high = value >> 32;
    uint32 low = value & 0xFFFFFFFF;
    output_side_packets->Tag(kHighTag) = Adopt(new uint32(high));
    output_side_packets->Tag(kLowTag) = Adopt(new uint32(low));
    output_side_packets->Tag(kPairTag) =
        Adopt(new std::pair<uint32, uint32>(high, low));
    return absl::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(TaggedIntSplitterPacketGenerator);

// A Calculator that gets a pointer to input side packet pair<int,
// int>(N, K), and outputs Packets each containing an int value of K,
// at timestamps 0, N, and all the timestamps between 0 and N that are
// divisible by K. Sets the output stream header to "RangeCalculatorK". In
// the second output stream output an int Packet at Timestamp::PostStream
// with the total sum of all values sent over the first stream.  In the
// third output a double Packet with the arithmetic mean of the values
// on the first stream (output at Timestamp::PreStream()).
class RangeCalculator : public CalculatorBase {
 public:
  RangeCalculator() : initialized_(false) {}

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    cc->Outputs().Index(1).Set<int>();
    cc->Outputs().Index(2).Set<double>();
    cc->InputSidePackets().Index(0).Set<std::pair<uint32, uint32>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    Initialize(cc);

    // Fail if requested, without setting any stream headers. This tests that
    // the downstream Calculators will not try to access the headers in case
    // this one failed.
    if (k_ == 0) {
      return absl::Status(absl::StatusCode::kCancelled, "k_ == 0");
    }
    cc->Outputs().Index(0).SetHeader(
        Adopt(new std::string(absl::StrCat(cc->CalculatorType(), k_))));
    // Output at timestamp 0.
    cc->Outputs().Index(0).AddPacket(GetNextPacket().At(Timestamp(0)));

    cc->Outputs().Index(1).SetNextTimestampBound(Timestamp::PostStream());
    cc->Outputs().Index(2).SetNextTimestampBound(Timestamp::PreStream());

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    // Output at timestamps 1:N-1 that are divisible by K.
    index_ += k_;
    if (index_ < n_) {
      cc->Outputs().Index(0).AddPacket(GetNextPacket().At(Timestamp(index_)));
      return absl::OkStatus();
    } else {
      return tool::StatusStop();
    }
  }

  absl::Status Close(CalculatorContext* cc) final {
    // Output at timestamp N.
    cc->Outputs().Index(0).AddPacket(GetNextPacket().At(Timestamp(n_)));
    // Output: ints from a range specified in the input side packet.
    cc->Outputs().Index(1).Add(new int(total_), Timestamp::PostStream());
    cc->Outputs().Index(2).Add(
        new double(static_cast<double>(total_) / static_cast<double>(count_)),
        Timestamp::PreStream());

    return absl::OkStatus();
  }

 private:
  Packet GetNextPacket() {
    int value = k_ * 100 + count_;
    total_ += value;
    ++count_;
    return Adopt(new int(value));
  }

  // Initializes this object.
  void Initialize(CalculatorContext* cc) {
    CHECK(!initialized_);

    cc->Options();  // Ensure Options() can be called here.
    std::tie(n_, k_) =
        cc->InputSidePackets().Index(0).Get<std::pair<uint32, uint32>>();

    index_ = 0;
    total_ = 0;
    count_ = 0;

    initialized_ = true;
  }

  // Parameters provided to the calculator.
  int n_;
  int k_;
  // Current timestamp.
  int index_;
  int total_;
  int count_;
  bool initialized_;
};
REGISTER_CALCULATOR(RangeCalculator);

// Compute the standard deviation of values on the stream "DATA" given
// the mean on stream "MEAN".
class StdDevCalculator : public CalculatorBase {
 public:
  StdDevCalculator() {}

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kDataTag).Set<int>();
    cc->Inputs().Tag(kMeanTag).Set<double>();
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).SetNextTimestampBound(Timestamp::PostStream());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (cc->InputTimestamp() == Timestamp::PreStream()) {
      RET_CHECK(cc->Inputs().Tag(kDataTag).Value().IsEmpty());
      RET_CHECK(!cc->Inputs().Tag(kMeanTag).Value().IsEmpty());
      mean_ = cc->Inputs().Tag(kMeanTag).Get<double>();
      initialized_ = true;
    } else {
      RET_CHECK(initialized_);
      RET_CHECK(!cc->Inputs().Tag(kDataTag).Value().IsEmpty());
      RET_CHECK(cc->Inputs().Tag(kMeanTag).Value().IsEmpty());
      double diff = cc->Inputs().Tag(kDataTag).Get<int>() - mean_;
      cummulative_variance_ += diff * diff;
      ++count_;
    }
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) final {
    cc->Outputs().Index(0).Add(
        new int(mediapipe::MathUtil::SafeRound<int, double>(
            sqrt(cummulative_variance_ / count_) * 100.0)),
        Timestamp::PostStream());
    return absl::OkStatus();
  }

 private:
  double cummulative_variance_ = 0.0;
  int count_ = 0;
  double mean_ = 0.0;
  bool initialized_ = false;
};
REGISTER_CALCULATOR(StdDevCalculator);

// A calculator that receives some number of input streams carrying ints.
// Outputs, for each input timestamp, a space separated string containing
// the timestamp and all the inputs for that timestamp (Empty inputs
// will be denoted with "empty"). Sets the header to be a space-separated
// concatenation of the input stream headers.
class MergeCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<int>();
    }
    cc->Outputs().Index(0).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    auto header = absl::make_unique<std::string>();
    for (auto& input : cc->Inputs()) {
      if (!input.Header().IsEmpty()) {
        if (!header->empty()) {
          absl::StrAppend(header.get(), " ");
        }
        absl::StrAppend(header.get(), input.Header().Get<std::string>());
      }
    }
    cc->Outputs().Index(0).SetHeader(Adopt(header.release()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    std::string result;
    if (cc->InputTimestamp().IsSpecialValue()) {
      absl::StrAppend(&result, cc->InputTimestamp().DebugString());
    } else {
      absl::StrAppend(&result, "Timestamp(", cc->InputTimestamp().DebugString(),
                      ")");
    }
    for (const auto& input : cc->Inputs()) {
      const auto& packet = input.Value();
      absl::StrAppend(&result, " ");
      if (!packet.IsEmpty()) {
        absl::StrAppend(&result, packet.Get<int>());
      } else {
        absl::StrAppend(&result, "empty");
      }
    }
    cc->Outputs().Index(0).Add(new std::string(result), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(MergeCalculator);

// A calculator receiving strings from the input stream, and setting
// the output PostStream packet to be the '/'-separated concatenation
// of all the input values.
class SaverCalculator : public CalculatorBase {
 public:
  SaverCalculator() : result_(new std::string) {}

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<std::string>();
    cc->Outputs().Index(0).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).SetNextTimestampBound(Timestamp::PostStream());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (!result_->empty()) {
      result_->append("/");
    }
    result_->append(cc->Inputs().Index(0).Get<std::string>());
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) final {
    cc->Outputs().Index(0).Add(result_.release(), Timestamp::PostStream());
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<std::string> result_;
};
REGISTER_CALCULATOR(SaverCalculator);

#ifndef MEDIAPIPE_MOBILE
// Source Calculator that produces matrices on the output stream with
// each coefficient from a normal gaussian.  A string seed must be given
// as an input side packet.
class RandomMatrixCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<Matrix>();
    cc->InputSidePackets().Index(0).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    auto& options = cc->Options<RandomMatrixCalculatorOptions>();
    CHECK_LT(0, options.timestamp_step());
    CHECK_LT(0, options.rows());
    CHECK_LT(0, options.cols());
    CHECK_LT(options.start_timestamp(), options.limit_timestamp());

    current_timestamp_ = Timestamp(options.start_timestamp());
    cc->Outputs().Index(0).SetNextTimestampBound(current_timestamp_);
    const std::string& seed_str =
        cc->InputSidePackets().Index(0).Get<std::string>();
    std::seed_seq seq(seed_str.begin(), seed_str.end());
    std::vector<std::uint32_t> seed(1);
    seq.generate(seed.begin(), seed.end());
    random_ = absl::make_unique<RandomEngine>(seed[0]);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto& options = cc->Options<RandomMatrixCalculatorOptions>();

    Matrix* matrix = new Matrix(options.rows(), options.cols());
    for (int i = 0; i < options.rows() * options.cols(); ++i) {
      std::normal_distribution<float> normal_distribution(0.0, 1.0);
      (*matrix)(i) = normal_distribution(*random_);
    }
    cc->Outputs().Index(0).Add(matrix, current_timestamp_);

    current_timestamp_ += TimestampDiff(options.timestamp_step());
    cc->Outputs().Index(0).SetNextTimestampBound(current_timestamp_);
    if (current_timestamp_ >= Timestamp(options.limit_timestamp())) {
      return tool::StatusStop();
    } else {
      return absl::OkStatus();
    }
  }

 private:
  Timestamp current_timestamp_;
  std::unique_ptr<RandomEngine> random_;
};
REGISTER_CALCULATOR(RandomMatrixCalculator);

#endif  // !defined(MEDIAPIPE_MOBILE)

// Calculate the mean and covariance of the input samples.  Each sample
// must be a column matrix.  The computation is done in an online fashion,
// so the number of samples can be arbitrarily large without fear of
// using too much memory (however, no algorithm is used to mitigate the
// effect of round off error).
class MeanAndCovarianceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Matrix>();
    cc->Outputs().Index(0).Set<std::pair<Matrix, Matrix>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->Outputs().Index(0).SetNextTimestampBound(Timestamp::PostStream());

    rows_ = -1;
    num_samples_ = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const Eigen::MatrixXd sample =
        cc->Inputs().Index(0).Get<Matrix>().cast<double>();
    CHECK_EQ(1, sample.cols());
    if (num_samples_ == 0) {
      rows_ = sample.rows();
      sum_vector_ = Eigen::VectorXd::Zero(rows_);
      outer_product_sum_ = Eigen::MatrixXd::Zero(rows_, rows_);
    } else {
      CHECK_EQ(sample.rows(), rows_);
    }
    sum_vector_ += sample;
    outer_product_sum_ += sample * sample.transpose();

    ++num_samples_;
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    Eigen::VectorXd mean_vector = sum_vector_ / num_samples_;
    Eigen::MatrixXd covariance_matrix(rows_, rows_);

    for (int i = 0; i < rows_; ++i) {
      for (int k = 0; k < rows_; ++k) {
        // Generate the covariance matrix, taking into account that the mean
        // must be subtracted from each sample.
        covariance_matrix(i, k) =
            (outer_product_sum_(i, k) - sum_vector_(k) * mean_vector(i) -
             sum_vector_(i) * mean_vector(k) +
             mean_vector(i) * mean_vector(k)) /
            num_samples_;
      }
    }

    cc->Outputs().Index(0).Add(
        new std::pair<Eigen::MatrixXf, Eigen::MatrixXf>(
            mean_vector.cast<float>(), covariance_matrix.cast<float>()),
        Timestamp::PostStream());
    return absl::OkStatus();
  }

 private:
  Eigen::VectorXd sum_vector_;
  Eigen::MatrixXd outer_product_sum_;
  int64 num_samples_;
  int rows_;
};
REGISTER_CALCULATOR(MeanAndCovarianceCalculator);

// Takes any number of input side packets and outputs them in order on the
// single output stream.  The timestamp of the packets starts with 0 and
// increases by 1 for each packet.
class SidePacketToOutputPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).SetAny();
    for (int i = 1; i < cc->InputSidePackets().NumEntries(); ++i) {
      cc->InputSidePackets().Index(i).SetSameAs(
          &cc->InputSidePackets().Index(0));
    }
    cc->Outputs().Index(0).SetSameAs(&cc->InputSidePackets().Index(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    int current_timestamp = 0;
    for (const Packet& packet : cc->InputSidePackets()) {
      cc->Outputs().Index(0).AddPacket(packet.At(Timestamp(current_timestamp)));
      ++current_timestamp;
    }
    cc->Outputs().Index(0).Close();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return tool::StatusStop();
  }
};
REGISTER_CALCULATOR(SidePacketToOutputPacketCalculator);

// TODO Remove copy of SidePacketToOutputPacketCalculator with
// old name once all clients are updated.
class ABSL_DEPRECATED("Use SidePacketToOutputPacketCalculator instead")
    ExternalInputToOutputPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).SetAny();
    for (int i = 1; i < cc->InputSidePackets().NumEntries(); ++i) {
      cc->InputSidePackets().Index(i).SetSameAs(
          &cc->InputSidePackets().Index(0));
    }
    cc->Outputs().Index(0).SetSameAs(&cc->InputSidePackets().Index(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    int current_timestamp = 0;
    for (const Packet& packet : cc->InputSidePackets()) {
      cc->Outputs().Index(0).AddPacket(packet.At(Timestamp(current_timestamp)));
      ++current_timestamp;
    }
    cc->Outputs().Index(0).Close();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return tool::StatusStop();
  }
};
REGISTER_CALCULATOR(ExternalInputToOutputPacketCalculator);

// A Calculator::Process callback function.
typedef std::function<absl::Status(const InputStreamShardSet&,
                                   OutputStreamShardSet*)>
    ProcessFunction;

// A callback function for Calculator::Open, Process, or Close.
typedef std::function<absl::Status(CalculatorContext* cc)>
    CalculatorContextFunction;

// A Calculator that runs a testing callback function in Process,
// Open, or Close, which is specified as an input side packet.
class LambdaCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).SetAny();
    }
    for (CollectionItemId id = cc->Outputs().BeginId();
         id < cc->Outputs().EndId(); ++id) {
      cc->Outputs().Get(id).SetAny();
    }
    if (cc->InputSidePackets().HasTag(kTag) > 0) {
      cc->InputSidePackets().Tag(kTag).Set<ProcessFunction>();
    }
    for (const std::string& tag : {"OPEN", "PROCESS", "CLOSE"}) {
      if (cc->InputSidePackets().HasTag(tag)) {
        cc->InputSidePackets().Tag(tag).Set<CalculatorContextFunction>();
      }
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    if (cc->InputSidePackets().HasTag(kOpenTag)) {
      return GetContextFn(cc, "OPEN")(cc);
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (cc->InputSidePackets().HasTag(kProcessTag)) {
      return GetContextFn(cc, "PROCESS")(cc);
    }
    if (cc->InputSidePackets().HasTag(kTag) > 0) {
      return GetProcessFn(cc, "")(cc->Inputs(), &cc->Outputs());
    }
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) final {
    if (cc->InputSidePackets().HasTag(kCloseTag)) {
      return GetContextFn(cc, "CLOSE")(cc);
    }
    return absl::OkStatus();
  }

 private:
  ProcessFunction GetProcessFn(CalculatorContext* cc, std::string tag) {
    return cc->InputSidePackets().Tag(tag).Get<ProcessFunction>();
  }
  CalculatorContextFunction GetContextFn(CalculatorContext* cc,
                                         std::string tag) {
    return cc->InputSidePackets().Tag(tag).Get<CalculatorContextFunction>();
  }
};
REGISTER_CALCULATOR(LambdaCalculator);

// A Calculator that doesn't check anything about input & output and doesn't do
// anything.
// It provides flexility to define the input, output, side packets as
// you wish with any type, with/out tag.
// This is useful if you need to test something about the graph definition and
// stream connections.
class DummyTestCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).SetAny();
    }
    for (CollectionItemId id = cc->Outputs().BeginId();
         id < cc->Outputs().EndId(); ++id) {
      cc->Outputs().Get(id).SetAny();
    }
    for (CollectionItemId id = cc->InputSidePackets().BeginId();
         id < cc->InputSidePackets().EndId(); ++id) {
      cc->InputSidePackets().Get(id).SetAny();
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(DummyTestCalculator);

// A Calculator that passes the input value to the output after sleeping for
// a set number of microseconds.
class PassThroughWithSleepCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->InputSidePackets().Tag(kSleepMicrosTag).Set<int>();
    cc->InputSidePackets().Tag(kClockTag).Set<std::shared_ptr<Clock>>();
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    sleep_micros_ = cc->InputSidePackets().Tag(kSleepMicrosTag).Get<int>();
    if (sleep_micros_ < 0) {
      return absl::InternalError("SLEEP_MICROS should be >= 0");
    }
    clock_ =
        cc->InputSidePackets().Tag(kClockTag).Get<std::shared_ptr<Clock>>();
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) final {
    clock_->Sleep(absl::Microseconds(sleep_micros_));
    int value = cc->Inputs().Index(0).Value().Get<int>();
    cc->Outputs().Index(0).Add(new int(value), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  int sleep_micros_ = 0;
  std::shared_ptr<Clock> clock_;
};
REGISTER_CALCULATOR(PassThroughWithSleepCalculator);

// A Calculator that multiples two input values.
class MultiplyIntCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Inputs().Index(1).SetSameAs(&cc->Inputs().Index(0));
    // cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    RET_CHECK(cc->Outputs().HasTag(kOutTag));
    cc->Outputs().Tag(kOutTag).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) final {
    int x = cc->Inputs().Index(0).Value().Get<int>();
    int y = cc->Inputs().Index(1).Value().Get<int>();
    cc->Outputs().Tag(kOutTag).Add(new int(x * y), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(MultiplyIntCalculator);

// A calculator that forwards nested input packets to the output stream if they
// are not empty, otherwise it transforms them into timestamp bound updates.
class ForwardNestedPacketOrEmitBoundUpdateCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<mediapipe::Packet>();
    cc->Outputs().Index(0).SetAny();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& nested_packet = cc->Inputs().Index(0).Get<mediapipe::Packet>();
    if (!nested_packet.IsEmpty()) {
      cc->Outputs().Index(0).AddPacket(nested_packet);
    } else {
      // Add 1 so that Process() of the downstream calculator is called with
      // exactly this timestamp.
      cc->Outputs().Index(0).SetNextTimestampBound(nested_packet.Timestamp() +
                                                   1);
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(ForwardNestedPacketOrEmitBoundUpdateCalculator);

// A calculator that outputs timestamp bound updates emitted by the upstream
// calculator.
class TimestampBoundReceiverCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).Set<Timestamp>();
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Index(0).IsEmpty()) {
      // Add 1 to get the exact value that was passed to SetNextTimestampBound()
      // in the upstream calculator.
      const Timestamp bound = cc->InputTimestamp() + 1;
      cc->Outputs().Index(0).AddPacket(
          mediapipe::MakePacket<Timestamp>(bound).At(bound));
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TimestampBoundReceiverCalculator);

}  // namespace mediapipe
