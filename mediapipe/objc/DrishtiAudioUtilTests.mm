#import "mediapipe/objc/MediaPipeAudioUtil.h"

#include <cassert>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#import <XCTest/XCTest.h>

static const float kMatrixComparisonPrecisionFloat = 1e-9;
static const float kMatrixComparisonPrecisionInt16 = 1e-4;

@interface MediaPipeAudioUtilTest : XCTestCase
@end

template <typename DataType>
class AudioBufferListWrapper {
 public:
  AudioBufferListWrapper(int num_frames, int num_channels, bool interleaved)
      : num_frames_(num_frames), num_channels_(num_channels), interleaved_(interleaved) {
    int num_buffers = interleaved_ ? 1 : num_channels_;
    int channels_per_buffer = interleaved_ ? num_channels_ : 1;
    int buffer_size_samples = num_frames_ * channels_per_buffer;
    int buffer_size_bytes = buffer_size_samples * static_cast<int>(BytesPerSample());

    buffer_list_.reset(reinterpret_cast<AudioBufferList*>(
        calloc(1, offsetof(AudioBufferList, mBuffers) +
                      (sizeof(AudioBuffer) * num_buffers))));  // Var. length array.
    assert(buffer_list_.get() != nullptr);

    buffer_list_->mNumberBuffers = static_cast<CMItemCount>(num_buffers);
    for (int buffer_index = 0; buffer_index < num_buffers; ++buffer_index) {
      AudioBuffer& buffer = GetBuffer(buffer_index);
      auto buffer_data = std::make_unique<DataType[]>(buffer_size_samples);
      assert(buffer_data != nullptr);

      buffer.mData = static_cast<void*>(buffer_data.get());
      buffer.mDataByteSize = buffer_size_bytes;
      buffer.mNumberChannels = channels_per_buffer;

      buffers_.push_back(std::move(buffer_data));
    }
  }

  UInt32 BytesPerSample() const { return static_cast<UInt32>(sizeof(DataType)); }
  UInt32 BytesPerPacket() const {
    return static_cast<UInt32>(BytesPerSample() * num_frames_ * num_channels_);
  }

  AudioBufferList* GetBufferList() { return buffer_list_.get(); };
  const AudioBufferList* GetBufferList() const { return buffer_list_.get(); };

  AudioBuffer& GetBuffer(int index) { return GetBufferList()->mBuffers[index]; }

  DataType* GetBufferData(int index) { return reinterpret_cast<DataType*>(GetBuffer(index).mData); }

  DataType& At(int channel, int frame) {
    assert(frame >= 0 && frame < num_frames_);
    assert(channel >= 0 && channel < num_channels_);
    if (interleaved_) {
      // [[L, R, L, R, ...]]
      return GetBufferData(0)[frame * num_channels_ + channel];
    } else {
      // [[L, L, ...], [R, R, ...]]
      return GetBufferData(channel)[frame];
    }
  }

  DataType ToDataType(float value) const;

  void InitFromMatrix(const mediapipe::Matrix& matrix) {
    assert(matrix.rows() == num_channels_);
    assert(matrix.cols() == num_frames_);
    for (int channel = 0; channel < num_channels_; ++channel) {
      for (int frame = 0; frame < num_frames_; ++frame) {
        this->At(channel, frame) = ToDataType(matrix(channel, frame));
        ;
      }
    }
  }

 private:
  int num_frames_;
  int num_channels_;
  bool interleaved_;
  std::unique_ptr<AudioBufferList> buffer_list_;
  std::vector<std::unique_ptr<DataType[]>> buffers_;
};

template <>
float AudioBufferListWrapper<float>::ToDataType(float value) const {
  return value;
}

template <>
int16_t AudioBufferListWrapper<int16_t>::ToDataType(float value) const {
  return static_cast<int16_t>(value * std::numeric_limits<int16_t>::max());
}

@implementation MediaPipeAudioUtilTest

- (void)testBufferListToMatrixStereoNonInterleavedFloat {
  constexpr int kChannels = 2;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  AudioBufferListWrapper<float> bufferList(/*num_frames=*/kFrames,
                                           /*num_channels=*/kChannels,
                                           /*interleaved=*/false);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags =
          kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionFloat));
}

- (void)testBufferListToMatrixStereoInterleavedFloat {
  constexpr int kChannels = 2;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  AudioBufferListWrapper<float> bufferList(/*num_frames=*/kFrames,
                                           /*num_channels=*/kChannels,
                                           /*interleaved=*/true);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionFloat));
}

- (void)testBufferListToMatrixMonoNonInterleavedFloat {
  constexpr int kChannels = 1;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4;
  AudioBufferListWrapper<float> bufferList(/*num_frames=*/kFrames,
                                           /*num_channels=*/kChannels,
                                           /*interleaved=*/false);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags =
          kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionFloat));
}

- (void)testBufferListToMatrixMonoInterleavedFloat {
  constexpr int kChannels = 1;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4;
  AudioBufferListWrapper<float> bufferList(/*num_frames=*/kFrames,
                                           /*num_channels=*/kChannels,
                                           /*interleaved=*/true);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionFloat));
}

- (void)testBufferListToMatrixStereoNonInterleavedInt16 {
  constexpr int kChannels = 2;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  AudioBufferListWrapper<int16_t> bufferList(/*num_frames=*/kFrames,
                                             /*num_channels=*/kChannels,
                                             /*interleaved=*/false);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked |
                      kAudioFormatFlagIsNonInterleaved,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionInt16));
}

- (void)testBufferListToMatrixStereoInterleavedInt16 {
  constexpr int kChannels = 2;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  AudioBufferListWrapper<int16_t> bufferList(/*num_frames=*/kFrames,
                                             /*num_channels=*/kChannels,
                                             /*interleaved=*/true);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionInt16));
}

- (void)testBufferListToMatrixMonoNonInterleavedInt16 {
  constexpr int kChannels = 1;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4;
  AudioBufferListWrapper<int16_t> bufferList(/*num_frames=*/kFrames,
                                             /*num_channels=*/kChannels,
                                             /*interleaved=*/false);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked |
                      kAudioFormatFlagIsNonInterleaved,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionInt16));
}

- (void)testBufferListToMatrixMonoInterleavedInt16 {
  constexpr int kChannels = 1;
  constexpr int kFrames = 5;
  mediapipe::Matrix inputMatrix(kChannels, kFrames);
  inputMatrix << 0, 0.1, 0.2, 0.3, 0.4;
  AudioBufferListWrapper<int16_t> bufferList(/*num_frames=*/kFrames,
                                             /*num_channels=*/kChannels,
                                             /*interleaved=*/true);
  bufferList.InitFromMatrix(inputMatrix);

  static const AudioStreamBasicDescription kStreamDescription = {
      .mSampleRate = 44100,
      .mFormatID = kAudioFormatLinearPCM,
      .mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
      .mBytesPerPacket = bufferList.BytesPerPacket(),
      .mFramesPerPacket = kFrames,
      .mBytesPerFrame = bufferList.BytesPerSample() * kChannels,
      .mChannelsPerFrame = kChannels,
      .mBitsPerChannel = bufferList.BytesPerSample() * 8,
  };

  absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> matrix =
      MediaPipeConvertAudioBufferListToAudioMatrix(bufferList.GetBufferList(), &kStreamDescription,
                                                 static_cast<CMItemCount>(kFrames));
  if (!matrix.ok()) {
    XCTFail(@"Unable to convert a sample buffer list to a matrix: %s",
            matrix.status().ToString().c_str());
  }

  XCTAssertTrue(inputMatrix.isApprox(**matrix, kMatrixComparisonPrecisionInt16));
}

@end
