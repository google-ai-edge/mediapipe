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

#import <UIKit/UIKit.h>
#import <XCTest/XCTest.h>

#include "absl/memory/memory.h"
#include "mediapipe/framework/formats/image.h"
#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPGraphTestBase.h"
#import "mediapipe/objc/NSError+util_status.h"
#import "mediapipe/objc/Weakify.h"
#import "mediapipe/objc/util.h"

static const char* kExpectedError = "Expected error.";

namespace mediapipe {

class GrayscaleCalculator : public Calculator {
 public:
  static absl::Status FillExpectations(const CalculatorOptions& options, PacketTypeSet* inputs,
                                       PacketTypeSet* outputs, PacketTypeSet* input_side_packets) {
    inputs->Index(0).Set<ImageFrame>();
    outputs->Index(0).Set<ImageFrame>();
    return absl::OkStatus();
  }

  absl::Status Process() final {
    const auto& input = Input()->Get<ImageFrame>();
    int w = input.Width();
    int h = input.Height();

    auto output = absl::make_unique<mediapipe::ImageFrame>(ImageFormat::GRAY8, w, h);

    vImage_Buffer src = vImageForImageFrame(input);
    vImage_Buffer dst = vImageForImageFrame(*output);
    vImage_Error vErr = vImageRGBAToGray(&src, &dst);
    NSCAssert(vErr == kvImageNoError, @"vImageRGBAToGray failed: %zd", vErr);

    Output()->Add(output.release(), InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(GrayscaleCalculator);

// For testing that video header is present. Open() will have a failure status
// if the video header is not present in the input stream.
class VideoHeaderCalculator : public Calculator {
 public:
  static absl::Status FillExpectations(const CalculatorOptions& options, PacketTypeSet* inputs,
                                       PacketTypeSet* outputs, PacketTypeSet* input_side_packets) {
    inputs->Index(0).Set<ImageFrame>();
    outputs->Index(0).Set<ImageFrame>();
    return absl::OkStatus();
  }

  absl::Status Open() final {
    if (Input()->Header().IsEmpty()) {
      return absl::UnknownError("No video header present.");
    }
    return absl::OkStatus();
  }

  absl::Status Process() final {
    Output()->AddPacket(Input()->Value());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(VideoHeaderCalculator);

class ErrorCalculator : public Calculator {
 public:
  static absl::Status FillExpectations(const CalculatorOptions& options, PacketTypeSet* inputs,
                                       PacketTypeSet* outputs, PacketTypeSet* input_side_packets) {
    inputs->Index(0).SetAny();
    outputs->Index(0).SetSameAs(&inputs->Index(0));
    return absl::OkStatus();
  }

  absl::Status Process() final { return absl::Status(absl::StatusCode::kUnknown, kExpectedError); }
};
REGISTER_CALCULATOR(ErrorCalculator);

}  // namespace mediapipe

@interface MPPGraphTests : MPPGraphTestBase{
  UIImage* _sourceImage;
  MPPGraph* _graph;
}
@end

@implementation MPPGraphTests

- (void)setUp {
  [super setUp];

  _sourceImage = [self testImageNamed:@"googlelogo_color_272x92dp" extension:@"png"];
}

- (void)tearDown {
  [super tearDown];
}

- (void)testPassThrough {
  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_frames");
  auto node = config.add_node();
  node->set_calculator("PassThroughCalculator");
  node->add_input_stream("input_frames");
  node->add_output_stream("output_frames");

  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"output_frames" outputPacketType:MPPPacketTypePixelBuffer];
  CFHolder<CVPixelBufferRef> inputBuffer;
  absl::Status status = CreateCVPixelBufferFromCGImage(_sourceImage.CGImage, &inputBuffer);
  XCTAssert(status.ok());
  CVPixelBufferRef outputBuffer = [self runGraph:_graph
                                 withPixelBuffer:*inputBuffer
                                      packetType:MPPPacketTypePixelBuffer];
  XCTAssert([self pixelBuffer:outputBuffer isEqualTo:*inputBuffer]);
}

- (UIImage*)grayImage:(UIImage*)inputImage {
  UIGraphicsBeginImageContextWithOptions(inputImage.size, YES, 1.0);
  CGRect imageRect = CGRectMake(0, 0, inputImage.size.width, inputImage.size.height);

  // Draw the image with the luminosity blend mode.
  // On top of a white background, this will give a black and white image.
  [inputImage drawInRect:imageRect blendMode:kCGBlendModeLuminosity alpha:1.0];

  UIImage *filteredImage = UIGraphicsGetImageFromCurrentImageContext();
  UIGraphicsEndImageContext();

  return filteredImage;
}

- (void)testMultipleOutputs {
  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_frames");
  auto passThroughNode = config.add_node();
  passThroughNode->set_calculator("PassThroughCalculator");
  passThroughNode->add_input_stream("input_frames");
  passThroughNode->add_output_stream("pass_frames");
  auto grayNode = config.add_node();
  grayNode->set_calculator("GrayscaleCalculator");
  grayNode->add_input_stream("input_frames");
  grayNode->add_output_stream("gray_frames");

  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"pass_frames" outputPacketType:MPPPacketTypeImageFrame];
  [_graph addFrameOutputStream:"gray_frames" outputPacketType:MPPPacketTypeImageFrame];

  CFHolder<CVPixelBufferRef> inputBuffer;
  absl::Status status = CreateCVPixelBufferFromCGImage(_sourceImage.CGImage, &inputBuffer);
  XCTAssert(status.ok());

  WEAKIFY(self);
  XCTestExpectation* passFrameReceive =
      [self expectationWithDescription:@"pass through output received"];
  XCTestExpectation* grayFrameReceive =
      [self expectationWithDescription:@"grayscale output received"];
  _pixelBufferOutputBlock = ^(MPPGraph* outputGraph, CVPixelBufferRef outputBuffer,
                              const std::string& outputStreamName) {
    STRONGIFY(self);
    XCTAssertEqualObjects(outputGraph, self->_graph);
    if (outputStreamName == "pass_frames") {
      [passFrameReceive fulfill];
    } else if (outputStreamName == "gray_frames") {
      [grayFrameReceive fulfill];
    }
  };

  [self runGraph:_graph withPixelBuffer:*inputBuffer packetType:MPPPacketTypeImageFrame];
}

- (void)testGrayscaleOutput {
  // When a calculator outputs a grayscale ImageFrame, it is returned to the
  // application as a BGRA pixel buffer. To test it, let's feed a grayscale
  // image into the graph and make sure it comes out unscathed.
  UIImage* grayImage = [self grayImage:_sourceImage];

  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_frames");
  auto node = config.add_node();
  node->set_calculator("GrayscaleCalculator");
  node->add_input_stream("input_frames");
  node->add_output_stream("output_frames");

  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"output_frames" outputPacketType:MPPPacketTypeImageFrame];
  CFHolder<CVPixelBufferRef> inputBuffer;
  absl::Status status = CreateCVPixelBufferFromCGImage(grayImage.CGImage, &inputBuffer);
  XCTAssert(status.ok());
  CVPixelBufferRef outputBuffer = [self runGraph:_graph
                                 withPixelBuffer:*inputBuffer
                                      packetType:MPPPacketTypeImageFrame];
  // We accept a small difference due to gamma correction and whatnot.
  XCTAssert([self pixelBuffer:outputBuffer isCloseTo:*inputBuffer
             maxLocalDifference:5 maxAverageDifference:FLT_MAX]);
}

- (void)testGraphError {
  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_frames");
  auto node = config.add_node();
  node->set_calculator("ErrorCalculator");
  node->add_input_stream("input_frames");
  node->add_output_stream("output_frames");
  CFHolder<CVPixelBufferRef> srcPixelBuffer;
  absl::Status status = CreateCVPixelBufferFromCGImage(_sourceImage.CGImage, &srcPixelBuffer);
  XCTAssert(status.ok());
  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"output_frames" outputPacketType:MPPPacketTypeImageFrame];
  _graph.delegate = self;

  XCTAssert([_graph startWithError:nil]);
  [_graph sendPixelBuffer:*srcPixelBuffer
               intoStream:"input_frames"
               packetType:MPPPacketTypeImageFrame];
  XCTAssert([_graph closeInputStream:"input_frames" error:nil]);

  __block NSError* error = nil;
  XCTestExpectation* graphDone = [self expectationWithDescription:@"graph done"];
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    XCTAssertFalse([_graph waitUntilDoneWithError:&error]);
    [graphDone fulfill];
  });

  [self waitForExpectationsWithTimeout:3.0 handler:NULL];
  XCTAssertNotNil(error);
  status = error.gus_status;
  XCTAssertNotEqual(status.message().find(kExpectedError), std::string::npos,
                    @"Missing expected std::string '%s' from error messge '%s'", kExpectedError,
                    status.message().data());
}

- (void)testSetStreamHeader {
  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_frames");
  auto node = config.add_node();
  node->set_calculator("VideoHeaderCalculator");
  node->add_input_stream("input_frames");
  node->add_output_stream("output_frames");

  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"output_frames" outputPacketType:MPPPacketTypeImageFrame];

  // We're no longer using video headers, let's just use an int as the header.
  auto header_packet = mediapipe::MakePacket<int>(0xDEADBEEF);
  [_graph setHeaderPacket:header_packet forStream:"input_frames"];

  // Verify that Open() on calculator succeeded.
  XCTAssert([_graph startWithError:nil]);

  // Tear down graph.
  XCTAssert([_graph closeInputStream:"input_frames" error:nil]);
  XCTestExpectation* graphDone = [self expectationWithDescription:@"graph done"];
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    XCTAssert([_graph waitUntilDoneWithError:nil]);
    [graphDone fulfill];
  });

  [self waitForExpectationsWithTimeout:3.0 handler:NULL];
}

- (void)testGraphIsDeallocated {
  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_frames");
  auto node = config.add_node();
  node->set_calculator("PassThroughCalculator");
  node->add_input_stream("input_frames");
  node->add_output_stream("output_frames");

  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"output_frames" outputPacketType:MPPPacketTypePixelBuffer];
  CFHolder<CVPixelBufferRef> inputBuffer;
  absl::Status status = CreateCVPixelBufferFromCGImage(_sourceImage.CGImage, &inputBuffer);
  XCTAssert(status.ok());
  [self runGraph:_graph withPixelBuffer:*inputBuffer packetType:MPPPacketTypePixelBuffer];
  __weak MPPGraph* weakGraph = _graph;
  _graph = nil;
  XCTAssertNil(weakGraph);
}

- (void)testRawPacketOutput {
  mediapipe::CalculatorGraphConfig config;
  config.add_input_stream("input_ints");
  auto node = config.add_node();
  node->set_calculator("PassThroughCalculator");
  node->add_input_stream("input_ints");
  node->add_output_stream("output_ints");

  const int kTestValue = 10;

  _graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [_graph addFrameOutputStream:"output_ints" outputPacketType:MPPPacketTypeRaw];
  _graph.delegate = self;

  WEAKIFY(self);
  XCTestExpectation* outputReceived = [self expectationWithDescription:@"output received"];
  _packetOutputBlock = ^(MPPGraph* outputGraph, const mediapipe::Packet& packet,
                         const std::string& outputStreamName) {
    STRONGIFY(self);
    XCTAssertEqualObjects(outputGraph, _graph);
    XCTAssertEqual(outputStreamName, "output_ints");
    XCTAssertEqual(packet.Get<int>(), kTestValue);
    [outputReceived fulfill];
  };

  XCTAssert([_graph startWithError:nil]);
  XCTAssert([_graph sendPacket:mediapipe::MakePacket<int>(kTestValue).At(mediapipe::Timestamp(1))
                    intoStream:"input_ints"
                         error:nil]);
  XCTAssert([_graph closeInputStream:"input_ints" error:nil]);
  XCTestExpectation* graphDone = [self expectationWithDescription:@"graph done"];
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    XCTAssert([_graph waitUntilDoneWithError:nil]);
    [graphDone fulfill];
  });

  [self waitForExpectationsWithTimeout:3.0 handler:NULL];
}

- (void)testPixelBufferToImage {
  CFHolder<CVPixelBufferRef> pixelBufferIn;
  absl::Status status = CreateCVPixelBufferFromCGImage(_sourceImage.CGImage, &pixelBufferIn);
  XCTAssert(status.ok());

  mediapipe::CalculatorGraphConfig config;
  _graph = [[MPPGraph alloc] initWithGraphConfig:config];

  mediapipe::Packet packet = [_graph imagePacketWithPixelBuffer:*pixelBufferIn];
  CVPixelBufferRef pixelBufferOut = packet.Get<mediapipe::Image>().GetCVPixelBufferRef();

  XCTAssertTrue([self pixelBuffer:*pixelBufferIn
                        isCloseTo:pixelBufferOut
               maxLocalDifference:0
             maxAverageDifference:0]);
}

@end
