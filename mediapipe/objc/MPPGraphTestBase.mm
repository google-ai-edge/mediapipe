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

#import "mediapipe/objc/MPPGraphTestBase.h"
#import "mediapipe/objc/Weakify.h"

#include "absl/memory/memory.h"

static UIImage* UIImageWithPixelBuffer(CVPixelBufferRef pixelBuffer) {
  CFHolder<CGImageRef> cgImage;
  absl::Status status = CreateCGImageFromCVPixelBuffer(pixelBuffer, &cgImage);
  if (!status.ok()) {
    return nil;
  }
  UIImage *uiImage =  [UIImage imageWithCGImage:*cgImage
                                          scale:1.0
                                    orientation:UIImageOrientationUp];
  return uiImage;
}

static void EnsureOutputDirFor(NSString *outputFile) {
  NSFileManager *fileManager = [NSFileManager defaultManager];
  NSError *error = nil;
  BOOL result = [fileManager createDirectoryAtPath:[outputFile stringByDeletingLastPathComponent]
                       withIntermediateDirectories:YES
                                        attributes:nil
                                             error:&error];
  // TODO: Log the error for clarity. The file-write will fail later
  // but it would be nice to see this error. However, 'error' is still testing
  // false and result is true even on an unwritable path-- not sure what's up.
}

@implementation MPPGraphTestBase

- (NSURL*)URLForTestFile:(NSString*)file extension:(NSString*)extension {
  NSBundle* testBundle = [NSBundle bundleForClass:[self class]];
  return [testBundle URLForResource:file withExtension:extension];
}

- (NSData*)testDataNamed:(NSString*)name extension:(NSString*)extension {
  NSURL* resourceURL = [self URLForTestFile:name extension:extension];
  XCTAssertNotNil(resourceURL,
      @"Unable to find data with name: %@.  Did you add it to your resources?", name);
  NSError* error;
  NSData* data = [NSData dataWithContentsOfURL:resourceURL options:0 error:&error];
  XCTAssertNotNil(data, @"%@: %@", resourceURL.path, error);
  return data;
}

- (UIImage*)testImageNamed:(NSString*)name extension:(NSString*)extension {
  return [self testImageNamed:name extension:extension subdirectory:nil];
}

- (UIImage*)testImageNamed:(NSString*)name
                 extension:(NSString*)extension
              subdirectory:(NSString *)subdirectory {
  // imageNamed does not work in our test bundle
  NSBundle* testBundle = [NSBundle bundleForClass:[self class]];
  NSURL* imageURL = subdirectory ?
      [testBundle URLForResource:name withExtension:extension subdirectory:subdirectory] :
      [testBundle URLForResource:name withExtension:extension];
  XCTAssertNotNil(imageURL,
                  @"Unable to find image with name: %@.  Did you add it to your resources?", name);
  NSError* error;
  NSData* imageData = [NSData dataWithContentsOfURL:imageURL options:0 error:&error];
  UIImage* image = [UIImage imageWithData:imageData];
  XCTAssertNotNil(image, @"%@: %@", imageURL.path, error);
  return image;
}

- (CVPixelBufferRef)runGraph:(MPPGraph*)graph
       withInputPixelBuffers:
           (const std::unordered_map<std::string, CFHolder<CVPixelBufferRef>>&)inputBuffers
                inputPackets:(const std::map<std::string, mediapipe::Packet>&)inputPackets
                   timestamp:(mediapipe::Timestamp)timestamp
                outputStream:(const std::string&)outputStream
                  packetType:(MPPPacketType)inputPacketType {
  __block CVPixelBufferRef output;
  graph.delegate = self;

  // The XCTAssert macros contain references to self, which causes a retain cycle,
  // since the block retains self and self retains the block. The cycle is broken
  // at the end of this method, with _pixelBufferOutputBlock = nil, but Clang does
  // not realize that and outputs a warning. WEAKIFY and STRONGIFY, though not
  // strictly necessary, are used here to avoid the warning.
  WEAKIFY(self);
  if (!_pixelBufferOutputBlock) {
    XCTestExpectation* outputReceived = [self expectationWithDescription:@"output received"];
    _pixelBufferOutputBlock = ^(MPPGraph* outputGraph, CVPixelBufferRef outputBuffer,
                                const std::string& outputStreamName) {
      STRONGIFY(self);
      XCTAssertEqualObjects(outputGraph, graph);
      XCTAssertEqual(outputStreamName, outputStream);
      CFRetain(outputBuffer);
      output = outputBuffer;
      [outputReceived fulfill];
    };
  }

  NSError *error;
  BOOL success = [graph startWithError:&error];
  // Normally we continue after failures, but there is no sense in waiting for an
  // output if the graph didn't even start.
  BOOL savedContinue = self.continueAfterFailure;
  self.continueAfterFailure = NO;
  XCTAssert(success, @"%@", error.localizedDescription);
  self.continueAfterFailure = savedContinue;
  for (const auto& stream_buffer : inputBuffers) {
    [graph sendPixelBuffer:*stream_buffer.second
                intoStream:stream_buffer.first
                packetType:inputPacketType
                 timestamp:timestamp];
    success = [graph closeInputStream:stream_buffer.first error:&error];
    XCTAssert(success, @"%@", error.localizedDescription);
  }
  for (const auto& stream_packet : inputPackets) {
    [graph sendPacket:stream_packet.second
           intoStream:stream_packet.first
                error:&error];
    success = [graph closeInputStream:stream_packet.first error:&error];
    XCTAssert(success, @"%@", error.localizedDescription);
  }

  XCTestExpectation* graphDone = [self expectationWithDescription:@"graph done"];
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSError *error;
    BOOL success = [graph waitUntilDoneWithError:&error];
    XCTAssert(success, @"%@", error.localizedDescription);
    [graphDone fulfill];
  });

  [self waitForExpectationsWithTimeout:8.0 handler:NULL];
  _pixelBufferOutputBlock = nil;
  return output;
}

- (CVPixelBufferRef)runGraph:(MPPGraph*)graph
             withPixelBuffer:(CVPixelBufferRef)inputBuffer
                  packetType:(MPPPacketType)inputPacketType {
  return [self runGraph:graph
      withInputPixelBuffers:{{"input_frames", MakeCFHolder(inputBuffer)}}
               inputPackets:{}
                  timestamp:mediapipe::Timestamp(1)
               outputStream:"output_frames"
                 packetType:inputPacketType];
}

- (CVPixelBufferRef)runGraph:(MPPGraph*)graph
       withInputPixelBuffers:
           (const std::unordered_map<std::string, CFHolder<CVPixelBufferRef>>&)inputBuffers
                outputStream:(const std::string&)output
                  packetType:(MPPPacketType)inputPacketType {
  return [self runGraph:graph
               withInputPixelBuffers:inputBuffers
           inputPackets:{}
              timestamp:mediapipe::Timestamp(1)
           outputStream:output
             packetType:inputPacketType];
}

// By using a block to handle the delegate message, we can change the
// implementation for each test.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)imageBuffer
              fromStream:(const std::string&)streamName {
  _pixelBufferOutputBlock(graph, imageBuffer, streamName);
}

- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  _packetOutputBlock(graph, packet, streamName);
}

- (BOOL)pixelBuffer:(CVPixelBufferRef)a isEqualTo:(CVPixelBufferRef)b {
  return [self pixelBuffer:a isCloseTo:b maxLocalDifference:0 maxAverageDifference:0];
}

- (BOOL)pixelBuffer:(CVPixelBufferRef)a
               isCloseTo:(CVPixelBufferRef)b
      maxLocalDifference:(int)maxLocalDiff
    maxAverageDifference:(float)maxAvgDiff {
  return [self pixelBuffer:a
                    isCloseTo:b
           maxLocalDifference:maxLocalDiff
         maxAverageDifference:maxAvgDiff
        maxLocalDifferenceOut:nil
      maxAverageDifferenceOut:nil];
}

- (BOOL)pixelBuffer:(CVPixelBufferRef)a
                  isCloseTo:(CVPixelBufferRef)b
         maxLocalDifference:(int)maxLocalDiff
       maxAverageDifference:(float)maxAvgDiff
      maxLocalDifferenceOut:(int*)maxLocalDiffOut
    maxAverageDifferenceOut:(float*)maxAvgDiffOut {
  size_t aBytesPerRow = CVPixelBufferGetBytesPerRow(a);
  size_t aWidth = CVPixelBufferGetWidth(a);
  size_t aHeight = CVPixelBufferGetHeight(a);
  OSType aPixelFormat = CVPixelBufferGetPixelFormatType(a);
  XCTAssertFalse(CVPixelBufferIsPlanar(a), @"planar buffers not supported");

  size_t bBytesPerRow = CVPixelBufferGetBytesPerRow(b);
  size_t bWidth = CVPixelBufferGetWidth(b);
  size_t bHeight = CVPixelBufferGetHeight(b);
  OSType bPixelFormat = CVPixelBufferGetPixelFormatType(b);
  XCTAssertFalse(CVPixelBufferIsPlanar(b), @"planar buffers not supported");

  if (aPixelFormat != bPixelFormat ||
      aWidth != bWidth ||
      aHeight != bHeight) return NO;

  size_t bytesPerPixel;  // is there a generic way to get this from a pixel buffer?
  switch (aPixelFormat) {
    case kCVPixelFormatType_32BGRA:
      bytesPerPixel = 4;
      break;
    case kCVPixelFormatType_OneComponent8:
      bytesPerPixel = 1;
      break;
    default:
      XCTFail(@"unsupported pixel format");
  }

  CVReturn err;
  err = CVPixelBufferLockBaseAddress(a, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(err, kCVReturnSuccess);
  err = CVPixelBufferLockBaseAddress(b, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(err, kCVReturnSuccess);
  const uint8_t* aData = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(a));
  const uint8_t* bData = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(b));

  // Let's not assume identical bytesPerRow. Also, the padding may not be equal
  // even if bytesPerRow match.
  size_t usedRowWidth = aWidth * bytesPerPixel;
  BOOL equal = YES;
  float count = 0;
  BOOL canSkipSomeDiffs = !maxLocalDiffOut && !maxAvgDiffOut;
  int computedMaxLocalDiff = 0;
  float computedAvgDiff = 0;
  for (int i = aHeight; i > 0 && equal; --i) {
    if (maxLocalDiff == 0 && canSkipSomeDiffs) {
      // If we can, use memcmp for speed.
      equal = memcmp(aData, bData, usedRowWidth) == 0;
    } else {
      for (int j = 0; j < usedRowWidth; j++) {
        int diff = abs(aData[j] - bData[j]);
        computedMaxLocalDiff = MAX(computedMaxLocalDiff, diff);
        if (diff > maxLocalDiff && canSkipSomeDiffs) {
          break;
        }
        // We use Welford's algorithm for computing a sample mean. This has better
        // numerical stability than the naive method, as noted in TAoCP. Not that it
        // particularly matters here.
        // Welford: http://www.jstor.org/stable/1266577
        // Knuth: The Art of Computer Programming Vol 2, section 4.2.2
        computedAvgDiff += (diff - computedAvgDiff) / ++count;
      }
    }
    aData += aBytesPerRow;
    bData += bBytesPerRow;
  }
  if (computedMaxLocalDiff > maxLocalDiff || computedAvgDiff > maxAvgDiff) {
    equal = NO;
  }
  if (maxLocalDiffOut) {
    *maxLocalDiffOut = computedMaxLocalDiff;
  }
  if (maxAvgDiffOut) {
    *maxAvgDiffOut = computedAvgDiff;
  }

  err = CVPixelBufferUnlockBaseAddress(b, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(err, kCVReturnSuccess);
  err = CVPixelBufferUnlockBaseAddress(a, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(err, kCVReturnSuccess);
  return equal;
}

- (CVPixelBufferRef)convertPixelBuffer:(CVPixelBufferRef)input
                         toPixelFormat:(OSType)pixelFormat {
  size_t width = CVPixelBufferGetWidth(input);
  size_t height = CVPixelBufferGetHeight(input);
  CVPixelBufferRef output;
  CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, pixelFormat,
      GetCVPixelBufferAttributesForGlCompatibility(), &output);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferLockBaseAddress(input, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferLockBaseAddress(output, 0);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = vImageConvertCVPixelBuffers(input, output);
  XCTAssertEqual(status, kvImageNoError);

  status = CVPixelBufferUnlockBaseAddress(output, 0);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferUnlockBaseAddress(input, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(status, kCVReturnSuccess);

  return output;
}

- (CVPixelBufferRef)scaleBGRAPixelBuffer:(CVPixelBufferRef)input
                                  toSize:(CGSize)size {
  CVPixelBufferRef output;
  CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, size.width, size.height, kCVPixelFormatType_32BGRA,
      GetCVPixelBufferAttributesForGlCompatibility(), &output);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferLockBaseAddress(input, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferLockBaseAddress(output, 0);
  XCTAssertEqual(status, kCVReturnSuccess);

  vImage_Buffer src = vImageForCVPixelBuffer(input);
  vImage_Buffer dst = vImageForCVPixelBuffer(output);
  status = vImageScale_ARGB8888(&src, &dst, NULL, kvImageNoFlags);
  XCTAssertEqual(status, kvImageNoError);

  status = CVPixelBufferUnlockBaseAddress(output, 0);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferUnlockBaseAddress(input, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(status, kCVReturnSuccess);

  return output;
}

- (CVPixelBufferRef)transformPixelBuffer:(CVPixelBufferRef)input
                       outputPixelFormat:(OSType)pixelFormat
                          transformation:(void(^)(CVPixelBufferRef input,
                                                  CVPixelBufferRef output))transformation {
  size_t width = CVPixelBufferGetWidth(input);
  size_t height = CVPixelBufferGetHeight(input);
  CVPixelBufferRef output;
  CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height,
      pixelFormat, NULL, &output);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferLockBaseAddress(input, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferLockBaseAddress(output, 0);
  XCTAssertEqual(status, kCVReturnSuccess);

  transformation(input, output);

  status = CVPixelBufferUnlockBaseAddress(output, 0);
  XCTAssertEqual(status, kCVReturnSuccess);

  status = CVPixelBufferUnlockBaseAddress(input, kCVPixelBufferLock_ReadOnly);
  XCTAssertEqual(status, kCVReturnSuccess);

  return output;
}

- (UIImage*)differenceOfImage:(UIImage*)inputA image:(UIImage*)inputB {
  UIGraphicsBeginImageContextWithOptions(inputA.size, YES, 1.0);
  CGRect imageRect = CGRectMake(0, 0, inputA.size.width, inputA.size.height);

  [inputA drawInRect:imageRect blendMode:kCGBlendModeNormal alpha:1.0];
  [inputB drawInRect:imageRect blendMode:kCGBlendModeDifference alpha:1.0];

  UIImage *differenceImage = UIGraphicsGetImageFromCurrentImageContext();
  UIGraphicsEndImageContext();

  return differenceImage;
}

- (void)testGraph:(MPPGraph*)graph
             input:(CVPixelBufferRef)inputBuffer
    expectedOutput:(CVPixelBufferRef)expectedBuffer
{
  CVPixelBufferRef outputBuffer = [self runGraph:graph
                                 withPixelBuffer:inputBuffer
                                      packetType:MPPPacketTypePixelBuffer];
#if DEBUG
  // Xcode can display UIImage objects right in the debugger. It is handy to
  // have these variables defined if the test fails.
  UIImage* output = UIImageWithPixelBuffer(outputBuffer);
  XCTAssertNotNil(output);
  UIImage* expected = UIImageWithPixelBuffer(expectedBuffer);
  XCTAssertNotNil(expected);
  UIImage* diff = [self differenceOfImage:output image:expected];
  (void)diff;  // Suppress unused variable warning.
#endif
  XCTAssert([self pixelBuffer:outputBuffer isCloseTo:expectedBuffer
                maxLocalDifference:INT_MAX maxAverageDifference:1]);
  CFRelease(outputBuffer);
}

- (void)testGraphConfig:(const mediapipe::CalculatorGraphConfig&)config
    inputStreamsAndFiles:(NSDictionary<NSString*, NSString*>*)inputs
            outputStream:(NSString*)outputStream
      expectedOutputFile:(NSString*)expectedPath {
  [self testGraphConfig:config
      inputStreamsAndFiles:inputs
    inputStreamsAndPackets:{}
               sidePackets:{}
                 timestamp:mediapipe::Timestamp(1)
              outputStream:outputStream
        expectedOutputFile:expectedPath
      maxAverageDifference:1.f];
}

- (void)testGraphConfig:(const mediapipe::CalculatorGraphConfig&)config
      inputStreamsAndFiles:(NSDictionary<NSString*, NSString*>*)fileInputs
    inputStreamsAndPackets:(const std::map<std::string, mediapipe::Packet>&)packetInputs
               sidePackets:(std::map<std::string, mediapipe::Packet>)sidePackets
                 timestamp:(mediapipe::Timestamp)timestamp
              outputStream:(NSString*)outputStream
        expectedOutputFile:(NSString*)expectedPath
      maxAverageDifference:(float)maxAverageDifference {
  NSBundle* testBundle = [NSBundle bundleForClass:[self class]];
  chdir([testBundle.resourcePath fileSystemRepresentation]);
  MPPGraph* graph = [[MPPGraph alloc] initWithGraphConfig:config];
  [graph addSidePackets:sidePackets];
  [graph addFrameOutputStream:outputStream.UTF8String
             outputPacketType:MPPPacketTypePixelBuffer];

  std::unordered_map<std::string, CFHolder<CVPixelBufferRef>> inputBuffers;
  for (NSString* inputStream in fileInputs) {
    UIImage* inputImage = [self testImageNamed:fileInputs[inputStream] extension:nil];
    XCTAssertNotNil(inputImage);
    absl::Status status =
        CreateCVPixelBufferFromCGImage(inputImage.CGImage, &inputBuffers[inputStream.UTF8String]);
    XCTAssert(status.ok());
  }

  UIImage* expectedImage = [self testImageNamed:expectedPath extension:nil];
  XCTAssertNotNil(expectedImage);
  CFHolder<CVPixelBufferRef> expectedBuffer;
  absl::Status status = CreateCVPixelBufferFromCGImage(expectedImage.CGImage, &expectedBuffer);
  XCTAssert(status.ok());

  CVPixelBufferRef outputBuffer = [self runGraph:graph
                           withInputPixelBuffers:inputBuffers
                                    inputPackets:packetInputs
                                       timestamp:timestamp
                                    outputStream:outputStream.UTF8String
                                      packetType:MPPPacketTypePixelBuffer];

  UIImage* output = UIImageWithPixelBuffer(outputBuffer);
  XCTAssertNotNil(output);

  UIImage* expected = UIImageWithPixelBuffer(*expectedBuffer);
  XCTAssertNotNil(expected);
  UIImage* diff = [self differenceOfImage:output image:expected];

  XCTAssert([self pixelBuffer:outputBuffer isCloseTo:*expectedBuffer
           maxLocalDifference:INT_MAX maxAverageDifference:maxAverageDifference]);

  CFRelease(outputBuffer);
}

@end
