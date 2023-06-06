#import "FaceMesh.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

//#import "mediapipe/objc/MPPLayerRenderer.h"

// The graph name specified is supposed to be the same as in the pb file (binarypb?)
static NSString* const kGraphName = @"face_mesh_ios_lib_gpu";

static const char* kInputStream = "input_video";
static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kLandmarksOutputStream = "multi_face_landmarks";

// Max number of faces to detect/process.
static const int kNumFaces = 1;


@interface FaceMesh () <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end

@implementation FaceMesh {}

#pragma mark - Cleanup methods

- (void)dealloc {
  self.mediapipeGraph.delegate = nil;
  [self.mediapipeGraph cancel];
  // Ignore errors since we're cleaning up.
  [self.mediapipeGraph closeAllInputStreamsWithError:nil];
  [self.mediapipeGraph waitUntilDoneWithError:nil];
}

#pragma mark - MediaPipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
  // Load the graph config resource.
  NSError* configLoadError = nil;
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  if (!resource || resource.length == 0) {
    return nil;
  }
  NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
  NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
  if (!data) {
    NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
    return nil;
  }

  // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
  mediapipe::CalculatorGraphConfig config;
  config.ParseFromArray(data.bytes, data.length);

  // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
  MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
  
  // Set graph configurations
  [newGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces))
                              named:kNumFacesInputSidePacket];

  [newGraph addFrameOutputStream:kLandmarksOutputStream
                outputPacketType:MPPPacketTypeRaw];
  return newGraph;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
    self.mediapipeGraph.delegate = self;
    
    // // Set maxFramesInFlight to a small value to avoid memory contention
    // // for real-time processing.
    // self.mediapipeGraph.maxFramesInFlight = 2;
    NSLog(@"inited graph %@", kGraphName);
  }
  return self;
}

- (void)startGraph {
  NSError* error;
  if (![self.mediapipeGraph startWithError:&error]) {
    NSLog(@"Failed to start graph: %@", error);
  }
  NSLog(@"Started graph %@", kGraphName);
}

#pragma mark - MPPGraphDelegate methods

// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string&)streamName {
  NSLog(@"recv pixelBuffer from %@", @(streamName.c_str()));
}

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  if (streamName == kLandmarksOutputStream) {
    if (packet.IsEmpty()) { // This condition never gets called because FaceLandmarkFrontGpu does not process when there are no detections
      return;
    }
    const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
    // NSLog(@"[TS:%lld] Number of face instances with landmarks: %lu", packet.Timestamp().Value(),
          // multi_face_landmarks.size());
    NSMutableArray <NSArray <FaceMeshLandmarkPoint *>*>*faceLandmarks = [NSMutableArray new];
    
    for (int face_index = 0; face_index < multi_face_landmarks.size(); ++face_index) {
      NSMutableArray *thisFaceLandmarks = [NSMutableArray new];
      const auto& landmarks = multi_face_landmarks[face_index];
//      NSLog(@"\tNumber of landmarks for face[%d]: %d", face_index, landmarks.landmark_size());
      for (int i = 0; i < landmarks.landmark_size(); ++i) {
//        NSLog(@"\t\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
//              landmarks.landmark(i).y(), landmarks.landmark(i).z());
        FaceMeshLandmarkPoint *obj_landmark = [FaceMeshLandmarkPoint new];
        obj_landmark.x = landmarks.landmark(i).x();
        obj_landmark.y = landmarks.landmark(i).y();
        obj_landmark.z = landmarks.landmark(i).z();
        [thisFaceLandmarks addObject:obj_landmark];
      }
      [faceLandmarks addObject:thisFaceLandmarks];
    }
    if([self.delegate respondsToSelector:@selector(didReceiveFaces:)]) {
      [self.delegate didReceiveFaces:faceLandmarks];
    }
  } else {
    NSLog(@"Unknown %@ packet with stream name %s", packet.IsEmpty() ? @"EMPTY" : @"NON-EMPTY",streamName.c_str());
  }
}


#pragma mark - MPPInputSourceDelegate methods

- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer {
  const auto ts =
      mediapipe::Timestamp(self.timestamp++ * mediapipe::Timestamp::kTimestampUnitsPerSecond);
  NSError* err = nil;
  // NSLog(@"sending imageBuffer @%@ to %s", @(ts.DebugString().c_str()), kInputStream);
  auto sent = [self.mediapipeGraph sendPixelBuffer:imageBuffer
                                        intoStream:kInputStream
                                        packetType:MPPPacketTypePixelBuffer
                                         timestamp:ts
                                    allowOverwrite:NO
                                             error:&err];
  // NSLog(@"imageBuffer %s", sent ? "sent!" : "not sent.");
  if (err) {
    NSLog(@"sendPixelBuffer error: %@", err);
  }
}

// Resize the CVPixelBufferRef with INTER_AREA.
- (CVPixelBufferRef)resize:(CVPixelBufferRef)pixelBuffer
                    width:(int)width
                    height:(int)height {
  
  OSType srcType = CVPixelBufferGetPixelFormatType(pixelBuffer);
  size_t channels = 2;
  if (srcType == kCVPixelFormatType_32ARGB || srcType == kCVPixelFormatType_32BGRA) {
    channels = 4;
  }
  
  // Lock the CVPixelBuffer
  CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  // Get the pixel buffer attributes
  size_t srcWidth = CVPixelBufferGetWidth(pixelBuffer);
  size_t srcHeight = CVPixelBufferGetHeight(pixelBuffer);
  size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

  // Get the base address of the pixel buffer
  unsigned char *baseAddress = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

  // Create a cv::Mat without copying the data
  cv::Mat argbImage(srcHeight, srcWidth, CV_8UC(channels), baseAddress, bytesPerRow);

  // Create a cv::Mat to hold the resized image
  cv::Mat resizedImage;

  // Resize the image using cv::resize
  cv::resize(argbImage, resizedImage, cv::Size(width, height), 0, 0, cv::INTER_AREA);

  // Unlock the CVPixelBuffer
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  // Create a new CVPixelBuffer with the desired size and format
  CVPixelBufferRef resizedPixelBuffer;
  CVReturn result = CVPixelBufferCreate(NULL, width, height, srcType, NULL, &resizedPixelBuffer);
  if (result != kCVReturnSuccess) {
      NSLog(@"Failed to create CVPixelBuffer. Error: %d", result);
      return nil;
  }

  // Lock the resized CVPixelBuffer
  CVPixelBufferLockBaseAddress(resizedPixelBuffer, 0);

  // Get the base address and bytes per row of the resized pixel buffer
  void *resizedBaseAddress = CVPixelBufferGetBaseAddress(resizedPixelBuffer);
  size_t resizedBytesPerRow = CVPixelBufferGetBytesPerRow(resizedPixelBuffer);

  // Create a cv::Mat wrapper for the resized pixel buffer
  cv::Mat resizedPixelBufferMat(height, width, CV_8UC(channels), resizedBaseAddress, resizedBytesPerRow);

  // Convert the resized image (cv::Mat) to the resized pixel buffer (CVPixelBuffer)
  resizedImage.copyTo(resizedPixelBufferMat);

  // Unlock the resized CVPixelBuffer
  CVPixelBufferUnlockBaseAddress(resizedPixelBuffer, 0);

  // Return the resized CVPixelBuffer
  return resizedPixelBuffer;
}

@end


@implementation FaceMeshLandmarkPoint
@end

@implementation FaceMeshNormalizedRect
@end
