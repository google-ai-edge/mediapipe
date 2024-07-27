#import "FaceMesh.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "opencv2/imgcodecs/ios.h" 

#import <UIKit/UIKit.h>

//#import "mediapipe/objc/MPPLayerRenderer.h"

// The graph name specified is supposed to be the same as in the pb file (binarypb?)
static NSString* const kGraphName = @"face_mesh_ios_lib_gpu";
// static NSString* const kGraphName = @"pure_face_mesh_mobile_gpu";

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
    //       multi_face_landmarks.size());
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

- (void)extractRegions:(NSURL *)fileName
            foreheadBoxes:(NSArray<NSArray<IntPoint *> *> *)foreheadBoxes
            leftCheekBoxes:(NSArray<NSArray<IntPoint *> *> *)leftCheekBoxes
            rightCheekBoxes:(NSArray<NSArray<IntPoint *> *> *)rightCheekBoxes
            totalFramesNeedProcess:(NSInteger)totalFramesNeedProcess
            skipNFirstFrames:(NSInteger)skipNFirstFrames {

    NSString *filename = fileName.path;
    std::string filePath = [filename UTF8String];

    cv::VideoCapture vid(filePath);

    if (!vid.isOpened()) {
        printf("@Error Opening video file");
    }
    else {
        printf("File Opened AAAA");

        // NSMutableArray *foreheadURLs = [NSMutableArray new];
        // NSMutableArray *leftcheekURLs = [NSMutableArray new];
        // NSMutableArray *rightcheekURLs = [NSMutableArray new];

        int startFrame = int(vid.get(cv::CAP_PROP_POS_FRAMES));
        int totalFrame = int(vid.get(cv::CAP_PROP_FRAME_COUNT));
        // int nframes = totalFrame - startFrame;

        // if (totalFramesNeedProcess > nframes) {
        //   NSLog(@"Video too short");
        //   return;
        // }

        if (skipNFirstFrames < 0 || totalFramesNeedProcess < 0) {
          vid.release();
          return;
        }

        int frameIdx = skipNFirstFrames;
        int maxFrameIndex = totalFramesNeedProcess;

        if (skipNFirstFrames > 0) {
          maxFrameIndex += skipNFirstFrames;
        }

        // Process forehead
        std::vector<cv::Mat> foreheads;
        std::vector<cv::Mat> leftcheeks;
        std::vector<cv::Mat> rightcheeks;
        
        while (frameIdx < maxFrameIndex) {
          cv::Mat frame;

          if (!vid.read(frame)) {
              NSLog(@"Failed to read frame.");
              break;
          }
          cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
          // Process the frame (e.g., display or save it)

          NSLog(@"frame index: %d", frameIdx);

          NSMutableArray *rowForehead = [foreheadBoxes objectAtIndex:(frameIdx - skipNFirstFrames)];
          NSMutableArray *rowLeftCheek = [leftCheekBoxes objectAtIndex:(frameIdx - skipNFirstFrames)];
          NSMutableArray *rowRightCheek = [rightCheekBoxes objectAtIndex:(frameIdx - skipNFirstFrames)];

          cv::Mat forehead = extractRegion(frame, rowForehead);
          cv::Mat leftCheek = extractRegion(frame, rowLeftCheek);
          cv::Mat rightCheek = extractRegion(frame, rowRightCheek);

          foreheads.push_back(forehead);
          leftcheeks.push_back(leftCheek);
          rightcheeks.push_back(rightCheek);
          
          frameIdx++;
        }

        NSLog(@"total foreheads: %d", foreheads.size());
        NSLog(@"total leftCheeks: %d", leftcheeks.size());
        NSLog(@"total rightCheeks: %d", rightcheeks.size());

        // for (int i = 0; i < foreheads.size(); i++) {
        //   cv::Mat rgbaImage;
        //   cv::cvtColor(foreheads[i], rgbaImage, cv::COLOR_RGB2RGBA);
        //   saveCVImageAsPNG(MatToUIImage(rgbaImage), @"forehead");
        // }

        NSMutableArray *foreheadURLs = saveCVImagesAsPNGs(foreheads, @"forehead");
        NSMutableArray *leftcheekURLs = saveCVImagesAsPNGs(leftcheeks, @"leftcheek");
        NSMutableArray *rightcheekURLs = saveCVImagesAsPNGs(rightcheeks, @"rightcheek");

        // cv::cvtColor(leftCheeks[0], rgbaImage, cv::COLOR_RGB2RGBA);
        // NSData *firstData = [NSData dataWithBytes:rgbaImage.data length:rgbaImage.total() * rgbaImage.elemSize()];
        // saveCVImageAsPNG([UIImage imageWithData:firstData], @"leftcheek");

        // cv::cvtColor(rightCheeks[0], rgbaImage, cv::COLOR_RGB2RGBA);
        // NSData *firstData = [NSData dataWithBytes:rgbaImage.data length:rgbaImage.total() * rgbaImage.elemSize()];
        // saveCVImageAsPNG([UIImage imageWithData:firstData], @"rightcheek");

        if([self.delegate respondsToSelector:@selector(didSavedRegions:leftcheekURLs:rightcheekURLs:)]) {
            NSLog(@"nguyencse ==> has didSavedRegions");
            [self.delegate didSavedRegions:foreheadURLs leftcheekURLs:leftcheekURLs rightcheekURLs:rightcheekURLs];
        }
    }

    vid.release();
}

cv::Mat extractRegion(cv::Mat img, NSArray<IntPoint *> *box) {
  IntPoint* point0 = [box objectAtIndex:0];
  IntPoint* point1 = [box objectAtIndex:1];
  IntPoint* point2 = [box objectAtIndex:2];
  IntPoint* point3 = [box objectAtIndex:3];

  // LEFT TOP --> RIGHT TOP --> RIGHT BOTTOM --> LEFT BOTTOM
  int frameWidth = point1.x - point0.x;
  int frameHeight = point3.y - point0.y;
  cv::Mat region = cropROI(img, cv::Rect(point0.x, point0.y, frameWidth, frameHeight));

  // square
  region = square(region);

  // resize to 32x32
  region = resizeWithAreaInterpolation(region, cv::Size(32, 32));

  return region;
}

cv::Mat cropROI(cv::Mat src, cv::Rect roi) {
  // Crop the full image to that image contained by the rectangle myROI
  // Note that this doesn't copy the data
  cv::Mat croppedRef(src, roi);

  cv::Mat cropped;
  // Copy the data into new matrix
  croppedRef.copyTo(cropped);

  return cropped;
}

cv::Mat square(cv::Mat frame) {
    if (frame.rows < frame.cols) {
        int diff = frame.cols - frame.rows;
        cv::Mat pad(frame.rows, diff, frame.type(), cv::Scalar(0));
        cv::hconcat(frame, pad, frame);
    } else if (frame.rows > frame.cols) {
        int diff = frame.rows - frame.cols;
        cv::Mat pad(diff, frame.cols, frame.type(), cv::Scalar(0));
        cv::vconcat(frame, pad, frame);
    }
    
    return frame;
}

cv::Mat resizeWithAreaInterpolation(cv::Mat image, cv::Size newShape) {
    cv::Size originalShape = image.size();
    cv::resize(image, image, newShape, 0, 0, cv::INTER_AREA);
    return image;
}

// NSURL *saveCVImageAsPNG(UIImage *image, NSString *fileName) {
//     // Create a unique file name for each image
//     NSString *fileNameIndex = [fileName stringByAppendingFormat:@"_%d", 0];
//     NSString *filePath = [NSTemporaryDirectory() stringByAppendingPathComponent:[fileNameIndex stringByAppendingString:@".png"]];
//     NSURL *fileURL = [NSURL fileURLWithPath:filePath];

//     NSData *pngData = UIImagePNGRepresentation(image);

//     if ([pngData writeToFile:filePath atomically:YES]) {
//         NSLog(@"PNG file saved successfully at path: %@", filePath);
//     } else {
//         NSLog(@"Failed to save PNG file at path: %@", filePath);
//     }

//     return fileURL;
// }

NSArray<NSURL *> *saveCVImagesAsPNGs(std::vector<cv::Mat> frames, NSString *folderName) {
    NSMutableArray<NSURL *> *fileURLs = [NSMutableArray arrayWithCapacity:frames.size()];

    for (int i = 0; i < frames.size(); i++) {
          // Create a unique file name for each image
        NSString *fileNameIndex = [folderName stringByAppendingFormat:@"_%d", i];
        NSString *filePath = [NSTemporaryDirectory() stringByAppendingPathComponent:[fileNameIndex stringByAppendingString:@".png"]];
        NSURL *fileURL = [NSURL fileURLWithPath:filePath];

        // NSLog(@"File URL: %@", fileURL);

        // Do NOT compress pixel
        std::vector<int> compressionParams;
        compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compressionParams.push_back(0);

        const char * cpath = [filePath cStringUsingEncoding:NSUTF8StringEncoding];
        const cv::String newPath = (const cv::String)cpath;
        cv::imwrite(newPath, frames[i], compressionParams);

        // Add file URL to the array
        [fileURLs addObject:fileURL];
    }

    return [fileURLs copy];
}

@end

 
@implementation NSValue (IntPoint)
+ (instancetype)valuewithIntPoint:(IntPoint *)value {
    return [self valueWithBytes:&value objCType:@encode(IntPoint)];
}
- (IntPoint *) intPointValue {
    IntPoint* value;
    [self getValue:&value];
    return value;
}
@end

@implementation FaceMeshLandmarkPoint
@end

@implementation FaceMeshNormalizedRect
@end

@implementation IntPoint
- (instancetype)initWithX:(NSInteger)x y:(NSInteger)y {
    self = [super init];
    if (self) {
        _x = x;
        _y = y;
    }
    return self;
}
@end