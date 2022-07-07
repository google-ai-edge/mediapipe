#import "MediaPipeController.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"

//#import "mediapipe/objc/MPPLayerRenderer.h"

static NSString* const kMeshGraphName = @"face_mesh_mobile_gpu";
static NSString* const kEffectsGraphName = @"face_effect_gpu";

static const char *kInputStream = "input_video";
static const char *kOutputStream = "output_video";
static const char *kNumFacesInputSidePacket = "num_faces";
static const char *kLandmarksOutputStream = "multi_face_landmarks";
static const char *kFaceRectsOutputStream = "face_rects_from_landmarks";
static const char *kLandmarkPresenceOutputStream = "landmark_presence";
static const char *kSelectedEffectIdInputStream = "selected_effect_id";
static const char *kMultiFaceGeometryStream = "multi_face_geometry";
static const char* kUseFaceDetectionInputSourceInputSidePacket = "use_face_detection_input_source";
static const BOOL kUseFaceDetectionInputSource = NO;

// Max number of faces to detect/process.
static const int kNumFaces = 2;

@interface MediaPipeController () <MPPGraphDelegate>
@property (nonatomic) MPPGraph* mediapipeGraph;
@property (nonatomic, copy) MediaPipeCompletionBlock completion;
@property (nonatomic) size_t timestamp;
@end

@implementation MediaPipeController

#pragma mark - Cleanup methods

- (void)dealloc {
    self.mediapipeGraph.delegate = nil;
    [self.mediapipeGraph cancel];
    // Ignore errors since we're cleaning up.
    [self.mediapipeGraph closeAllInputStreamsWithError:nil];
    [self.mediapipeGraph waitUntilDoneWithError:nil];
    
    NSLog(@"dealloc MediaPipeController");
}

#pragma mark - MediaPipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
    // Load the graph config resource.
    NSError* configLoadError = nil;
    NSBundle* bundle = [NSBundle bundleForClass:[self class]];
    if (!resource || resource.length == 0) {
        bundle = NSBundle.mainBundle;
    }
    NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
    NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
    if (!data) {
        NSLog(@"MediaPipe: Failed to load graph config: %@", configLoadError);
        return nil;
    }
    
    mediapipe::CalculatorGraphConfig config;
    config.ParseFromArray(data.bytes, data.length);
    
    MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
    [newGraph setSidePacket:(mediapipe::MakePacket<bool>(kUseFaceDetectionInputSource)) named:kUseFaceDetectionInputSourceInputSidePacket];
    [newGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces)) named:kNumFacesInputSidePacket];

    [newGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
    //[newGraph addFrameOutputStream:kMultiFaceGeometryStream outputPacketType:MPPPacketTypeRaw];
    
//    [newGraph addFrameOutputStream:kLandmarksOutputStream outputPacketType:MPPPacketTypeRaw];
//    [newGraph addFrameOutputStream:kFaceRectsOutputStream outputPacketType:MPPPacketTypeRaw];
//    [newGraph addFrameOutputStream:kLandmarkPresenceOutputStream outputPacketType:MPPPacketTypeRaw];
    return newGraph;
}

- (instancetype)initWithGraphName:(NSString *)graphName {
    self = [super init];
    if (self) {
        self.mediapipeGraph = [[self class] loadGraphFromResource:graphName];
        self.mediapipeGraph.delegate = self;
        
        // Set maxFramesInFlight to a small value to avoid memory contention for real-time processing.
        self.mediapipeGraph.maxFramesInFlight = 2;
        NSLog(@"MediaPipe: Inited graph %@", graphName);
        NSLog(@"alloc MediaPipeController");
    }
    return self;
}

+ (instancetype)facemesh {
    return [[MediaPipeController alloc] initWithGraphName:kMeshGraphName];
}

+ (instancetype)effects {
    return [[MediaPipeController alloc] initWithGraphName:kEffectsGraphName];
}

- (void)startGraph {
    NSError* error;
    if (![self.mediapipeGraph startWithError:&error]) {
        NSLog(@"MediaPipe: Failed to start graph: %@", error);
    }
    NSLog(@"MediaPipe: Started graph");
}

#pragma mark - MPPGraphDelegate methods

- (void)mediapipeGraph:(MPPGraph*)graph
  didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
            fromStream:(const std::string&)streamName {
    //NSLog(@"MediaPipe: didOutputPixelBuffer %s %@", streamName.c_str(), self.completion);
    if (streamName == kOutputStream) {
        if([self.delegate respondsToSelector:@selector(mediaPipeController:didOutputPixelBuffer:)]) {
            [_delegate mediaPipeController:self didOutputPixelBuffer:pixelBuffer];
        }
        if (self.completion) {
            self.completion(pixelBuffer);
        }
    }
}

- (void)mediapipeGraph:(MPPGraph*)graph
       didOutputPacket:(const ::mediapipe::Packet&)packet
            fromStream:(const std::string&)streamName {
    if (streamName == kLandmarksOutputStream) {
        if (packet.IsEmpty()) {
            return;
        }
        if(![self.delegate respondsToSelector:@selector(mediaPipeController:didReceiveFaces:)]) {
            return;
        }
        const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
        // NSLog(@"[TS:%lld] Number of face instances with landmarks: %lu", packet.Timestamp().Value(),
        // multi_face_landmarks.size());
        NSMutableArray <NSArray <MediaPipeFaceLandmarkPoint *>*>*faceLandmarks = [NSMutableArray new];
        
        for (int face_index = 0; face_index < multi_face_landmarks.size(); ++face_index) {
            NSMutableArray *thisFaceLandmarks = [NSMutableArray new];
            const auto& landmarks = multi_face_landmarks[face_index];
            //      NSLog(@"\tNumber of landmarks for face[%d]: %d", face_index, landmarks.landmark_size());
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                //        NSLog(@"\t\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
                //              landmarks.landmark(i).y(), landmarks.landmark(i).z());
                MediaPipeFaceLandmarkPoint *obj_landmark = [MediaPipeFaceLandmarkPoint new];
                obj_landmark.x = landmarks.landmark(i).x();
                obj_landmark.y = landmarks.landmark(i).y();
                obj_landmark.z = landmarks.landmark(i).z();
                [thisFaceLandmarks addObject:obj_landmark];
            }
            [faceLandmarks addObject:thisFaceLandmarks];
        }
        [self.delegate mediaPipeController:self didReceiveFaces:faceLandmarks];
    }
    
    else if (streamName == kFaceRectsOutputStream) {
        if (packet.IsEmpty()) { // This condition never gets called because FaceLandmarkFrontGpu does not process when there are no detections
            // NSLog(@"[TS:%lld] No face rects", packet.Timestamp().Value());
            if([self.delegate respondsToSelector:@selector(mediaPipeController:didReceiveFaceBoxes:)]) {
                [self.delegate mediaPipeController:self didReceiveFaceBoxes:@[]];
            }
            return;
        }
        if(![self.delegate respondsToSelector:@selector(mediaPipeController:didReceiveFaceBoxes:)]) {
            return;
        }
        const auto& face_rects_from_landmarks = packet.Get<std::vector<::mediapipe::NormalizedRect>>();
        NSMutableArray <MediaPipeNormalizedRect *>*outRects = [NSMutableArray new];
        for (int face_index = 0; face_index < face_rects_from_landmarks.size(); ++face_index) {
            const auto& face = face_rects_from_landmarks[face_index];
            float centerX = face.x_center();
            float centerY = face.y_center();
            float height = face.height();
            float width = face.width();
            float rotation = face.rotation();
            MediaPipeNormalizedRect *rect = [MediaPipeNormalizedRect new];
            rect.centerX = centerX; rect.centerY = centerY; rect.height = height; rect.width = width; rect.rotation = rotation;
            [outRects addObject:rect];
        }
        [self.delegate mediaPipeController:self didReceiveFaceBoxes:outRects];
    } else if (streamName == kLandmarkPresenceOutputStream) {
        bool is_landmark_present = true;
        if (packet.IsEmpty()) {
            is_landmark_present = false;
        } else {
            is_landmark_present = packet.Get<bool>();
        }
        
        if (is_landmark_present) {
        } else {
            // NSLog(@"Landmarks not present");
            if([self.delegate respondsToSelector:@selector(mediaPipeController:didReceiveFaceBoxes:)]) {
                [self.delegate mediaPipeController:self didReceiveFaceBoxes:@[]];
            }
            if([self.delegate respondsToSelector:@selector(mediaPipeController:didReceiveFaces:)]) {
                [self.delegate mediaPipeController:self didReceiveFaces:@[]];
            }
        }
    } else {
        //NSLog(@"MediaPipe: Unknown %@ packet with stream name %s", packet.IsEmpty() ? @"EMPTY" : @"NON-EMPTY",streamName.c_str());
    }
}


#pragma mark - MPPInputSourceDelegate methods

- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer timestamp:(CMTime)timestamp completion:(MediaPipeCompletionBlock)completion {
    const auto ts = mediapipe::Timestamp(self.timestamp++ * mediapipe::Timestamp::kTimestampUnitsPerSecond);
    self.completion = completion;
    
    NSError* err = nil;
    //NSLog(@"sending imageBuffer @%@ to %s", @(ts.DebugString().c_str()), kInputStream);
    auto sent = [self.mediapipeGraph sendPixelBuffer:imageBuffer
                                          intoStream:kInputStream
                                          packetType:MPPPacketTypePixelBuffer
                                           timestamp:ts
                                      allowOverwrite:NO
                                               error:&err
                 ];
    //NSLog(@"imageBuffer %s", sent ? "sent!" : "not sent.");
    if (err) {
        NSLog(@"MediaPipe: sendPixelBuffer error: %@", err);
    }
    
    mediapipe::Packet selectedEffectIdPacket = mediapipe::MakePacket<int>(2).At(ts);
    [self.mediapipeGraph movePacket:std::move(selectedEffectIdPacket)
                         intoStream:kSelectedEffectIdInputStream
                              error:nil];
}

@end


@implementation MediaPipeFaceLandmarkPoint
@end

@implementation MediaPipeNormalizedRect
@end
