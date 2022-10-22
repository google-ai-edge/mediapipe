#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>

@class Landmark;
@class SYFaceMesh;

@protocol SYFaceMeshDelegate <NSObject>
- (void)faceMeshTracker: (SYFaceMesh*)faceMeshTracker didOutputLandmarks: (NSArray<Landmark *> *)landmarks;
- (void)faceMeshTracker: (SYFaceMesh*)faceMeshTracker didOutputPixelBuffer: (CVPixelBufferRef)pixelBuffer;
@end

@interface SYFaceMesh : NSObject
- (instancetype)init;
- (void)startGraph;
- (void)processVideoFrame: (CVPixelBufferRef)imageBuffer;
@property (weak, nonatomic) id <SYFaceMeshDelegate> delegate;
@end

@interface Landmark: NSObject
@property(nonatomic, readonly) float x;
@property(nonatomic, readonly) float y;
@property(nonatomic, readonly) float z;
@end