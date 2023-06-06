#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

@interface FaceMeshLandmarkPoint : NSObject
@property(nonatomic) float x;
@property(nonatomic) float y;
@property(nonatomic) float z;
@end

@interface FaceMeshNormalizedRect : NSObject
@property(nonatomic) float centerX;
@property(nonatomic) float centerY;
@property(nonatomic) float height;
@property(nonatomic) float width;
@property(nonatomic) float rotation;
@end

@protocol FaceMeshDelegate <NSObject>
@optional
/**
 * Array of faces, with faces represented by arrays of face landmarks
 */
- (void)didReceiveFaces:(NSArray<NSArray<FaceMeshLandmarkPoint *> *> *)faces;
@end

@interface FaceMesh : NSObject
- (instancetype)init;
- (void)startGraph;
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer;
- (CVPixelBufferRef)resize:(CVPixelBufferRef)pixelBuffer
                     width:(int)width
                    height:(int)height;
@property(weak, nonatomic) id<FaceMeshDelegate> delegate;
@property(nonatomic) size_t timestamp;
@end