#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

@interface IntPoint : NSObject
@property(nonatomic) NSInteger x;
@property(nonatomic) NSInteger y;

- (instancetype)initWithX:(NSInteger)x y:(NSInteger)y;
@end

@interface NSValue (IntPoint)
+ (instancetype)valuewithIntPoint:(IntPoint *)value;
@property (readonly) IntPoint* intPointValue;
@end

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

- (void)didSavedRegions:(NSArray<NSURL *> *)foreheadURLs
          leftcheekURLs:(NSArray<NSURL *> *)leftcheekURLs
          rightcheekURLs:(NSArray<NSURL *> *)rightcheekURLs;
@end

@interface FaceMesh : NSObject
- (instancetype)init;
- (void)startGraph;
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer;
- (CVPixelBufferRef)resize:(CVPixelBufferRef)pixelBuffer
                     width:(int)width
                    height:(int)height;
- (uint8_t **)buffer2Array2D:(CVPixelBufferRef)pixelBuffer;
- (void)extractRegions:(NSURL *)fileName
            foreheadBoxes:(NSArray<NSArray<IntPoint *> *> *)foreheadBoxes
            leftCheekBoxes:(NSArray<NSArray<IntPoint *> *> *)leftCheekBoxes
            rightCheekBoxes:(NSArray<NSArray<IntPoint *> *> *)rightCheekBoxes
            totalFramesNeedProcess:(NSInteger)totalFramesNeedProcess
            skipNFirstFrames:(NSInteger)skipNFirstFrames;
@property(weak, nonatomic) id<FaceMeshDelegate> delegate;
@property(nonatomic) size_t timestamp;
@end