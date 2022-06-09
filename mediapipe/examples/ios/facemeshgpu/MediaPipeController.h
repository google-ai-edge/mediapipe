#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@class MediaPipeController;
@class MediaPipeFaceLandmarkPoint;
@class MediaPipeNormalizedRect;

typedef void (^MediaPipeCompletionBlock)(CVPixelBufferRef pixelBuffer);

@protocol MediaPipeControllerDelegate <NSObject>
@optional
- (void)mediaPipeController:(MediaPipeController *)controller didReceiveFaces:(NSArray<NSArray<MediaPipeFaceLandmarkPoint *>*>*)faces;
- (void)mediaPipeController:(MediaPipeController *)controller didReceiveFaceBoxes:(NSArray<MediaPipeNormalizedRect *>*)faces;
- (void)mediaPipeController:(MediaPipeController *)controller didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer;
@end

@interface MediaPipeController : NSObject

+ (instancetype)facemesh;
+ (instancetype)effects;

- (void)startGraph;
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer timestamp:(CMTime)timestamp completion:(nullable MediaPipeCompletionBlock)completion;
@property (nullable, weak, nonatomic) id<MediaPipeControllerDelegate> delegate;
@end

@interface MediaPipeFaceLandmarkPoint : NSObject
@property (nonatomic) float x;
@property (nonatomic) float y;
@property (nonatomic) float z;
@end

@interface MediaPipeNormalizedRect : NSObject
@property (nonatomic) float centerX;
@property (nonatomic) float centerY;
@property (nonatomic) float height;
@property (nonatomic) float width;
@property (nonatomic) float rotation;
@end

NS_ASSUME_NONNULL_END
