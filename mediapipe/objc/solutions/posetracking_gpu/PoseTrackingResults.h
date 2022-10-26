#ifndef MEDIAPIPE_POSETRACKINGRESULTS_H
#define MEDIAPIPE_POSETRACKINGRESULTS_H

#import <Foundation/Foundation.h>

static const NSInteger POSE_NOSE = 0;
static const NSInteger POSE_LEFT_EYE_INNER = 1;
static const NSInteger POSE_LEFT_EYE = 2;
static const NSInteger POSE_LEFT_EYE_OUTER = 3;
static const NSInteger POSE_RIGHT_EYE_INNER = 4;
static const NSInteger POSE_RIGHT_EYE = 5;
static const NSInteger POSE_RIGHT_EYE_OUTER = 6;
static const NSInteger POSE_LEFT_EAR = 7;
static const NSInteger POSE_RIGHT_EAR = 8;
static const NSInteger POSE_MOUTH_LEFT = 9;
static const NSInteger POSE_MOUTH_RIGHT = 10;
static const NSInteger POSE_LEFT_SHOULDER = 11;
static const NSInteger POSE_RIGHT_SHOULDER = 12;
static const NSInteger POSE_LEFT_ELBOW = 13;
static const NSInteger POSE_RIGHT_ELBOW = 14;
static const NSInteger POSE_LEFT_WRIST = 15;
static const NSInteger POSE_RIGHT_WRIST = 16;
static const NSInteger POSE_LEFT_PINKY = 17;
static const NSInteger POSE_RIGHT_PINKY = 18;
static const NSInteger POSE_LEFT_INDEX = 19;
static const NSInteger POSE_RIGHT_INDEX = 20;
static const NSInteger POSE_LEFT_THUMB = 21;
static const NSInteger POSE_RIGHT_THUMB = 22;
static const NSInteger POSE_LEFT_HIP = 23;
static const NSInteger POSE_RIGHT_HIP = 24;
static const NSInteger POSE_LEFT_KNEE = 25;
static const NSInteger POSE_RIGHT_KNEE = 26;
static const NSInteger POSE_LEFT_ANKLE = 27;
static const NSInteger POSE_RIGHT_ANKLE = 28;
static const NSInteger POSE_LEFT_HEEL = 29;
static const NSInteger POSE_RIGHT_HEEL = 30;
static const NSInteger POSE_LEFT_FOOT = 31;
static const NSInteger POSE_RIGHT_FOOT = 32;




@interface PoseLandmark: NSObject

@property float x;
@property float y;
@property float z;
@property float presence;
@property float visibility;

- (instancetype) initWithX: (float) x y:(float) y z:(float) z presence:(float) presence visibility:(float) visibility;

@end

@interface PoseTrackingResults : NSObject


@property NSArray<PoseLandmark*>* landmarks;

- (instancetype) initWithLandmarks: (NSArray<PoseLandmark*>*) landmarks;

@end

#endif //MEDIAPIPE_POSETRACKINGRESULTS_H
