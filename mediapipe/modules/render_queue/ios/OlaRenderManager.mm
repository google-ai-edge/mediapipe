//
//  OlaRenderManager.m
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/17.
//

#import "OlaRenderManager.h"

#include "mediapipe/modules/render_queue/OlaRender.hpp"
#include "mediapipe/modules/render_queue/image_queue.h"

USING_NS_OLA

@interface OlaRenderManager() {
    OlaRender *olaRender;
#if USE_OLARENDER
    OLARenderView *targetView;
#endif
    EAGLContext *currentContext;
}

@end

@implementation OlaRenderManager

+ (OlaRenderManager *)sharedInstance {
    static dispatch_once_t onceToken;
    static OlaRenderManager *instance = nil;
    dispatch_once(&onceToken, ^{
        instance = [OlaRenderManager new];
    });
    return instance;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
       
    }
    return self;
}

- (void)dispose
{
    if (olaRender) {
        olaRender->release();
        olaRender = nullptr;
    }
    
}

- (int)render:(int64_t)frameTime textureId:(NSUInteger)inputTexture renderSize:(CGSize)size
{
    [self resume];
    TextureInfo info;

    info.width = size.width;
    info.height = size.height;
    info.frameTime = frameTime;
    info.textureId = (int)inputTexture;
    TextureInfo rs = olaRender->render(info, false);
    return rs.textureId;
}

- (void)resume
{
    if (olaRender == nullptr) {
        olaRender = OlaRender::create();
    }
   
}

- (void)setRenderView:(UIView *)renderView
{
#if USE_OLARENDER
    targetView = (OLARenderView *)renderView;
    olaRender->setDisplayView(targetView);
#endif
}

@end
