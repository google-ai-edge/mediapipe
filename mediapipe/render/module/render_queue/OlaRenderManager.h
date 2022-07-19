//
//  OlaRenderManager.h
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/17.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface OlaRenderManager : NSObject

+ (OlaRenderManager *)sharedInstance;

- (void)resume;

- (void)dispose;

- (int)render:(int64_t)frameTime textureId:(NSUInteger)inputTexture renderSize:(CGSize)size;

- (void)setRenderView:(UIView *)renderView;

+ (void) addImg;

+(void) disposeImg;

@end

NS_ASSUME_NONNULL_END
