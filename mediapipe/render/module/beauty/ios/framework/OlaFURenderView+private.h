//
//  OlaFURenderView+private.h
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/20.
//

#ifndef OlaFURenderView_private_h
#define OlaFURenderView_private_h
#import "OlaFURenderView.h"
#import "mediapipe/render/core/GPUImageTarget.h"
#import "mediapipe/render/core/Context.hpp"
#include "mediapipe/render/core/TargetView.hpp"

@interface OlaFURenderView()  <GPUImageTarget>
{
    Opipe::RotationMode inputRotation;
    GLuint displayFramebuffer;
    GLuint displayRenderbuffer;
    Opipe::GLProgram* displayProgram;
    
    GLfloat displayVertices[8];
    GLint framebufferWidth, framebufferHeight;
    CGSize lastBoundsSize;
    Opipe::Context *_context;
    CGRect renderBounds;
    GLfloat backgroundColorRed, backgroundColorGreen, backgroundColorBlue, backgroundColorAlpha;
}
@property(readwrite, nonatomic) Opipe::TargetView::FillMode fillMode;
@property(nonatomic) Opipe::Framebuffer* inputFramebuffer;
@property(nonatomic) GLuint positionAttribLocation;
@property(nonatomic) GLuint texCoordAttribLocation;
@property(nonatomic) GLuint colorMapUniformLocation;
- (void)presentFramebuffer;
@end

#endif /* OlaFURenderView_private_h */
