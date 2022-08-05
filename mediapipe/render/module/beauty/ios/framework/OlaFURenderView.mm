//
//  OlaFURenderView.m
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/20.
//

#import "OlaFURenderView+private.h"

#import <UIKit/UIKit.h>
#include "mediapipe/render/core/GLProgram.hpp"
#include "mediapipe/render/core/Filter.hpp"
#import <AVFoundation/AVFoundation.h>
NS_ASSUME_NONNULL_BEGIN


@implementation OlaFURenderView

+ (Class)layerClass
{
    return [CAEAGLLayer class];
}

- (id)initWithFrame:(CGRect)frame context:(void *)context
{
    if (!(self = [super initWithFrame:frame]))
    {
        return nil;
    }
    _context = (Opipe::Context *)context;
    _context->useAsCurrent();
    [self commonInit];
    renderBounds = self.bounds;
    return self;
}

- (void)commonInit;
{
    inputRotation = Opipe::NoRotation;
    self.opaque = YES;
    self.hidden = NO;
    CAEAGLLayer* eaglLayer = (CAEAGLLayer*)self.layer;
    eaglLayer.opaque = YES;
    eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:NO],
                                    kEAGLDrawablePropertyRetainedBacking,
                                    kEAGLColorFormatRGBA8,
                                    kEAGLDrawablePropertyColorFormat, nil];
     displayProgram = Opipe::GLProgram::createByShaderString(_context, Opipe::kDefaultVertexShader,
                                                             Opipe::kDefaultDisplayFragmentShader);
    _positionAttribLocation = displayProgram->getAttribLocation("position");
    _texCoordAttribLocation = displayProgram->getAttribLocation("texCoord");
    _colorMapUniformLocation = displayProgram->getUniformLocation("colorMap");
    _context->setActiveShaderProgram(displayProgram);
    glEnableVertexAttribArray(_positionAttribLocation);
    glEnableVertexAttribArray(_texCoordAttribLocation);
        
    [self setBackgroundColorRed:0.0 green:0.0 blue:0.0 alpha:0.0];
    _fillMode = Opipe::TargetView::FillMode::PreserveAspectRatioAndFill;
    [self createDisplayFramebuffer];
        
    
}

- (void)layoutSubviews {
    [super layoutSubviews];
    renderBounds = self.bounds;
    if (!CGSizeEqualToSize(self.bounds.size, lastBoundsSize) &&
        !CGSizeEqualToSize(self.bounds.size, CGSizeZero)) {
        _context->useAsCurrent();
        [self destroyDisplayFramebuffer];
        [self createDisplayFramebuffer];
    }
    
}

- (void)dealloc
{
    [self destroyDisplayFramebuffer];
}

- (void)createDisplayFramebuffer;
{
    CAEAGLLayer *layer = self.layer;
    CGSize bounds = self.bounds.size;
    glGenRenderbuffers(1, &displayRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, displayRenderbuffer);
    
    [_context->getEglContext() renderbufferStorage:GL_RENDERBUFFER fromDrawable:layer];
        
    glGenFramebuffers(1, &displayFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, displayFramebuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, displayRenderbuffer);
        
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &framebufferWidth);
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &framebufferHeight);
        
    lastBoundsSize = bounds;
    [self updateDisplayVertices];
}

- (void)destroyDisplayFramebuffer;
{
    if (displayFramebuffer)
    {
        glDeleteFramebuffers(1, &displayFramebuffer);
        displayFramebuffer = 0;
    }
        
    if (displayRenderbuffer)
    {
        glDeleteRenderbuffers(1, &displayRenderbuffer);
        displayRenderbuffer = 0;
    }
}

- (void)setDisplayFramebuffer;
{
    if (!displayFramebuffer)
    {
        [self createDisplayFramebuffer];
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, displayFramebuffer);
    glViewport(0, 0, framebufferWidth, framebufferHeight);
}

- (void)presentFramebuffer;
{
    glBindRenderbuffer(GL_RENDERBUFFER, displayRenderbuffer);
    _context->presentBufferForDisplay();
}

- (void)setBackgroundColorRed:(GLfloat)redComponent green:(GLfloat)greenComponent blue:(GLfloat)blueComponent alpha:(GLfloat)alphaComponent;
{
    backgroundColorRed = redComponent;
    backgroundColorGreen = greenComponent;
    backgroundColorBlue = blueComponent;
    backgroundColorAlpha = alphaComponent;
}

- (void)update:(float)frameTime {
    
    _context->setActiveShaderProgram(displayProgram);

    [self setDisplayFramebuffer];
    glClearColor(backgroundColorRed, backgroundColorGreen, backgroundColorBlue, backgroundColorAlpha);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _inputFramebuffer->getTexture());
    glUniform1i(_colorMapUniformLocation, 0);
    glVertexAttribPointer(_positionAttribLocation, 2, GL_FLOAT, 0, 0, displayVertices);
    glVertexAttribPointer(_texCoordAttribLocation, 2, GL_FLOAT, 0, 0, [self textureCoordinatesForRotation:inputRotation] );
        
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        
    [self presentFramebuffer];
}

- (void)setInputFramebuffer:(Opipe::Framebuffer*)newInputFramebuffer withRotation:(Opipe::RotationMode)rotation atIndex:(NSInteger)texIdx {
    Opipe::Framebuffer* lastFramebuffer = _inputFramebuffer;
    Opipe::RotationMode lastInputRotation = inputRotation;
    
    inputRotation = rotation;
    _inputFramebuffer = newInputFramebuffer;
    
    if (lastFramebuffer != newInputFramebuffer && newInputFramebuffer &&
        ( !lastFramebuffer ||
          !(lastFramebuffer->getWidth() == newInputFramebuffer->getWidth() &&
            lastFramebuffer->getHeight() == newInputFramebuffer->getHeight() &&
            lastInputRotation == rotation)
        )) {
        [self updateDisplayVertices];
    }
}

- (void)setFillMode:(Opipe::TargetView::FillMode)newValue;
{
    if (_fillMode != newValue) {
        _fillMode = newValue;
        [self updateDisplayVertices];
    }
}

- (void)updateDisplayVertices;
{
    if (_inputFramebuffer == 0) return;
    
    CGFloat scaledWidth = 1.0;
    CGFloat scaledHeight = 1.0;

    int rotatedFramebufferWidth = _inputFramebuffer->getWidth();
    int rotatedFramebufferHeight = _inputFramebuffer->getHeight();
    if (rotationSwapsSize(inputRotation))
    {
        rotatedFramebufferWidth = _inputFramebuffer->getHeight();
        rotatedFramebufferHeight = _inputFramebuffer->getWidth();
    }
    
    CGRect insetRect = AVMakeRectWithAspectRatioInsideRect(CGSizeMake(rotatedFramebufferWidth,
                                                                      rotatedFramebufferHeight),
                                                           renderBounds);
    
    if (_fillMode == Opipe::TargetView::FillMode::PreserveAspectRatio) {
        scaledWidth = insetRect.size.width / self.bounds.size.width;
        scaledHeight = insetRect.size.height / self.bounds.size.height;
    } else if (_fillMode == Opipe::TargetView::FillMode::PreserveAspectRatioAndFill) {
        scaledWidth = renderBounds.size.height / insetRect.size.height;
        scaledHeight = renderBounds.size.width / insetRect.size.width;
    }
    
    displayVertices[0] = -scaledWidth;
    displayVertices[1] = -scaledHeight;
    displayVertices[2] = scaledWidth;
    displayVertices[3] = -scaledHeight;
    displayVertices[4] = -scaledWidth;
    displayVertices[5] = scaledHeight;
    displayVertices[6] = scaledWidth;
    displayVertices[7] = scaledHeight;
}


- (const GLfloat *)textureCoordinatesForRotation:(Opipe::RotationMode)rotationMode;
{
    static const GLfloat noRotationTextureCoordinates[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f,
    };
    
    static const GLfloat rotateRightTextureCoordinates[] = {
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
    };
    
    static const GLfloat rotateLeftTextureCoordinates[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
    };
    
    static const GLfloat verticalFlipTextureCoordinates[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    
    static const GLfloat horizontalFlipTextureCoordinates[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    
    static const GLfloat rotateRightVerticalFlipTextureCoordinates[] = {
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        0.0f, 1.0f,
    };
    
    static const GLfloat rotateRightHorizontalFlipTextureCoordinates[] = {
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
    };
    
    static const GLfloat rotate180TextureCoordinates[] = {
        1.0f, 0.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
    };
    
    switch(inputRotation)
    {
        case Opipe::NoRotation: return noRotationTextureCoordinates;
        case Opipe::RotateLeft: return rotateLeftTextureCoordinates;
        case Opipe::RotateRight: return rotateRightTextureCoordinates;
        case Opipe::FlipVertical: return verticalFlipTextureCoordinates;
        case Opipe::FlipHorizontal: return horizontalFlipTextureCoordinates;
        case Opipe::RotateRightFlipVertical: return rotateRightVerticalFlipTextureCoordinates;
        case Opipe::RotateRightFlipHorizontal: return rotateRightHorizontalFlipTextureCoordinates;
        case Opipe::Rotate180: return rotate180TextureCoordinates;
    }
}

@end

NS_ASSUME_NONNULL_END

