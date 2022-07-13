/*
 * GPUImage-x
 *
 * Copyright (C) 2017 Yijin Wang, Yiqian Wang
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SourceImage.hpp"
#include "Context.hpp"
#include "GPUImageUtil.h"

#if defined(__APPLE__)
#include "CVFramebuffer.hpp"
#endif

USING_NS_GI

SourceImage::~SourceImage() {
    if (_customTexture) {
        delete _framebuffer;
    }
    _framebuffer = 0;
    removeAllTargets();
}

SourceImage::SourceImage(Context *context) : Source(context) {
    
}

SourceImage* SourceImage::create(Context *context, int width, int height, const void* pixels) {
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(width, height, pixels);
    return sourceImage;
}

SourceImage* SourceImage::create(Context *context, int width, int height, GLuint textureId)
{
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(width, height, textureId);
    return sourceImage;
}

SourceImage* SourceImage::create(Context *context, int width, int height, GLuint textureId, RotationMode rotationMode)
{
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(width, height, textureId, rotationMode);
    return sourceImage;
}

SourceImage* SourceImage::setImage(int width, int height, GLuint textureId, RotationMode rotationMode)
{
    this->setFramebuffer(0);
    _customTexture = true;
    Framebuffer *framebuffer = getContext()->getFramebufferCache()->fetchFramebufferUseTextureId(_context, width, height, textureId);
    this->setFramebuffer(framebuffer, rotationMode);
    framebuffer->lock("SourceImage");
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    
    return this;
}

SourceImage* SourceImage::setImage(int width, int height, GLuint textureId)
{
    this->setFramebuffer(0);
    _customTexture = true;
    Framebuffer *framebuffer = getContext()->getFramebufferCache()->fetchFramebufferUseTextureId(_context, width, height, textureId);
    this->setFramebuffer(framebuffer);
    framebuffer->lock("SourceImage");
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    
    return this;
}

SourceImage* SourceImage::setImage(int width, int height, const void* pixels) {
    this->setFramebuffer(0);
    Framebuffer* framebuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, width, height, false);
    this->setFramebuffer(framebuffer);
    framebuffer->lock("SourceImage");
    
    if (pixels) {
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));
        CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    }
    
    return this;
}

#if defined(__APPLE__)

static size_t const kQuarameraDynamicTextureByteAlignment = 16;

NS_INLINE size_t QAAlignSize(size_t size)
{
    return ceil(size / (double)kQuarameraDynamicTextureByteAlignment) * kQuarameraDynamicTextureByteAlignment;
}

SourceImage* SourceImage::create(Context *context, int width, int height, const void *pixels, int extraWidth)
{
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(width, height, pixels, extraWidth);
    return sourceImage;
}

SourceImage* SourceImage::setImage(int width, int height, const void *pixels, int extraWidth)
{
    this->setFramebuffer(0);
    Framebuffer* framebuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, width, height, true);
    this->setFramebuffer(framebuffer);
    framebuffer->lock("SourceImage");
    
    size_t alignWidth = QAAlignSize(width);
    if (pixels) {
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, this->getFramebuffer()->getTexture()));
        CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)alignWidth, height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, pixels));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
    }
    return this;
}

SourceImage* SourceImage::create(Context *context, NSURL* imageUrl) {
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(imageUrl);
    return sourceImage;
}

SourceImage* SourceImage::setImage(NSURL* imageUrl) {
    NSData *imageData = [[NSData alloc] initWithContentsOfURL:imageUrl];
    setImage(imageData);
    return this;
}

SourceImage* SourceImage::create(Context *context, NSData* imageData) {
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(imageData);
    return sourceImage;
}

SourceImage* SourceImage::setImage(NSData* imageData) {
    UIImage* inputImage = [[UIImage alloc] initWithData:imageData];
    setImage(inputImage);
    return this;
}

SourceImage* SourceImage::create(Context *context, UIImage* image) {
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(image);
    return sourceImage;
}

SourceImage* SourceImage::setImage(UIImage* image) {
    UIImage* img = _adjustImageOrientation(image);
    setImage([img CGImage]);
    return this;
}

SourceImage* SourceImage::create(Context *context, CGImageRef image) {
    SourceImage* sourceImage = new SourceImage(context);
    sourceImage->setImage(image);
    return sourceImage;
}

SourceImage* SourceImage::setImage(CGImageRef image) {
    GLubyte *imageData = NULL;
    CFDataRef dataFromImageDataProvider = CGDataProviderCopyData(CGImageGetDataProvider(image));
    imageData = (GLubyte *)CFDataGetBytePtr(dataFromImageDataProvider);
    int width = (int)CGImageGetWidth(image);
    int height = (int)CGImageGetHeight(image);
    assert((width > 0 && height > 0) && "image can not be empty");

    this->setFramebuffer(0);
    
    CVFramebuffer *framebuffer = (CVFramebuffer *)getContext()->
    getFramebufferCache()->
    fetchFramebuffer(_context,
                     width,
                     height,
                     false,
                     Framebuffer::defaultTextureAttribures,
                     true);
    this->setFramebuffer(framebuffer);
    framebuffer->lock("SourceImage");
    
    CIImage *inputImage = [CIImage imageWithCGImage:image];
    CIContext *context = [CIContext contextWithCGContext:UIGraphicsGetCurrentContext() options:nil];
    CVPixelBufferRef pixelBuffer = framebuffer->renderTarget;
    [context render:inputImage toCVPixelBuffer:pixelBuffer];
    
    CFRelease(dataFromImageDataProvider);
    context = nil;
    inputImage = nil;
    
    return this;

}

UIImage* SourceImage::_adjustImageOrientation(UIImage* image)
{
    if (image.imageOrientation == UIImageOrientationUp)
        return image;
    
    CGAffineTransform transform = CGAffineTransformIdentity;
    switch (image.imageOrientation) {
        case UIImageOrientationDown:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.width, image.size.height);
            transform = CGAffineTransformRotate(transform, M_PI);
            break;
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.width, 0);
            transform = CGAffineTransformRotate(transform, M_PI_2);
            break;
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, 0, image.size.height);
            transform = CGAffineTransformRotate(transform, -M_PI_2);
            break;
        default:
            break;
    }
    
    switch (image.imageOrientation) {
        case UIImageOrientationUpMirrored:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.width, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.height, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
        default:
            break;
    }
    
    CGContextRef ctx = CGBitmapContextCreate(NULL, image.size.width, image.size.height,
                                             CGImageGetBitsPerComponent(image.CGImage), 0,
                                             CGImageGetColorSpace(image.CGImage),
                                             CGImageGetBitmapInfo(image.CGImage));
    CGContextConcatCTM(ctx, transform);
    switch (image.imageOrientation) {
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            CGContextDrawImage(ctx, CGRectMake(0,0,image.size.height,image.size.width), image.CGImage);
            break;
        default:
            CGContextDrawImage(ctx, CGRectMake(0,0,image.size.width,image.size.height), image.CGImage);
            break;
    }
    
    CGImageRef cgImage = CGBitmapContextCreateImage(ctx);
    UIImage* newImage = [UIImage imageWithCGImage:cgImage];
    CGContextRelease(ctx);
    CGImageRelease(cgImage);
    return newImage;
}

#endif
