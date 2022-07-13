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

#ifndef GPUIMAGE_X_SOURCEIMAGE_H
#define GPUIMAGE_X_SOURCEIMAGE_H

#include "Source.hpp"

NS_GI_BEGIN
class Context;
class SourceImage : public Source {
public:
    SourceImage(Context *context);
    ~SourceImage();

    static SourceImage* create(Context *context, int width, int height, const void* pixels);
    static SourceImage* create(Context *context, int width, int height, GLuint textureId);
    static SourceImage* create(Context *context, int width, int height, GLuint textureId, RotationMode rotationMode);
    
    SourceImage* setImage(int width, int height, GLuint textureId);
    SourceImage* setImage(int width, int height, GLuint textureId, RotationMode rotationMode);
    SourceImage* setImage(int width, int height, const void* pixels);

#if defined(__APPLE__)
    static SourceImage* create(Context *context, int width, int height, const void* pixels, int extraWidth);
    SourceImage* setImage(int width, int height, const void* pixels, int extraWidth);
    
    static SourceImage* create(Context *context, NSURL* imageUrl);
    SourceImage* setImage(NSURL* imageUrl);
    
    static SourceImage* create(Context *context, NSData* imageData);
    SourceImage* setImage(NSData* imageData);
    
    static SourceImage* create(Context *context, UIImage* image);
    SourceImage* setImage(UIImage* image);
    
    static SourceImage* create(Context *context, CGImageRef image);
    SourceImage* setImage(CGImageRef image);

private:
    UIImage* _adjustImageOrientation(UIImage* image);
#endif
private:
    bool _customTexture = false;
};

NS_GI_END

#endif //GPUIMAGE_X_SOURCEIMAGE_H
