/*
 * Opipe-x
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

#ifndef Opipe_X_SOURCECAMERA_H
#define Opipe_X_SOURCECAMERA_H

#include "Source.hpp"
#if defined(__APPLE__)
#include <AVFoundation/AVFoundation.h>
#endif

NS_GI_BEGIN
class Context;
class SourceCamera : public Source
{

public:
    enum SourceType
    {
        SourceType_RGBA = 0,
        SourceType_YUV420SP,
        SourceType_YUV420P
    };
    
public:
    SourceCamera(Context *context);
    virtual ~SourceCamera();
    
    static SourceCamera* create(Context *context);
    
    virtual void setRenderTexture(GLuint texture,
                                  int width,
                                  int height,
                                  Opipe::RotationMode outputRotation = Opipe::RotationMode::NoRotation,
                                  SourceType sourceType = SourceType_RGBA,
                                  const Opipe::TextureAttributes textureAttributes = Opipe::Framebuffer::defaultTextureAttribures);
#if defined(__APPLE__)
    virtual void setIORenderTexture(IOSurfaceID surfaceID,
                                    GLuint texture,
                                    int width,
                                    int height,
                                    Opipe::RotationMode outputRotation = Opipe::RotationMode::NoRotation,
                                    SourceType sourceType = SourceType_RGBA,
                                    const Opipe::TextureAttributes textureAttributes = Opipe::Framebuffer::defaultTextureAttribures);
#endif

    
    virtual void setFrameData(int width,
                              int height,
                              const void* pixels,
                              GLenum pixelsType,
                              GLuint texture,
                              RotationMode outputRotation = RotationMode::NoRotation,
                              SourceType sourceType = SourceType_RGBA,
                              const void* upixels = NULL,
                              const void* vpixels = NULL,
                              bool keep_white = false);
    
    virtual void updateTargets(float frameTime) override;
    
protected:
    Framebuffer *_UVFrameBuffer = 0;
    Framebuffer *_VFrameBuffer = 0;
    GLuint  _inputTexture = -1;
    bool _customTexture = false;
};
NS_GI_END

#endif //Opipe_X_SOURCECAMERA_H


