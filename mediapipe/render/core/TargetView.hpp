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

#ifndef GPUIMAGE_X_TARGETVIEW_H
#define GPUIMAGE_X_TARGETVIEW_H

#include "Target.hpp"
#include "GLProgram.hpp"

NS_GI_BEGIN
class Context;
class TargetView : public Target {
public:
    enum FillMode {
        Stretch = 0,                    // Stretch to fill the view, and may distort the image
        PreserveAspectRatio = 1,        // preserve the aspect ratio of the image
        PreserveAspectRatioAndFill = 2  // preserve the aspect ratio, and zoom in to fill the view
    };

public:
    TargetView(Context *context);
    ~TargetView();

    void init();
    virtual void setInputFramebuffer(Framebuffer* framebuffer, RotationMode rotationMode = NoRotation,
                                     int texIdx = 0, bool ignoreForPrepared = false) override;
    void setFillMode(FillMode fillMode);
    void onSizeChanged(int width, int height);
    int getViewWidth(){return _viewWidth;};
    int getViewHeight(){return _viewHeight;};
    virtual void update(float frameTime) override;
    GLuint getProgram();
private:
    int _viewWidth;
    int _viewHeight;
    FillMode _fillMode;
    GLProgram* _displayProgram;
    GLuint _positionAttribLocation;
    GLuint _texCoordAttribLocation;
    GLuint _colorMapUniformLocation;
    
    struct {
        float r; float g; float b; float a;
    } _backgroundColor;

    GLfloat _displayVertices[8];

    void _updateDisplayVertices();
    const GLfloat* _getTexureCoordinate(RotationMode rotationMode);
    
    Context *_context;
};

NS_GI_END

#endif //GPUIMAGE_X_TARGETVIEW_H
