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

#include "TargetView.hpp"
#include "Context.hpp"
#include "GPUImageUtil.h"
#include "Filter.hpp"

USING_NS_GI

TargetView::TargetView(Context *context)
:_viewWidth(0)
,_viewHeight(0)
 ,_fillMode(FillMode::PreserveAspectRatioAndFill)
,_displayProgram(0)
,_positionAttribLocation(0)
,_texCoordAttribLocation(0)
,_colorMapUniformLocation(0)
,_context(context)
{
    _backgroundColor.r = 0.0;
    _backgroundColor.g = 0.0;
    _backgroundColor.b = 0.0;
    _backgroundColor.a = 0.0;
    init();
}

TargetView::~TargetView()
{
    if (_displayProgram) {
        delete _displayProgram;
        _displayProgram = 0;
    }
}

void TargetView::init() {
    _displayProgram = GLProgram::createByShaderString(_context, kDefaultVertexShader, kDefaultDisplayFragmentShader);
    _positionAttribLocation = _displayProgram->getAttribLocation("position");
    _texCoordAttribLocation = _displayProgram->getAttribLocation("texCoord");
    _colorMapUniformLocation = _displayProgram->getUniformLocation("colorMap");
    _context->setActiveShaderProgram(_displayProgram);
    CHECK_GL(glEnableVertexAttribArray(_positionAttribLocation));
    CHECK_GL(glEnableVertexAttribArray(_texCoordAttribLocation));

};

void TargetView::setInputFramebuffer(Framebuffer* framebuffer,
                                     RotationMode rotationMode/* = NoRotation*/,
                                     int texIdx/* = 0*/, bool ignoreForPrepared)
{
    Framebuffer* lastInputFramebuffer = 0;
    RotationMode lastInputRotation = NoRotation;
    if (_inputFramebuffers.find(0) != _inputFramebuffers.end()) {
        lastInputFramebuffer = _inputFramebuffers[0].frameBuffer;
        lastInputRotation = _inputFramebuffers[0].rotationMode;
    }

    Target::setInputFramebuffer(framebuffer, rotationMode, texIdx, ignoreForPrepared);

    if (lastInputFramebuffer != framebuffer && framebuffer &&
            (!lastInputFramebuffer ||
            !(lastInputFramebuffer->getWidth() == framebuffer->getWidth() &&
              lastInputFramebuffer->getHeight() == framebuffer->getHeight() &&
              lastInputRotation == rotationMode)
            )) {
        _updateDisplayVertices();
    }
}

void TargetView::setFillMode(FillMode fillMode) {
    if (_fillMode != fillMode) {
        _fillMode = fillMode;
        _updateDisplayVertices();
    }
}

void TargetView::onSizeChanged(int width, int height) {
    if (_viewWidth != width || _viewHeight != height) {
        _viewWidth = width;
        _viewHeight = height;
        _updateDisplayVertices();
    }
}

void TargetView::update(float frameTime)
{
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    CHECK_GL(glViewport(0, 0, _viewWidth, _viewHeight));
    CHECK_GL(glClearColor(_backgroundColor.r, _backgroundColor.g, _backgroundColor.b, _backgroundColor.a));
    CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    _context->setActiveShaderProgram(_displayProgram);
    CHECK_GL(glActiveTexture(GL_TEXTURE0));
    CHECK_GL(glBindTexture(GL_TEXTURE_2D, _inputFramebuffers[0].frameBuffer->getTexture()));
    CHECK_GL(glUniform1i(_colorMapUniformLocation, 0));
    CHECK_GL(glVertexAttribPointer(_positionAttribLocation, 2, GL_FLOAT, 0, 0, _displayVertices));
    CHECK_GL(glVertexAttribPointer(_texCoordAttribLocation, 2, GL_FLOAT, 0, 0, _getTexureCoordinate(_inputFramebuffers[0].rotationMode)));
    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

void TargetView::_updateDisplayVertices()
{
    if (_inputFramebuffers.find(0) == _inputFramebuffers.end() || _inputFramebuffers[0].frameBuffer == 0) return;

    Framebuffer* inputFramebuffer = _inputFramebuffers[0].frameBuffer;
    RotationMode inputRotation = _inputFramebuffers[0].rotationMode;

    int rotatedFramebufferWidth = inputFramebuffer->getWidth();
    int rotatedFramebufferHeight = inputFramebuffer->getHeight();
    if (rotationSwapsSize(inputRotation))
    {
        rotatedFramebufferWidth = inputFramebuffer->getHeight();
        rotatedFramebufferHeight = inputFramebuffer->getWidth();
    }

    float framebufferAspectRatio = rotatedFramebufferHeight / (float)rotatedFramebufferWidth;
    float viewAspectRatio = _viewHeight / (float)_viewWidth;

    float insetFramebufferWidth = 0.0;
    float insetFramebufferHeight = 0.0;
    if (framebufferAspectRatio > viewAspectRatio) {
        insetFramebufferWidth = _viewHeight / (float)rotatedFramebufferHeight * rotatedFramebufferWidth;
        insetFramebufferHeight = _viewHeight;
    } else {
        insetFramebufferWidth = _viewWidth;
        insetFramebufferHeight = _viewWidth / (float)rotatedFramebufferWidth * rotatedFramebufferHeight;
    }

    float scaledWidth = 1.0;
    float scaledHeight = 1.0;
    if (_fillMode == FillMode::PreserveAspectRatio) {
        scaledWidth = insetFramebufferWidth / _viewWidth;
        scaledHeight = insetFramebufferHeight / _viewHeight;
    } else if (_fillMode == FillMode::PreserveAspectRatioAndFill) {
        scaledWidth = _viewHeight/ insetFramebufferHeight;
        scaledHeight = _viewWidth  / insetFramebufferWidth;
    }

    _displayVertices[0] = -scaledWidth;
    _displayVertices[1] = -scaledHeight;
    _displayVertices[2] = scaledWidth;
    _displayVertices[3] = -scaledHeight;
    _displayVertices[4] = -scaledWidth;
    _displayVertices[5] = scaledHeight;
    _displayVertices[6] = scaledWidth;
    _displayVertices[7] = scaledHeight;

}

const GLfloat* TargetView::_getTexureCoordinate(RotationMode rotationMode)
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

    switch(rotationMode)
    {
    case NoRotation: return noRotationTextureCoordinates;
    case RotateLeft: return rotateLeftTextureCoordinates;
    case RotateRight: return rotateRightTextureCoordinates;
    case FlipVertical: return verticalFlipTextureCoordinates;
    case FlipHorizontal: return horizontalFlipTextureCoordinates;
    case RotateRightFlipVertical: return rotateRightVerticalFlipTextureCoordinates;
    case RotateRightFlipHorizontal: return rotateRightHorizontalFlipTextureCoordinates;
    case Rotate180: return rotate180TextureCoordinates;
    }
}
GLuint TargetView::getProgram()
{
    return _displayProgram->getID();
}
