//
//  OlaRenderIMP.cpp
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/20.
//


#include "OlaRenderIMP.h"
#include "image_queue.h"

#if USE_OLARENDER
#include <OlaDispatch.hpp>
#endif

#if USE_OLARENDER
#else
const std::string TransformFragmentShaderString = SHADER_STRING
        (
                uniform
sampler2D colorMap;
varying highp
vec2 vTexCoord;

void main() {
    highp
    vec4 textureColor;
    highp
    vec2 uv = vTexCoord;
    textureColor = texture2D(colorMap, uv);

    gl_FragColor = vec4(textureColor.rgb, textureColor.a);
//    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

);

const std::string TransformVertexShaderString = SHADER_STRING
        (
                attribute
vec4 position;
attribute vec4
texCoord;

uniform mat4
mvp;
varying vec2
vTexCoord;

void main() {
    gl_Position = mvp * position;
    vTexCoord = texCoord.xy;
}

);

const std::string VertexShaderString = SHADER_STRING
        (
                attribute
vec4 position;
attribute vec4
texCoord;
attribute vec4
texCoord1;

varying vec2
vTexCoord;
varying vec2
vTexCoord1;

void main() {
    gl_Position = position;
    vTexCoord = texCoord.xy;
    vTexCoord1 = texCoord1.xy;
}

);

const std::string FragmentShaderString = SHADER_STRING
        (

                varying
highp vec2
vTexCoord;
varying highp
vec2 vTexCoord1;
uniform sampler2D
colorMap;
uniform sampler2D
colorMap1;

void main() {
    lowp
    vec4 textureColor = texture2D(colorMap, vTexCoord);
    lowp
    vec4 textureColor2 = texture2D(colorMap1, vTexCoord1);
    gl_FragColor = vec4(textureColor2 + textureColor * (1.0 - textureColor2.a));
}

);
#endif

NS_OLA_BEGIN

static GLfloat
positionCoords[] = {
-1.0f, -1.0f, 0.0f,
1.0f, -1.0f, 0.0f,
-1.0f, 1.0f, 0.0f,
1.0f, 1.0f, 0.0f,
};

static GLfloat textureCoords[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
};

static GLfloat textureCoords1[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f,
};

OLARenderIMP::OLARenderIMP() {

}

OLARenderIMP::~OLARenderIMP() {
//    imageData = nil;
    ImageQueue::getInstance()->dispose();
    _mvp_matrix.set_identity();
}

int OLARenderIMP::release() {
#if USE_OLARENDER
    if (_sobelFilter) {
        _sobelFilter->removeAllTargets();
        _sobelFilter->release();
        _sobelFilter = nullptr;
    }

    if (_brightFilter) {
        _brightFilter->removeAllTargets();
        _brightFilter->release();
        _brightFilter = nullptr;
    }


    if (_inputFramebuffer) {
        _inputFramebuffer->setExternalTexture(-1);
        _inputFramebuffer->release(false);
        _inputFramebuffer = nullptr;
    }
    if (_bridgeFilter) {
        _bridgeFilter->release();
        _bridgeFilter = nullptr;
    }

    _terminalFilter = nullptr;
    Filter::releaseVBOBuffers();
    Context::getInstance()->getFramebufferCache()->purge();
    Context::destroy();
#else
    if (_outputTexture > 0) {
        glDeleteTextures(1, &_outputTexture);
        _outputTexture = -1;
    }

    if (_outputFramebuffer > 0) {
        glDeleteFramebuffers(1, &_outputFramebuffer);
        _outputFramebuffer = -1;
    }
    if (_blendProgram > 0) {
        glDeleteProgram(_blendProgram);
        _blendProgram = -1;
    }

    if (_blendTexture > 0) {
        glDeleteTextures(1, &_blendTexture);
        _blendTexture = -1;
    }

    if (_blendFbo) {
        glDeleteFramebuffers(1, &_blendFbo);
        _blendFbo = -1;
    }

    if (_transformProgram) {
        glDeleteProgram(_transformProgram);
        _transformProgram = -1;
    }

    if (_transformTexture > 0) {
        glDeleteTextures(1, &_transformTexture);
        _transformTexture = -1;
    }
#endif
    _isInit = false;
    return 1;
}

int OLARenderIMP::loadGraph() {
    if (!_isInit) {
#if USE_OLARENDER
#if USE_MULTICONTEXT
        OlaDispatch::getSharedInstance()->runSync([&] {
#endif
            _brightFilter = BrightnessFilter::create();
            _bridgeFilter = OlaBridgeTextureFilter::create(-1, -1, -1);
            _sobelFilter = SobelEdgeDetectionFilter::create();
            _sobelFilter->addTarget(_brightFilter)->addTarget(_bridgeFilter);

            _terminalFilter = _bridgeFilter;

#if USE_MULTICONTEXT
        }, Context::ContextType::IOContext);
#endif
#else

#endif
        _isInit = true;

    }
    return 1;
}

TextureInfo OLARenderIMP::render(TextureInfo inputTexture, bool exportFlag) {
#if USE_NEED_RECREATE
    release();
#endif
    _renderWidth = inputTexture.width;
    _renderHeight = inputTexture.height;
    loadGraph();
    TextureInfo outputTexture;
    outputTexture.textureId = inputTexture.textureId;
#if USE_OLARENDER
#if USE_RESTORE_FBO
    GLint curFbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &curFbo);
#endif
#if USE_MULTICONTEXT
    GLsync sync;

    OlaDispatch::getSharedInstance()->runSync([&] {
#endif

        if (_terminalFilter) {
            if (_inputFramebuffer == nullptr || _inputFramebuffer->getTexture() != inputTexture.textureId) {
                if (_inputFramebuffer) {
                    _inputFramebuffer->release(false);
                    _inputFramebuffer = nullptr;
                }
                _inputFramebuffer = new Framebuffer(inputTexture.width, inputTexture.height,
                                                    (GLuint)inputTexture.textureId, false);
                _inputFramebuffer->setNoCacheFramebuffer();

            }

            _inputFramebuffer->setExternalTexture(inputTexture.textureId);
            if (_bridgeFilter) {
                //这是我们内部创建一个纹理id
#if USE_RENDER_TO_SRCTEXTURE
                _bridgeFilter->updateTargetTexture(inputTexture.textureId, inputTexture.width,
                                                   inputTexture.height, -1);
#else
                _bridgeFilter->updateTargetTexture(-1, inputTexture.width,
                                                   inputTexture.height, -1);
#endif
            }
            /**
                            下面这一段是测试代码验证引擎用
             */
            if (_brightFilter && _sobelFilter) {
                _sobelFilter->unPrepear();
                _sobelFilter->setInputFramebuffer(_inputFramebuffer);

                _brightFilter->setBrightness(0.5);
                if (_tempFactor > 1.0) {
                    _tempFactor = 0.0;
                } else {
                    _tempFactor += 1.0 / 120.0;
                }
                _sobelFilter->setProperty("edgeStrength", _tempFactor);
                _sobelFilter->update(inputTexture.frameTime); //测试渲染
            }

            auto *framebuffer = _terminalFilter->getFramebuffer();
            if (framebuffer) {
                outputTexture.textureId = framebuffer->getTexture();
                outputTexture.width = framebuffer->getWidth();
                outputTexture.height = framebuffer->getHeight();
                outputTexture.ioSurfaceId = framebuffer->getSurfaceID();
            }
        }
#if USE_MULTICONTEXT
        sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();
    }, Context::ContextType::IOContext);
    glWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
    glDeleteSync(sync);
#endif

#if USE_TEXImage2D
    IOSurfaceRef surface = IOSurfaceLookup(outputTexture.ioSurfaceId);
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, 0);
    void *pixels = IOSurfaceGetBaseAddress(surface);

    glBindTexture(GL_TEXTURE_2D, inputTexture.textureId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, outputTexture.width, outputTexture.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    outputTexture.textureId = inputTexture.textureId;
#endif

#if USE_RESTORE_FBO
    glBindFramebuffer(GL_FRAMEBUFFER, curFbo);
#endif
#else

    _loadProgram();
    _loadOutputTexture(inputTexture.width, inputTexture.height);

    ImageInfo rs;

    ImageQueue::getInstance()->pop(rs, exportFlag);
    Log("OlaRender", "aaa");
//    while (exportFlag && rs.len == 0);
    if (rs.len > 0) {
        int width = rs.width;
        int height = rs.height;

        setCanvasPixels(width, height, rs.data, inputTexture.frameTime,
                        Vec4(rs.startX, rs.startY, rs.normalWidth, rs.normalHeight));
        Log("OlaRender", "bbb");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glUseProgram(_blendProgram);
    glBindFramebuffer(GL_FRAMEBUFFER, _outputFramebuffer);
    glViewport(0, 0, inputTexture.width, inputTexture.height);
    glClearColor(0, 0, 0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTexture.textureId);
    glUniform1i(_inputTextureSlot, 0);
    glActiveTexture(GL_TEXTURE1);


    CHECK_GL(glBindTexture(GL_TEXTURE_2D, _blendTexture));
    CHECK_GL(glUniform1i(_inputTextureSlot1, 1));


    glVertexAttribPointer(_texCoordSlot, 2, GL_FLOAT, 0, 0, textureCoords);
    glEnableVertexAttribArray(_texCoordSlot);

    glVertexAttribPointer(_texCoordSlot1, 2, GL_FLOAT, 0, 0, textureCoords1);
    glEnableVertexAttribArray(_texCoordSlot1);

    glVertexAttribPointer(_positionSlot, 3, GL_FLOAT, 0, 0, positionCoords);
    glEnableVertexAttribArray(_positionSlot);

    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
    outputTexture.textureId = _outputTexture;
    if (rs.len > 0) {
        ImageQueue::getInstance()->releaseNode(rs);
    }
#endif

    return outputTexture;
}

void OLARenderIMP::setCanvasPixels(int width, int height, const void *pixels, int64_t frameTime,
                                   Vec4 roi) {
    if (_blendTexture != -1) {
        if (_lastTransformSize != Vec2(width, height)) {
            glDeleteTextures(1, &_transformTexture);
            _transformTexture = -1;
        }

        if (_transformTexture == -1) {
            glGenTextures(1, &_transformTexture);
            glBindTexture(GL_TEXTURE_2D, _transformTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels));
            glBindTexture(GL_TEXTURE_2D, 0);

        }

        if (_transformTexture != -1 && _blendFbo != -1) {
            // test
            _setROI(roi);

            glUseProgram(_transformProgram);
            glBindFramebuffer(GL_FRAMEBUFFER, _blendFbo);
            glViewport(0, 0, _renderWidth, _renderHeight);


            glClearColor(0, 0, 0, 0.0);
            glClear(GL_COLOR_BUFFER_BIT);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _transformTexture);
            Log("OLARender", "transformTexture :%d", _transformTexture);
            glUniform1i(_transformTextureSlot, 0);
            glUniformMatrix4fv(_transform_mvp, 1, GL_FALSE, (GLfloat * ) & _mvp_matrix);

            glVertexAttribPointer(_transformTexCoordSlot, 2, GL_FLOAT, 0, 0, textureCoords);
            glEnableVertexAttribArray(_transformTexCoordSlot);
            glVertexAttribPointer(_transformPositionSlot, 3, GL_FLOAT, 0, 0, positionCoords);
            glEnableVertexAttribArray(_transformPositionSlot);

            CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        }
    }
}

#if USE_OLARENDER
#if PLATFORM == PLATFORM_IOS
void OLARenderIMP::setDisplayView(OLARenderView *target) {

#else
void OLARenderIMP::setDisplayView(TargetView *target) {
#endif
    _targetView = target;
    if (_terminalFilter) {
        _terminalFilter->addTarget(target);
    }
}


void OLARenderIMP::removeRenderTarget() {
    _terminalFilter->removeAllTargets();
}

Source* OLARenderIMP::getTerminalSource() {
    return _terminalFilter;
}
#else

void OLARenderIMP::_loadProgram() {
    if (_blendProgram == -1) {
        GLuint vertexShader = _loadShader(GL_VERTEX_SHADER, VertexShaderString);
        GLuint fragmentShader = _loadShader(GL_FRAGMENT_SHADER, FragmentShaderString);

        _blendProgram = glCreateProgram();
        if (!_blendProgram) {
            assert(0);
            return;
        }

        glAttachShader(_blendProgram, vertexShader);
        glAttachShader(_blendProgram, fragmentShader);

        glLinkProgram(_blendProgram);

        // 检查错误
        GLint linked;
        glGetProgramiv(_blendProgram, GL_LINK_STATUS, &linked);
        if (!linked) {
            glDeleteProgram(_blendProgram);
            _blendProgram = 0;
            return;
        }

        glUseProgram(_blendProgram);

        _positionSlot = glGetAttribLocation(_blendProgram, "position");
        _texCoordSlot = glGetAttribLocation(_blendProgram, "texCoord");
        _texCoordSlot1 = glGetAttribLocation(_blendProgram, "texCoord1");
        _inputTextureSlot = glGetUniformLocation(_blendProgram, "colorMap");
        _inputTextureSlot1 = glGetUniformLocation(_blendProgram, "colorMap1");
    }

    if (_transformProgram == -1) {
        GLuint vertexShader = _loadShader(GL_VERTEX_SHADER, TransformVertexShaderString);
        GLuint fragmentShader = _loadShader(GL_FRAGMENT_SHADER, TransformFragmentShaderString);

        _transformProgram = glCreateProgram();
        if (!_transformProgram) {
            assert(0);
            return;
        }

        glAttachShader(_transformProgram, vertexShader);
        glAttachShader(_transformProgram, fragmentShader);

        glLinkProgram(_transformProgram);

        // 检查错误
        GLint linked;
        glGetProgramiv(_transformProgram, GL_LINK_STATUS, &linked);
        if (!linked) {
            glDeleteProgram(_transformProgram);
            _transformProgram = 0;
            return;
        }

        glUseProgram(_transformProgram);
        _transformPositionSlot = glGetAttribLocation(_transformProgram, "position");
        _transformTexCoordSlot = glGetAttribLocation(_transformProgram, "texCoord");
        _transformTextureSlot = glGetUniformLocation(_transformProgram, "colorMap");
        _transform_mvp = glGetUniformLocation(_transformProgram, "mvp");
    }
}

void OLARenderIMP::_loadOutputTexture(int width, int height) {
    if (_outputTexture == -1 || _outputFramebuffer == -1) {
        glGenFramebuffers(1, &_outputFramebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, _outputFramebuffer);

        glGenTextures(1, &_outputTexture);
        glBindTexture(GL_TEXTURE_2D, _outputTexture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _outputTexture, 0);
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        assert(status == GL_FRAMEBUFFER_COMPLETE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &_blendFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, _blendFbo);
        glGenTextures(1, &_blendTexture);
        glBindTexture(GL_TEXTURE_2D, _blendTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _blendTexture, 0);
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        assert(status == GL_FRAMEBUFFER_COMPLETE);

        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

    }
}

void OLARenderIMP::_setROI(Vec4 roi) {
    _mvp_matrix.set_identity();
    _roi.x = roi.x;
    _roi.y = roi.y;
    _roi.z = roi.z;
    _roi.w = roi.w;
    //transform OK
    float realXOffset = 1.0 / _roi.z * (1.0 - _roi.z);
    float realYOffset = 1.0 / _roi.w * (1.0 - _roi.w);

    auto roiAdjusted = Vec4(1.0 / _roi.z * _roi.x * 2.0,
                            1.0 / _roi.w * _roi.y * 2.0, 1.0, 1.0);
    _mvp_matrix.scale(_roi.z, _roi.w, 1.0);


// #if defined(__ANDROID__) || defined(ANDROID)
    _mvp_matrix.translate(-realXOffset, realYOffset, 0.0);
    _mvp_matrix.translate(roiAdjusted.x, -roiAdjusted.y, 0.0);
// #else
//     _mvp_matrix.translate(-realXOffset, -realYOffset, 0.0);
//     _mvp_matrix.translate(roiAdjusted.x, roiAdjusted.y, 0.0);
// #endif
}

GLuint OLARenderIMP::_loadShader(GLenum shaderType, const std::string &shaderString) {
    GLuint shader = glCreateShader(shaderType);
    if (shader == 0) {
        return 0;
    }

    // Load the shader source
    const char *shaderStringUTF8 = shaderString.c_str();
    glShaderSource(shader, 1, &shaderStringUTF8, NULL);

    // Compile the shader
    glCompileShader(shader);

    // Check the compile status
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled) {
        assert(0);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

#endif
NS_OLA_END
