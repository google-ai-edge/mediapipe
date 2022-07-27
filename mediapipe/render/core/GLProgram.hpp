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

#ifndef Shader_hpp
#define Shader_hpp

#include "GPUImageMacros.h"
#include "string"
#if defined(__ANDROID__) || defined(ANDROID)
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#elif defined(__APPLE__)
#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>

#endif

#include "math_utils.hpp"
#include "math.hpp"
#include <vector>


NS_GI_BEGIN
class Context;
class GLProgram{
public:
    GLProgram(Context *context);
    ~GLProgram();
    
    static GLProgram* createByShaderString(Context *context, const std::string& vertexShaderSource, const std::string& fragmentShaderSource);
    void use();
    GLuint getID() const { return _program; }
    
    bool isValid() {
        if (glIsProgram(_program) != GL_TRUE ) {
            return false;
        } else {
            return true;
        }
    }

    GLuint getAttribLocation(const std::string& attribute);
    GLuint getUniformLocation(const std::string& uniformName);
    
    void setUniformValue(const std::string& uniformName, int value);
    void setUniformValue(const std::string& uniformName, int count, int* value, int valueSize = 1);
    void setUniformValue(const std::string& uniformName, float value);
    void setUniformValue(const std::string& uniformName, int count, float* value, int valueSize = 1);
    void setUniformValue(const std::string& uniformName, Vector2 value);
    void setUniformValue(const std::string& uniformName, Vector4 value);
    void setUniformValue(const std::string& uniformName, Matrix3 value);
    void setUniformValue(const std::string& uniformName, Opipe::Mat4 value);
    
    void setUniformValue(int uniformLocation, int value);
    void setUniformValue(int uniformLocation, int count, int* value, int valueSize = 1);
    void setUniformValue(int uniformLocation, float value);
    void setUniformValue(int uniformLocation, int count, float* value, int valueSize = 1);
    void setUniformValue(int uniformLocation, Vector2 value);
    void setUniformValue(int uniformLocation, Vector4 value);
    void setUniformValue(int uniformLocation, Matrix3 value);
    void setUniformValue(int uniformLocation, Opipe::Mat4 value);

    Context *getContext();
private:

    GLuint _program;
    bool _initWithShaderString(const std::string& vertexShaderSource, const std::string& fragmentShaderSource);
    Context *_context = 0;
};


NS_GI_END

#endif /* GLProgram_hpp */
