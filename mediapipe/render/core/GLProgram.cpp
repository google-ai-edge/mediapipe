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

#include <algorithm>
#include "GLProgram.hpp"
#include "Context.hpp"
#include "GPUImageUtil.h"

namespace Opipe {
    
    //std::vector<GLProgram*> GLProgram::_context->_programs;
    
    GLProgram::GLProgram(Context *context) : _program(-1), _context(context) {
        _context->_programs.push_back(this);
    }
    
    GLProgram::~GLProgram() {
        if (nullptr != _context && _context->_programs.size() > 0) { //context 可能为空
            std::vector<GLProgram *>::iterator itr = std::find(_context->_programs.begin(), _context->_programs.end(), this);
            if (itr != _context->_programs.end()) {
                _context->_programs.erase(itr);
            }

            bool bDeleteProgram = (_program != -1);

            for (auto const &program : _context->_programs) {
                if (bDeleteProgram) {
                    if (_program == program->getID()) {
                        bDeleteProgram = false;
                        break;
                    }
                }
            }

            if (bDeleteProgram) {
                glDeleteProgram(_program);
                _program = -1;
            }
        } else {
            glDeleteProgram(_program);
            _program = -1;
        }
    }
    
    GLProgram *GLProgram::createByShaderString(Context *context, const std::string &vertexShaderSource, const std::string &fragmentShaderSource) {
        GLProgram *ret = new(std::nothrow) GLProgram(context);
        if (ret) {
            if (!ret->_initWithShaderString(vertexShaderSource, fragmentShaderSource)) {
                delete ret;
                ret = 0;
            }
        }
        return ret;
    }
    
    GLuint loadShader(GLenum shaderType, const char *pSource) {
        GLuint shader = 0;
        shader = CHECK_GL(glCreateShader(shaderType));
        if (shader) {
            CHECK_GL(glShaderSource(shader, 1, &pSource, nullptr));
            CHECK_GL(glCompileShader(shader));
            GLint compiled = 0;
            CHECK_GL(glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled));
            if (!compiled) {
                GLint infoLen = 0;
                CHECK_GL(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen));
                if (infoLen) {
                    char *buf = (char *) malloc((size_t) infoLen);
                    if (buf) {
                        CHECK_GL(glGetShaderInfoLog(shader, infoLen, nullptr, buf));
                        std::string shaderTypeStr("unknown");
                        if (shaderType == GL_FRAGMENT_SHADER) {
                            shaderTypeStr = std::string("GL_FRAGMENT_SHADER");
                        } else if (shaderType == GL_VERTEX_SHADER) {
                            shaderTypeStr = std::string("GL_VERTEX_SHADER");
                        }
                        Opipe::LogE("GPUImage-x", "LoadShader Could not compile shader type : %s \n because of %s", shaderTypeStr.c_str(), buf);
                        Opipe::Log("GPUImage-x", "\n%s\n", pSource);
                        free(buf);
                    }
                    CHECK_GL(glDeleteShader(shader));
                    shader = 0;
                }
            }
        }
        return shader;
    }
    
    
    bool GLProgram::_initWithShaderString(const std::string &vertexShaderSource, const std::string &fragmentShaderSource) {
        
        if (_program != -1) {
            CHECK_GL(glDeleteProgram(_program));
            _program = -1;
        }
        CHECK_GL(_program = glCreateProgram());
        
        GLuint vertShader = loadShader(GL_VERTEX_SHADER, vertexShaderSource.c_str());
        
        GLuint fragShader = loadShader(GL_FRAGMENT_SHADER, fragmentShaderSource.c_str());
        
        CHECK_GL(glAttachShader(_program, vertShader));
        CHECK_GL(glAttachShader(_program, fragShader));
        
        CHECK_GL(glLinkProgram(_program));
        
        GLint linkStatus = GL_FALSE;
        CHECK_GL(glGetProgramiv(_program, GL_LINK_STATUS, &linkStatus));
        
        CHECK_GL(glDeleteShader(vertShader));
        CHECK_GL(glDeleteShader(fragShader));
        
        if (!linkStatus) {
            GLint bufLength = 0;
            CHECK_GL(glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &bufLength));
            if (bufLength) {
                char *buf = (char *) malloc((size_t) bufLength);
                if (buf) {
                    CHECK_GL(glGetProgramInfoLog(_program, bufLength, NULL, buf));
                    Opipe::LogE("GPUImage-x", "compile gl program error %s", buf);
                    free(buf);
                }
            }
            CHECK_GL(glDeleteProgram(_program));
            _program = 0;
            return false;
        }
        
        return true;
    }
    
    
    void GLProgram::use() {
        CHECK_GL(glUseProgram(_program));
    }
    
    GLuint GLProgram::getAttribLocation(const std::string &attribute) {
        return glGetAttribLocation(_program, attribute.c_str());
    }
    
    GLuint GLProgram::getUniformLocation(const std::string &uniformName) {
        return glGetUniformLocation(_program, uniformName.c_str());
    }
    
    
    void GLProgram::setUniformValue(const std::string &uniformName, int value) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, value);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, int count, int *value, int valueSize/*=1*/) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, count, value, valueSize);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, float value) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, value);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, int count, float *value, int valueSize/*=1*/) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, count, value, valueSize);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, Opipe::Mat4 value) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, value);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, Vector2 value) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, value);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, Vector4 value) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, value);
        }
    }
    
    void GLProgram::setUniformValue(const std::string &uniformName, Matrix3 value) {
        getContext()->setActiveShaderProgram(this);
        GLuint location = getUniformLocation(uniformName);
        if (location != -1) {
            setUniformValue(location, value);
        }
    }
    
    void GLProgram::setUniformValue(int uniformLocation, int value) {
        getContext()->setActiveShaderProgram(this);
        CHECK_GL(glUniform1i(uniformLocation, value));
    }
    
    void GLProgram::setUniformValue(int uniformLocation, int count, int *value, int valueSize /*=1*/) {
        getContext()->setActiveShaderProgram(this);
        if (valueSize == 1) {
            CHECK_GL(glUniform1iv(uniformLocation, count, value));
        } else if (valueSize == 2) {
            CHECK_GL(glUniform2iv(uniformLocation, count, value));
        } else if (valueSize == 3) {
            CHECK_GL(glUniform3iv(uniformLocation, count, value));
        } else if (valueSize == 4) {
            CHECK_GL(glUniform4iv(uniformLocation, count, value));
        }
        
    }
    
    void GLProgram::setUniformValue(int uniformLocation, float value) {
        getContext()->setActiveShaderProgram(this);
        CHECK_GL(glUniform1f(uniformLocation, value));
    }
    
    void GLProgram::setUniformValue(int uniformLocation, int count, float *value, int valueSize/*=1*/) {
        getContext()->setActiveShaderProgram(this);
        if (valueSize == 1) {
            CHECK_GL(glUniform1fv(uniformLocation, count, value));
        } else if (valueSize == 2) {
            CHECK_GL(glUniform2fv(uniformLocation, count, value));
        } else if (valueSize == 3) {
            CHECK_GL(glUniform3fv(uniformLocation, count, value));
        } else if (valueSize == 4) {
            CHECK_GL(glUniform4fv(uniformLocation, count, value));
        }
    }
    
    void GLProgram::setUniformValue(int uniformLocation, Opipe::Mat4 value) {
        getContext()->setActiveShaderProgram(this);
        CHECK_GL(glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, (GLfloat *) &value));
    }
    
    void GLProgram::setUniformValue(int uniformLocation, Vector2 value) {
        getContext()->setActiveShaderProgram(this);
        CHECK_GL(glUniform2f(uniformLocation, value.x, value.y));
    }
    
    void GLProgram::setUniformValue(int uniformLocation, Vector4 value) {
        getContext()->setActiveShaderProgram(this);
        CHECK_GL(glUniform4f(uniformLocation, value.x, value.y, value.z, value.w));
    }
    
    void GLProgram::setUniformValue(int uniformLocation, Matrix3 value) {
        getContext()->setActiveShaderProgram(this);
        CHECK_GL(glUniformMatrix3fv(uniformLocation, 1, GL_FALSE, (GLfloat *) &value));
    }
    
    Context *GLProgram::getContext() {
        if (_context) {
            return _context;
        }
        
        return NULL;
    }
    
}
