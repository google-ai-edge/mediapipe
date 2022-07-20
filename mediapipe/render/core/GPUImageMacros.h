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

#ifndef GPUImageMacros_h
#define GPUImageMacros_h


#define NS_GI_BEGIN                     namespace Opipe {
#define NS_GI_END                       }
#define USING_NS_GI                     using namespace Opipe;


#define STRINGIZE(x) #x
#define SHADER_STRING(text) STRINGIZE(text)

#define PI 3.14159265358979323846264338327950288

#if defined(__ANDROID__) || defined(ANDROID)
  #define ENABLE_GL_CHECK true
#else
  #define ENABLE_GL_CHECK false
#endif
#ifndef PLATFORM_WINDOWS

#if ENABLE_GL_CHECK
#define CHECK_GL(glFunc) \
        glFunc; \
    { \
        int e = glGetError(); \
        if (e != 0) \
        { \
            std::string errorString = ""; \
            switch (e) \
            { \
            case GL_INVALID_ENUM:       errorString = "GL_INVALID_ENUM";        break; \
            case GL_INVALID_VALUE:      errorString = "GL_INVALID_VALUE";       break; \
            case GL_INVALID_OPERATION:  errorString = "GL_INVALID_OPERATION";   break; \
            case GL_OUT_OF_MEMORY:      errorString = "GL_OUT_OF_MEMORY";       break; \
            default:                                                            break; \
            } \
            Opipe::Log("QuarameraGL", "GL ERROR 0x%04X %s in %s at line %i\n", e, \
            errorString.c_str(), __PRETTY_FUNCTION__, __LINE__); \
            assert(0);\
        } \
    }
#else
#define CHECK_GL(glFunc)  glFunc;
#endif

#endif

#  ifndef GPUIMAGE_EXPORT
#    define GPUIMAGE_EXPORT __attribute__((visibility("default")))
#  endif

#include "GPUImageUtil.h"



#endif /* macros_h */
