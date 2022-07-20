

#ifndef macros_h
#define macros_h

#define PLATFORM_UNKNOW 0
#define PLATFORM_ANDROID 1
#define PLATFORM_IOS 2

#define PLATFORM PLATFORM_UNKNOW
#if defined(__ANDROID__) || defined(ANDROID)
#undef  PLATFORM
#define PLATFORM PLATFORM_ANDROID
#include <assert.h>
#elif defined(__APPLE__)
#undef  PLATFORM
#define PLATFORM PLATFORM_IOS
#endif

#define NS_OLA_BEGIN                     namespace Opipe {
#define NS_OLA_END                       }
#define USING_NS_OLA                     using namespace Opipe;


#define STRINGIZE(x) #x
#define SHADER_STRING(text) STRINGIZE(text)

#define PI 3.14159265358979323846264338327950288

#define ENABLE_GL_CHECK false

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
            Opipe::Log("ERROR", "GL ERROR 0x%04X %s in %s at line %i\n", e, \
            errorString.c_str(), __PRETTY_FUNCTION__, __LINE__); \
            assert(0);\
        } \
    }
#else
#define CHECK_GL(glFunc)  glFunc;
#endif

#endif

#include "util.h"

#endif /* macros_h */
