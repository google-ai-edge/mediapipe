#ifndef OlaContext_hpp
#define OlaContext_hpp

#if defined(__APPLE__)
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES3/gl.h>
#import <CoreVideo/CoreVideo.h>
#else
#include <EGL/egl.h>
#endif

namespace Opipe {
    class Context;
    class OlaContext {
        public:
#if defined(__APPLE__)
        OlaContext(EAGLContext *context);
#endif
        OlaContext();
        ~OlaContext();

        #if defined(__APPLE__)
        EAGLContext* currentContext();
        #else
        EGLContext* currentContext();
        void initEGLContext(EGLContext shareContext);
        #endif

        Context* glContext();

        private:
        Context *_currentContext = nullptr;

    };
}

#endif