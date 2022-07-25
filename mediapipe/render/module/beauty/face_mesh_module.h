#ifndef OPIPE_FaceMeshModule
#define OPIPE_FaceMeshModule
#include <stdio.h>
#if defined(__APPLE__)
#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>
#import <CoreVideo/CoreVideo.h>
#elif defined(__ANDROID__) || defined(ANDROID)
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#endif


namespace Opipe
{
    class FaceMeshModule
    {
    public:
        FaceMeshModule();
        virtual ~FaceMeshModule();
        
        static FaceMeshModule* create();
        

        // 暂停渲染
        virtual void suspend() = 0;

        // 恢复渲染
        virtual void resume() = 0;

        virtual bool init(void *env, void *binaryData, int size) = 0;

        virtual void startModule() = 0;

        virtual void stopModule() = 0;  

        virtual GLuint renderTexture(GLuint textureId, int64_t timeStamp, int width, int height) = 0;
        
#if defined(__APPLE__)
        virtual void processVideoFrame(CVPixelBufferRef pixelbuffer, int64_t timeStamp) = 0;
#endif
        
        virtual void processVideoFrame(char *pixelbuffer,
                                       int width,
                                       int height,
                                       int step,
                                       int64_t timeStamp) = 0;
    };
}
#endif
