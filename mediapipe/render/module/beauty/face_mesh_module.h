#ifndef OPIPE_FaceMeshModule
#define OPIPE_FaceMeshModule
#include <stdio.h>
#include "mediapipe/render/core/OlaContext.hpp"
#include "face_mesh_common.h"
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
        struct OMat
        {
                int width = 0;
                int height = 0;
                char *data = 0;
                int widthStep = 0;
                int channels = 4; //暂时只支持4
                bool create = false;
                OMat()
                {
                        channels = 0;
                }

                OMat(int w, int h, int ws)
                {
                        width = w;
                        height = h;
                        channels = 4;
                        widthStep = ws;
                        data = new char[widthStep * height];
                        memset(data, 0, sizeof(data));
                        create = true;
                }

                OMat(int w, int h)
                {
                        width = w;
                        height = h;
                        channels = 4;
                        if (w % 32 != 0)
                        {
                                widthStep = ((w / 32) + 1) * 32 * channels;
                        }
                        else
                        {
                                widthStep = w * channels;
                        }

                        data = new char[widthStep * height];
                        memset(data, 0, sizeof(data));
                        create = true;
                }

                OMat(int w, int h, char *d)
                {
                        width = w;
                        height = h;
                        channels = 4;
                        data = d;
                        if (w % 32 != 0)
                        {
                                widthStep = ((w / 32) + 1) * 32 * channels;
                        }
                        else
                        {
                                widthStep = w * channels;
                        }
                }

                void release()
                {
                        if (create)
                        {
                                delete data;
                        }
                        data = 0;
                }

                bool empty()
                {
                        return data == 0;
                }
        };

        class FaceMeshModule
        {
        public:
                FaceMeshModule();
                virtual ~FaceMeshModule();

                static FaceMeshModule *create();

                virtual OlaContext *currentContext() = 0;

                // 暂停渲染
                virtual void suspend() = 0;

                // 恢复渲染
                virtual void resume() = 0;

                virtual bool init(void *env, void *binaryData, int size) = 0;

                virtual void startModule() = 0;

                virtual void stopModule() = 0;

                virtual TextureInfo renderTexture(TextureInfo inputTexture) = 0;

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
