#ifndef OPIPE_FaceMeshCommon
#define OPIPE_FaceMeshCommon

#include <stdio.h>

typedef struct {
        int width;
        int height;
        int textureId;
        int ioSurfaceId; // iOS 专属
        int64_t frameTime;
} TextureInfo;

#endif