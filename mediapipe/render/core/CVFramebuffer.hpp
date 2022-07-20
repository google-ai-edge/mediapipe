//
//  CVFramebuffer.hpp
//  Quaramera
//
//  Created by wangrenzhu on 2021/4/30.
//

#ifndef CVFramebuffer_hpp
#define CVFramebuffer_hpp

#include <stdio.h>
#include <OpenGLES/ES3/gl.h>
#include <OpenGLES/ES3/glext.h>
#include <CoreVideo/CoreVideo.h>
#include <vector>
#include "Ref.hpp"
#include "Framebuffer.hpp"

namespace Opipe {
    
    
    class CVFramebuffer : public Opipe::Framebuffer {
    public:
        
        CVFramebuffer(Context *context, int width, int height,
                      const TextureAttributes textureAttributes = defaultTextureAttribures,
                      GLuint textureId = -1);
        CVFramebuffer(Context *context, int width, int height, bool onlyGenerateTexture = false,
                      const TextureAttributes textureAttributes = defaultTextureAttribures);

        CVFramebuffer(Context *context,
                      int width, int height,
                      GLuint handle, IOSurfaceID surfaceID,
                      const TextureAttributes textureAttributes = defaultTextureAttribures);
        
        void SetRenderTarget(CVPixelBufferRef pixel_buffer);
        virtual ~CVFramebuffer();
        
        void lockAddress() override;
        void unlockAddress() override;
        void* frameBufferGetBaseAddress() override;
        int getBytesPerRow() override;
        CVPixelBufferRef renderTarget = 0;
        IOSurfaceRef renderIOSurface = 0;
    private:
        
        void _generateTexture() override;
        void _bindFramebuffer();
        void _generateFramebuffer(bool needGenerateTexture = true) override;
        
        CVOpenGLESTextureRef _glTexture = 0;
        IOSurfaceID _ioSurfaceId = -1;
        
        
    };
    
}


#endif /* CVFramebuffer_hpp */
