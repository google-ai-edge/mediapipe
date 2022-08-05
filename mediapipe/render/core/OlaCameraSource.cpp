#include "OlaCameraSource.hpp"
#include "Context.hpp""
#if defined(__APPLE__)
#import <OpenGLES/EAGLIOSurface.h>
#endif

using namespace Opipe;

namespace Opipe
{

OlaCameraSource::OlaCameraSource(Context *context, SourceType sourceType) : SourceCamera(context)
{
    _sourceType = sourceType;
    _lastIOSurface = -1;
    
    switch (_sourceType)
    {
        case SourceType_RGBA:
            _yuvTexture = nullptr;
            break;
        case SourceType_YUV420SP:
            _yuvTexture = OlaYUVTexture::create(context);
            break;
        case SourceType_YUV420P:
            _yuvTexture = OlaYUVTexture420P::create(context);
            break;
        default:
            break;
    }
    if (_yuvTexture) {
        _scaleTexture = OlaShareTextureFilter::create(context);
        _scaleTexture->setFramebufferScale(0.5);
        addTarget(_yuvTexture);
        _yuvTexture->addTarget(_scaleTexture);
    } else {
        addTarget(_scaleTexture);
    }
}

OlaCameraSource::~OlaCameraSource()
{
    if (_yuvTexture)
    {
        _yuvTexture->removeAllTargets();
        _yuvTexture->release();
        _yuvTexture = nullptr;
    }
    
    if (_scaleTexture) {
        _scaleTexture->release();
        _scaleTexture = nullptr;
    }
}

void OlaCameraSource::setFrameData(int width,
                                   int height,
                                   const void *pixels,
                                   GLenum type,
                                   GLuint texture,
                                   RotationMode outputRotation,
                                   SourceType sourceType,
                                   const void *upixels,
                                   const void *vpixels,
                                   bool keep_white)
{
    if (_sourceType != sourceType)
    {
        _sourceType = sourceType;
        if (_yuvTexture)
        {
            _yuvTexture->removeAllTargets();
            _yuvTexture->release();
            _yuvTexture = nullptr;
        }
        
        removeAllTargets();
        
        switch (_sourceType)
        {
            case SourceType_RGBA:
                _yuvTexture = nullptr;
                break;
            case SourceType_YUV420SP:
                _yuvTexture = new OlaYUVTexture(_context);
                break;
            case SourceType_YUV420P:
                _yuvTexture = new OlaYUVTexture420P(_context);
                break;
            default:
                break;
        }
        if (_yuvTexture) {
            addTarget(_yuvTexture);
        }
    }
    
    SourceCamera::setFrameData(width, height, pixels, type, texture,
                               outputRotation, sourceType,
                               upixels, vpixels, keep_white);
}

OlaCameraSource* OlaCameraSource::create(Context *context)
{
    return new OlaCameraSource(context);
}

Source* OlaCameraSource::addTarget(Target *target)
{
    if (_yuvTexture && target != _yuvTexture)
    {
        return _yuvTexture->addTarget(target);
    }
    return SourceCamera::addTarget(target);
}

#if defined(__APPLE__)
void OlaCameraSource::setIORenderTexture(IOSurfaceID surfaceID,
                                         GLuint texture,
                                         int width,
                                         int height,
                                         RotationMode outputRotation,
                                         SourceType sourceType,
                                         const TextureAttributes textureAttributes)
{
    // iOS 版不支持切换格式 要么RGBA 要么YUV420F
    _sourceType = sourceType;
    if (sourceType == SourceType_RGBA) {
        SourceCamera::setIORenderTexture(surfaceID, texture, width, height,
                                         outputRotation, sourceType, textureAttributes);
    } else {
        if (surfaceID != _lastIOSurface) {
            // surfaceID 变了需要重新创建Framebuffer
            _bindIOSurfaceToTexture(surfaceID);
            _lastIOSurface = surfaceID;
        }
        setFramebuffer(_framebuffer, outputRotation);
    }
}

void OlaCameraSource::_bindIOSurfaceToTexture(int iosurface, RotationMode outputRotation)
{
    IOSurfaceRef surface = IOSurfaceLookup(iosurface);
    int width = (int)IOSurfaceGetWidth(surface);
    int height = (int)IOSurfaceGetHeight(surface);
    if (surface)
    {
        if (_UVFrameBuffer == nullptr) {
            _UVFrameBuffer = _context->getFramebufferCache()->
            fetchFramebuffer(_context, width * 0.5, height * 0.5, true);
        }
        EAGLContext *eglContext = _context->getEglContext();
        if (_UVFrameBuffer) {
            _UVFrameBuffer->active();
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            
            BOOL rs = [eglContext texImageIOSurface:surface target:GL_TEXTURE_2D internalFormat:GL_LUMINANCE_ALPHA
                                              width:width * 0.5 height:height * 0.5 format:GL_LUMINANCE_ALPHA type:GL_UNSIGNED_BYTE plane:1];
            if (rs) {
                Log("Opipe", "IOSurface 绑定UV Texture 成功");
            }
        }
        
        this->setFramebuffer(nullptr);
        Framebuffer* framebuffer = _context->getFramebufferCache()->fetchFramebuffer(_context, width, height, true);
        
        this->setFramebuffer(framebuffer, outputRotation);
        
        _framebuffer->active();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                        GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        BOOL rs = [eglContext texImageIOSurface:surface target:GL_TEXTURE_2D internalFormat:GL_LUMINANCE
                                          width:width height:height format:GL_LUMINANCE type:GL_UNSIGNED_BYTE plane:0];
        if (rs) {
            Log("Opipe", "IOSurface 绑定Y Texture 成功");
        }
        
    }
}

Framebuffer* OlaCameraSource::getScaleFramebuffer() {
    if (_scaleTexture && _scaleTexture->getFramebuffer()) {
        return _scaleTexture->getFramebuffer();
    } else {
        return nullptr;
    }
}

#endif
}
