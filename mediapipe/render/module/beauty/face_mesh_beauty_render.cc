#include "face_mesh_beauty_render.h"

namespace Opipe
{
    FaceMeshBeautyRender::FaceMeshBeautyRender(Context *context)
    {
        _context = context;
        _olaBeautyFilter = OlaBeautyFilter::create(context);
        _isRendering = false;

        _outputFilter = OlaShareTextureFilter::create(context);
        _olaBeautyFilter->addTarget(_outputFilter);
    }

    FaceMeshBeautyRender::~FaceMeshBeautyRender()
    {
    }

    void FaceMeshBeautyRender::suspend()
    {
        _isRendering = true;
    }

    void FaceMeshBeautyRender::resume()
    {
        _isRendering = false;
    }

    TextureInfo FaceMeshBeautyRender::renderTexture(TextureInfo inputTexture)
    {
        TextureInfo outputTexture;

        if (!_inputFramebuffer)
        {
            _inputFramebuffer = new Framebuffer(_context, inputTexture.width, inputTexture.height,
                                                Framebuffer::defaultTextureAttribures,
                                                inputTexture.textureId);
        }
        else if (_inputFramebuffer->getWidth() != inputTexture.width || _inputFramebuffer->getHeight() != inputTexture.height)
        {
            _inputFramebuffer->unlock();
            delete _inputFramebuffer;
            _inputFramebuffer = nullptr;
            _inputFramebuffer = new Framebuffer(_context, inputTexture.width, inputTexture.height,
                                                Framebuffer::defaultTextureAttribures,
                                                inputTexture.textureId);
        }

        return outputTexture;
    }

    float FaceMeshBeautyRender::getSmoothing()
    {
        return _smoothing;
    }

    float FaceMeshBeautyRender::getWhitening()
    {
        return _whitening;
    }

    void FaceMeshBeautyRender::setSmoothing(float smoothing)
    {
        _smoothing = smoothing;
    }

    void FaceMeshBeautyRender::setWhitening(float whitening)
    {
        _whitening = whitening;
    }

}