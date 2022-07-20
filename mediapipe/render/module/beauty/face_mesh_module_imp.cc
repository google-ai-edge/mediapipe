#include "face_mesh_module_imp.h"

namespace Opipe
{

    FaceMeshCallFrameDelegate::FaceMeshCallFrameDelegate()
    {
    }

    FaceMeshCallFrameDelegate::~FaceMeshCallFrameDelegate()
    {
    }
#if defined(__APPLE__)
    void FaceMeshCallFrameDelegate::outputPixelbuffer(OlaGraph *graph, CVPixelBufferRef pixelbuffer,
                                                      const std::string &streamName, int64_t timstamp)
    {

    }
#endif

    void FaceMeshCallFrameDelegate::outputPacket(OlaGraph *graph, const mediapipe::Packet &packet,
                                                 MPPPacketType packetType, const std::string &streamName)
    {

    }

    FaceMeshModuleIMP::FaceMeshModuleIMP()
    {
    }

    FaceMeshModuleIMP::~FaceMeshModuleIMP()
    {
    }

    void FaceMeshModuleIMP::suspend()
    {

    }

    void FaceMeshModuleIMP::resume()
    {
    }

    bool FaceMeshModuleIMP::init(void *env, void *binaryData,
                                 int size)
    {
        std::string graphName = "face_mesh_mobile_gpu";
        _delegate = std::make_shared<FaceMeshCallFrameDelegate>();
        mediapipe::CalculatorGraphConfig config;
        config.ParseFromArray(binaryData, size);
        _context = std::make_unique<Context>();
        
    #if defined(__ANDROID__)
        _context->initEGLContext(env);
    #endif

        _dispatch = std::make_unique<OpipeDispatch>(_context.get(), nullptr, nullptr);

        _graph = std::make_unique<OlaGraph>(config);
        _graph->setDelegate(_delegate);

        return true;
    }

    void FaceMeshModuleIMP::startModule()
    {
        if (!_isInit)
        {
            return;
        }
        _dispatch->runSync([&] {
             _graph->start(); 
        }, Context::IOContext);
    }

    void FaceMeshModuleIMP::stopModule()
    {
        if (!_isInit)
        {
            return;
        }
        _dispatch->runSync([&] {
            _graph->setDelegate(nullptr);
            _graph->cancel();
            _graph->closeAllInputStreams();
            _graph->waitUntilDone(); 
        }, Context::IOContext);
    }

#if defined(__APPLE__)
    void FaceMeshModuleIMP::processVideoFrame(CVPixelBufferRef pixelbuffer,
                                              int64_t timeStamp)
    {
        if (!_isInit)
        {
            return;
        }
        _dispatch->runAsync([&] { 
            Timestamp ts(timeStamp);
            _graph->sendPixelBuffer(pixelbuffer, "input_video", 
            MPPPacketTypePixelBuffer,
            ts);
        }, Context::IOContext);
    }
#endif

    void FaceMeshModuleIMP::processVideoFrame(char *pixelbuffer,
                                              int width,
                                              int height,
                                              int step,
                                              int64_t timeStamp)
    {
        if (!_isInit)
        {
            return;
        }
    }

    GLuint FaceMeshModuleIMP::renderTexture(GLuint textureId,
                                            int64_t timeStamp,
                                            int width, int height)
    {
        if (!_isInit)
        {
            return textureId;
        }
        _dispatch->runAsync([&] {

        });
        return textureId;
    }

}
