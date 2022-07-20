#include "mediapipe/render/module/common/ola_graph.h"
#include "mediapipe/render/core/OpipeDispatch.hpp"
#include "face_mesh_module.h"

namespace Opipe
{
    class FaceMeshCallFrameDelegate : public MPPGraphDelegate
    {
    public:
        FaceMeshCallFrameDelegate();
        ~FaceMeshCallFrameDelegate();
#if defined(__APPLE__)
        void outputPixelbuffer(OlaGraph *graph, CVPixelBufferRef pixelbuffer,
                               const std::string &streamName, int64_t timstamp) override;
#endif
        void outputPacket(OlaGraph *graph, const mediapipe::Packet &packet,
                          MPPPacketType packetType, const std::string &streamName) override;
    };

    class FaceMeshModuleIMP : public FaceMeshModule
    {
    public:
        FaceMeshModuleIMP();
        ~FaceMeshModuleIMP();

        // 暂停渲染
        virtual void suspend() override;

        // 恢复渲染
        virtual void resume() override;

        // env iOS给空
        virtual bool init(void *env, void *binaryData, int size) override;

        virtual void startModule() override;

        virtual void stopModule() override;

#if defined(__APPLE__)

        /// 算法流输入
        /// @param pixelbuffer pixelbuffer description
        /// @param timeStamp timeStamp description
        virtual void processVideoFrame(CVPixelBufferRef pixelbuffer, int64_t timeStamp) override;
#endif

        virtual void processVideoFrame(char *pixelbuffer,
                                       int width,
                                       int height,
                                       int step,
                                       int64_t timeStamp) override;

        virtual GLuint renderTexture(GLuint textureId, int64_t timeStamp, int width, int height) override;

    private:
        std::unique_ptr<OpipeDispatch> _dispatch;
        std::unique_ptr<OlaGraph> _graph;
        std::unique_ptr<Context> _context;
        bool _isInit = false;
        std::shared_ptr<FaceMeshCallFrameDelegate> _delegate;
    };
}
