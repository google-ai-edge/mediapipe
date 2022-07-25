#include "face_mesh_module_imp.h"
#include "mediapipe/framework/formats/landmark.pb.h"

static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kLandmarksOutputStream = "multi_face_landmarks";
static const char* kOutputVideo = "output_video";

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
#if defined(__APPLE__)
        if (streamName == kLandmarksOutputStream) {
          if (packet.IsEmpty()) {
            NSLog(@"[TS:%lld] No face landmarks", packet.Timestamp().Value());
            return;
          }
          const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
          NSLog(@"[TS:%lld] Number of face instances with landmarks: %lu", packet.Timestamp().Value(),
                multi_face_landmarks.size());
          for (int face_index = 0; face_index < multi_face_landmarks.size(); ++face_index) {
            const auto& landmarks = multi_face_landmarks[face_index];
            NSLog(@"\tNumber of landmarks for face[%d]: %d", face_index, landmarks.landmark_size());
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
              NSLog(@"\t\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
                    landmarks.landmark(i).y(), landmarks.landmark(i).z());
            }
          }
        }
#endif
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
        _graph->setSidePacket(mediapipe::MakePacket<int>(1), kNumFacesInputSidePacket);
        _graph->addFrameOutputStream(kLandmarksOutputStream, MPPPacketTypeRaw);
#if defined(__APPLE__)
        _graph->addFrameOutputStream(kOutputVideo, MPPPacketTypePixelBuffer);
#endif
        _isInit = true;

        return true;
    }

    void FaceMeshModuleIMP::startModule()
    {
        if (!_isInit)
        {
            return;
        }
        _isInit = _graph->start(); 
    }

    void FaceMeshModuleIMP::stopModule()
    {
        if (!_isInit)
        {
            return;
        }
        _graph->setDelegate(nullptr);
        _graph->cancel();
        _graph->closeAllInputStreams();
        _graph->waitUntilDone();
    }

#if defined(__APPLE__)
    void FaceMeshModuleIMP::processVideoFrame(CVPixelBufferRef pixelbuffer,
                                              int64_t timeStamp)
    {
        if (!_isInit)
        {
            return;
        }
        Timestamp ts(timeStamp * 1000);
        CVPixelBufferLockBaseAddress(pixelbuffer, 0);
        _graph->sendPixelBuffer(pixelbuffer, "input_video",
        MPPPacketTypePixelBuffer,
        ts);
        CVPixelBufferUnlockBaseAddress(pixelbuffer, 0);
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
