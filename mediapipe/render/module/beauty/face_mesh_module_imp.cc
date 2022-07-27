#include "face_mesh_module_imp.h"

static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kLandmarksOutputStream = "multi_face_landmarks";
static const char* kDetectionsOutputStream = "face_detections";
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

    void FaceMeshCallFrameDelegate::outputPacket(OlaGraph *graph, const mediapipe::Packet &packet, const std::string &streamName) {
#if defined(__APPLE__)
        NSLog(@"streamName:%@ ts:%lld 是否有人脸:%@", [NSString stringWithUTF8String:streamName.c_str()],
              packet.Timestamp().Value(), @(_hasFace));
#endif
        if (_imp == nullptr) {
            return;
        }
        
        
        if (streamName == kLandmarksOutputStream) {
            _last_landmark_ts = packet.Timestamp().Value();
            
            if (_last_video_ts == _last_landmark_ts) {
                //有人脸
                _hasFace = true;
                const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
                _lastLandmark = multi_face_landmarks[0];
            }
        }
        
        if (_last_video_ts != _last_landmark_ts) {
            _hasFace = false;
        }
        
        _last_video_ts = packet.Timestamp().Value();
        
        if (_hasFace) {
            
            _imp->setLandmark(_lastLandmark);
        } else {
            _imp->setLandmark(_emptyLandmark);
        }
    }

    void FaceMeshCallFrameDelegate::outputPacket(OlaGraph *graph, const mediapipe::Packet &packet,
                                                 MPPPacketType packetType, const std::string &streamName)
    {
#if defined(__APPLE__)
//        if (streamName == kLandmarksOutputStream) {
//          if (packet.IsEmpty()) {
//            NSLog(@"[TS:%lld] No face landmarks", packet.Timestamp().Value());
//            return;
//          }
//          const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
//          NSLog(@"[TS:%lld] Number of face instances with landmarks: %lu", packet.Timestamp().Value(),
//                multi_face_landmarks.size());
//          for (int face_index = 0; face_index < multi_face_landmarks.size(); ++face_index) {
//            const auto& landmarks = multi_face_landmarks[face_index];
//            NSLog(@"\tNumber of landmarks for face[%d]: %d", face_index, landmarks.landmark_size());
//            for (int i = 0; i < landmarks.landmark_size(); ++i) {
//              NSLog(@"\t\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
//                    landmarks.landmark(i).y(), landmarks.landmark(i).z());
//            }
//          }
//        } else if (streamName == kDetectionsOutputStream) {
//            if (packet.IsEmpty()) {
//              NSLog(@"[TS:%lld] No face detections", packet.Timestamp().Value());
//              return;
//            }
//            const auto& face_detections = packet.Get<std::vector<::mediapipe::Detection>>();
//            NSLog(@"[TS:%lld] Number of face instances with detections: %lu", packet.Timestamp().Value(),
//                  face_detections.size());
//
//        }
#endif
    }

    FaceMeshModuleIMP::FaceMeshModuleIMP()
    {
    }

    FaceMeshModuleIMP::~FaceMeshModuleIMP()
    {
        _delegate->attach(nullptr);
        _delegate = 0;
        
        if (_olaContext) {
            delete _olaContext;
            _olaContext = nullptr;
        }

        _graph = nullptr;

        if (_render) {
            _dispatch->runSync([&] {
                delete _render;
                _render = nullptr;
            });
        }
        
        _context = nullptr;


    }

    void FaceMeshModuleIMP::suspend()
    {
        _render->suspend();
    }

    void FaceMeshModuleIMP::resume()
    {
        _render->resume();
    }

    bool FaceMeshModuleIMP::init(void *env, void *binaryData,
                                 int size)
    {
        std::string graphName = "face_mesh_mobile_gpu";
        _delegate = std::make_shared<FaceMeshCallFrameDelegate>();
        _delegate->attach(this);
        mediapipe::CalculatorGraphConfig config;
        config.ParseFromArray(binaryData, size);
        _olaContext = new OlaContext();
        _context = _olaContext->glContext();
        _render = new FaceMeshBeautyRender(_context);

    #if defined(__ANDROID__)
        _context->initEGLContext(env);
    #endif

        _dispatch = std::make_unique<OpipeDispatch>(_context, nullptr, nullptr);

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

    void FaceMeshModuleIMP::setLandmark(NormalizedLandmarkList landmark)
    {
        _lastLandmark = std::move(landmark);
        if (_lastLandmark.landmark_size() == 0) {
#if defined(__APPLE__)
            NSLog(@"没有人脸");
#endif
        }
        for (int i = 0; i < _lastLandmark.landmark_size(); ++i) {
#if defined(__APPLE__)
            NSLog(@"######## Set Landmark[%d]: (%f, %f, %f)", i, _lastLandmark.landmark(i).x(),
                  _lastLandmark.landmark(i).y(), _lastLandmark.landmark(i).z());
#endif
        }
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

    TextureInfo FaceMeshModuleIMP::renderTexture(TextureInfo inputTexture)
    {
        TextureInfo textureInfo;

        if (!_isInit)
        {
            return textureInfo;
        }

        _dispatch->runSync([&] {
            textureInfo = _render->renderTexture(inputTexture);
        });

        return textureInfo;
    }

    // OlaContext* currentContext() {
    //     return _olaContext;
    // }

}
