#ifndef OPIPE_FaceMeshBeautyRender
#define OPIPE_FaceMeshBeautyRender
#include "face_mesh_common.h"
#include "mediapipe/render/module/beauty/filters/OlaBeautyFilter.hpp"
#include "mediapipe/render/core/OlaShareTextureFilter.hpp"
#include "mediapipe/render/core/SourceImage.hpp"
#include "mediapipe/render/core/math/vec2.hpp"
namespace Opipe {
    class FaceMeshBeautyRender {
        public:
            FaceMeshBeautyRender(Context *context);
            ~FaceMeshBeautyRender();

            void suspend();

            void resume();

            void renderTexture(TextureInfo inputTexture);
        
            TextureInfo outputRenderTexture(TextureInfo inputTexture);

            /// 磨皮
            float getSmoothing();
            
            /// 美白
            float getWhitening();
            
            float getEye() {
                return _eyeFactor;
            }
        
            
            float getFace() {
                return _faceFactor;
            }
        
            float getNose() {
                return _noseFactor;
            }
        
            /// 磨皮
            /// @param smoothing 磨皮 0.0 - 1.0
            void setSmoothing(float smoothing);
            
            
            /// 美白
            /// @param whitening 美白 0.0 - 1.0
            void setWhitening(float whitening);
        
            /// 设置人脸点 mediapipe版
            void setFacePoints(std::vector<Vec2> facePoints);
            
            // 大眼
            void setEye(float eyeFactor);
        
            // 瘦脸
            void setFaceSlim(float slimFactor);
        
            // 瘦鼻
            void setNoseFactor(float noseFactor);

            Filter* outputFilter() {
                return _outputFilter;
            }
        
            void setInputSource(Source *source);             
        
        private:
            Source *_source = nullptr;
            OlaBeautyFilter *_olaBeautyFilter = nullptr;
            OlaShareTextureFilter *_outputFilter = nullptr;
            Framebuffer *_inputFramebuffer = nullptr;
            float _smoothing = 0.0;
            float _whitening = 0.0;
            float _noseFactor = 0.0;
            float _faceFactor = 0.0;
            float _eyeFactor = 0.0;
            bool _isRendering = false;
            Context *_context = nullptr;
            SourceImage *_lutImage = nullptr;

    };
    
}
#endif
