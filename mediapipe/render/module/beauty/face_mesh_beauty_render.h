#ifndef OPIPE_FaceMeshBeautyRender
#define OPIPE_FaceMeshBeautyRender
#include "face_mesh_common.h"
#include "mediapipe/render/module/beauty/filters/OlaBeautyFilter.hpp"
#include "mediapipe/render/core/OlaShareTextureFilter.hpp"
#include "mediapipe/render/core/SourceImage.hpp"

namespace Opipe {
    class FaceMeshBeautyRender {
        public:
            FaceMeshBeautyRender(Context *context);
            ~FaceMeshBeautyRender();

            void suspend();

            void resume();

            TextureInfo renderTexture(TextureInfo inputTexture);

            /// 磨皮
            float getSmoothing();
            
            /// 美白
            float getWhitening();
            
            
            /// 磨皮
            /// @param smoothing 磨皮 0.0 - 1.0
            void setSmoothing(float smoothing);
            
            
            /// 美白
            /// @param whitening 美白 0.0 - 1.0
            void setWhitening(float whitening);
        private:
            OlaBeautyFilter *_olaBeautyFilter = nullptr;
            OlaShareTextureFilter *_outputFilter = nullptr;
            Framebuffer *_inputFramebuffer = nullptr;
            float _smoothing = 0.0;
            float _whitening = 0.0;
            bool _isRendering = false;
            Context *_context = nullptr;
            SourceImage *_lutImage = nullptr;

    };
    
}
#endif
