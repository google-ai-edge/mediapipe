#include "mediapipe/render/module/common/OlaGraph.hpp"
#include "mediapipe/render/module/beauty/face_mesh_module.h"

namespace Opipe {
    class FaceMeshModuleIMP : public FaceMeshModule {
        public:
        FaceMeshModuleIMP();
        ~FaceMeshModuleIMP();
    };
}