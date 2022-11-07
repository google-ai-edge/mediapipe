//
// Created by Mautisim Munir on 05/11/2022.
//

#include "calculator_registry.h"
#include "mediapipe/calculators/core/flow_limiter_calculator.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.h"

// We need namespaces for subgraphs because of the static variables inside the files
namespace PLG {

#include "mediapipe/modules/pose_landmark/pose_landmark_gpu_linked.h"

}

namespace PDROI{
#include "mediapipe/modules/pose_landmark/pose_detection_to_roi_linked.h"

}

namespace PRG {

#include "mediapipe/graphs/pose_tracking/subgraphs/pose_renderer_gpu_linked.h"

}
namespace PDG {
#include "mediapipe/modules/pose_detection/pose_detection_gpu_linked.h"
}

namespace PLROIG{
#include "mediapipe/modules/pose_landmark/pose_landmark_by_roi_gpu_linked.h"

}
namespace PLF{
#include "mediapipe/modules/pose_landmark/pose_landmark_filtering_linked.h"
}
namespace PLTOROI{
#include "mediapipe/modules/pose_landmark/pose_landmarks_to_roi_linked.h"
}
namespace PSF{
#include "mediapipe/modules/pose_landmark/pose_segmentation_filtering_linked.h"
}
namespace PLML{
#include "mediapipe/modules/pose_landmark/pose_landmark_model_loader_linked.h"
}
namespace PLSIP{
#include "mediapipe/modules/pose_landmark/pose_landmarks_and_segmentation_inverse_projection_linked.h"
}
namespace TPLS{
#include "mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation_linked.h"
}

MPPCalculator::MPPCalculator() {
    typeid(TPLS::mediapipe::TensorsToPoseLandmarksAndSegmentation);
    typeid(::mediapipe::FlowLimiterCalculator);
    typeid(::mediapipe::ConstantSidePacketCalculator);
    typeid(PLG::mediapipe::PoseLandmarkGpu);
    typeid(PRG::mediapipe::PoseRendererGpu);
    typeid(PDG::mediapipe::PoseDetectionGpu);
    typeid(PDROI::mediapipe::PoseDetectionToRoi);
    typeid(PLROIG::mediapipe::PoseLandmarkByRoiGpu);
    typeid(PLF::mediapipe::PoseLandmarkFiltering);
    typeid(PLTOROI::mediapipe::PoseLandmarksToRoi);
    typeid(PSF::mediapipe::PoseSegmentationFiltering);
    typeid(PLML::mediapipe::PoseLandmarkModelLoader);
    typeid(PLSIP::mediapipe::PoseLandmarksAndSegmentationInverseProjection);

}
