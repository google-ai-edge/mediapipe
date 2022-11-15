//
// Created by Mautisim Munir on 05/11/2022.
//
#include <string>

#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include <functional>
#include "calculator_registry.h"
#include <typeinfo>

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
