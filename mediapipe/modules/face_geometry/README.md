# face_transform

Protos|Details
:--- | :---
[`face_geometry.Environment`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/environment.proto)| Describes an environment; includes the camera frame origin point location as well as virtual camera parameters.
[`face_geometry.GeometryPipelineMetadata`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.proto)| Describes metadata needed to estimate face 3D transform based on the face landmark module result.
[`face_geometry.FaceGeometry`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/face_geometry.proto)| Describes 3D transform data for a single face; includes a face mesh surface and a face pose in a given environment.
[`face_geometry.Mesh3d`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/mesh_3d.proto)| Describes a 3D mesh triangular surface.

Calculators|Details
:--- | :---
[`FaceGeometryEnvGeneratorCalculator`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/env_generator_calculator.cc)| Generates an environment that describes a virtual scene.
[`FaceGeometryPipelineCalculator`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/geometry_pipeline_calculator.cc)| Extracts face 3D transform for multiple faces from a vector of landmark lists.
[`FaceGeometryEffectRendererCalculator`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/effect_renderer_calculator.cc)| Renders a face effect.

Subgraphs|Details
:--- | :---
[`FaceGeometryFromDetection`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry_from_detection.pbtxt)| Extracts 3D transform from face detection for multiple faces.
[`FaceGeometryFromLandmarks`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry_from_landmarks.pbtxt)| Extracts 3D transform from face landmarks for multiple faces.
[`FaceGeometry`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry.pbtxt)| Extracts 3D transform from face landmarks for multiple faces. Deprecated, please use `FaceGeometryFromLandmarks` in the new code.
