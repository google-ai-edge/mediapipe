# face_geometry

Protos|Details
:--- | :---
[`face_geometry.Environment`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/environment.proto)| Describes an environment; includes the camera frame origin point location as well as virtual camera parameters.
[`face_geometry.GeometryPipelineMetadata`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.proto)| Describes metadata needed to estimate face geometry based on the face landmark module result.
[`face_geometry.FaceGeometry`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/face_geometry.proto)| Describes geometry data for a single face; includes a face mesh surface and a face pose in a given environment.
[`face_geometry.Mesh3d`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/mesh_3d.proto)| Describes a 3D mesh surface.

Calculators|Details
:--- | :---
[`FaceGeometryEnvGeneratorCalculator`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/env_generator_calculator.cc)| Generates an environment that describes a virtual scene.
[`FaceGeometryPipelineCalculator`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/geometry_pipeline_calculator.cc)| Extracts face geometry for multiple faces from a vector of landmark lists.
[`FaceGeometryEffectRendererCalculator`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/effect_renderer_calculator.cc)| Renders a face effect.

Subgraphs|Details
:--- | :---
[`FaceGeometry`](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry.pbtxt)| Extracts face geometry from landmarks for multiple faces.
