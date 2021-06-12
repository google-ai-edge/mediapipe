# Modules

Each module (represented as a subfolder) provides subgraphs and corresponding resources (e.g. tflite models) to perform domain-specific tasks (e.g. detect faces, detect face landmarks).

*Modules listed below are already used in some of `mediapipe/graphs` and more graphs are being migrated to use existing and upcoming modules.*

| Module | Description |
| :--- | :--- |
| [`face_detection`](face_detection/README.md) | Subgraphs to detect faces. |
| [`face_geometry`](face_geometry/README.md) | Subgraphs to extract face geometry. |
| [`face_landmark`](face_landmark/README.md) | Subgraphs to detect and track face landmarks. |
| [`hand_landmark`](hand_landmark/README.md) | Subgraphs to detect and track hand landmarks. |
| [`holistic_landmark`](holistic_landmark/README.md) | Subgraphs to detect and track holistic pose which consists of pose, face and hand landmarks. |
| [`iris_landmark`](iris_landmark/README.md) | Subgraphs to detect iris landmarks. |
| [`palm_detection`](palm_detection/README.md) | Subgraphs to detect palms/hands. |
| [`pose_detection`](pose_detection/README.md) | Subgraphs to detect poses. |
| [`pose_landmark`](pose_landmark/README.md) | Subgraphs to detect and track pose landmarks. |
| [`objectron`](objectron/README.md) | Subgraphs to detect and track 3D objects. |
