/** @externs */

const DrawingUtils = {};
DrawingUtils.clamp = function() {};
DrawingUtils.lerp = function() {};
DrawingUtils.drawLandmarks = function() {};
DrawingUtils.drawConnectors = function() {};
DrawingUtils.drawBoundingBox = function() {};

const FaceDetector = {};
FaceDetector.createFromModelBuffer = function() {};
FaceDetector.createFromOptions = function() {};
FaceDetector.createFromModelPath = function() {};
FaceDetector.detect = function() {};
FaceDetector.detectForVideo = function() {};
FaceDetector.setOptions = function() {};
FaceDetector.close = function() {};

const FaceLandmarker = {};
FaceLandmarker.FACE_LANDMARKS_LIPS = {};
FaceLandmarker.FACE_LANDMARKS_LEFT_EYE = {};
FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW = {};
FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS = {};
FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE = {};
FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW = {};
FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS = {};
FaceLandmarker.FACE_LANDMARKS_FACE_OVAL = {};
FaceLandmarker.FACE_LANDMARKS_CONTOURS = {};
FaceLandmarker.FACE_LANDMARKS_TESSELATION = {};
FaceLandmarker.createFromModelBuffer = function() {};
FaceLandmarker.createFromOptions = function() {};
FaceLandmarker.createFromModelPath = function() {};
FaceLandmarker.detect = function() {};
FaceLandmarker.detectForVideo = function() {};
FaceLandmarker.setOptions = function() {};
FaceLandmarker.close = function() {};

const FaceStylizer = {};
FaceStylizer.createFromModelBuffer = function() {};
FaceStylizer.createFromOptions = function() {};
FaceStylizer.createFromModelPath = function() {};
FaceStylizer.stylize = function() {};
FaceStylizer.setOptions = function() {};
FaceStylizer.close = function() {};

const FilesetResolver = {};
FilesetResolver.isSimdSupported = function() {};
FilesetResolver.forAudioTasks = function() {};
FilesetResolver.forTextTasks = function() {};
FilesetResolver.forVisionTasks = function() {};

const GestureRecognizer = {};
GestureRecognizer.createFromModelBuffer = function() {};
GestureRecognizer.createFromOptions = function() {};
GestureRecognizer.createFromModelPath = function() {};
GestureRecognizer.recognize = function() {};
GestureRecognizer.recognizeForVideo = function() {};
GestureRecognizer.setOptions = function() {};
GestureRecognizer.close = function() {};

const HandLandmarker = {};
HandLandmarker.HAND_CONNECTIONS = {};
HandLandmarker.createFromModelBuffer = function() {};
HandLandmarker.createFromOptions = function() {};
HandLandmarker.createFromModelPath = function() {};
HandLandmarker.detect = function() {};
HandLandmarker.detectForVideo = function() {};
HandLandmarker.setOptions = function() {};
HandLandmarker.close = function() {};

const ImageClassifier = {};
ImageClassifier.createFromModelBuffer = function() {};
ImageClassifier.createFromOptions = function() {};
ImageClassifier.createFromModelPath = function() {};
ImageClassifier.classify = function() {};
ImageClassifier.classifyForVideo = function() {};
ImageClassifier.setOptions = function() {};
ImageClassifier.close = function() {};

const ImageEmbedder = {};
ImageEmbedder.createFromModelBuffer = function() {};
ImageEmbedder.createFromOptions = function() {};
ImageEmbedder.createFromModelPath = function() {};
ImageEmbedder.embded = function() {};
ImageEmbedder.embedForVideo = function() {};
ImageEmbedder.setOptions = function() {};
ImageEmbedder.cosineSimilarity = function() {};
ImageEmbedder.close = function() {};

const ImageSegmenter = {};
ImageSegmenter.createFromModelBuffer = function() {};
ImageSegmenter.createFromOptions = function() {};
ImageSegmenter.createFromModelPath = function() {};
ImageSegmenter.segmment = function() {};
ImageSegmenter.segmentForVideo = function() {};
ImageSegmenter.setOptions = function() {};
ImageSegmenter.getLabels = function() {};
ImageSegmenter.close = function() {};

const InteractiveSegmenter = {};
InteractiveSegmenter.createFromModelBuffer = function() {};
InteractiveSegmenter.createFromOptions = function() {};
InteractiveSegmenter.createFromModelPath = function() {};
InteractiveSegmenter.segmment = function() {};
InteractiveSegmenter.setOptions = function() {};
InteractiveSegmenter.close = function() {};

const MPImage = {};
MPImage.hasImageData = function() {};
MPImage.hasImageBitmap = function() {};
MPImage.hasWebGLTexture = function() {};
MPImage.getAsImageData = function() {};
MPImage.getAsImageBitmap = function() {};
MPImage.getAsWebGLTexture = function() {};
MPImage.clone = function() {};
MPImage.close = function() {};

const MPMask = {};
MPMask.hasUint8Array = function() {};
MPMask.hasFloat32Array = function() {};
MPMask.hasWebGLTexture = function() {};
MPMask.getAsUint8Array = function() {};
MPMask.getAsFloat32Array = function() {};
MPMask.getAsWebGLTexture = function() {};
MPMask.clone = function() {};
MPMask.close = function() {};

const ObjectDetector = {};
ObjectDetector.createFromModelBuffer = function() {};
ObjectDetector.createFromOptions = function() {};
ObjectDetector.createFromModelPath = function() {};
ObjectDetector.detect = function() {};
ObjectDetector.detectForVideo = function() {};
ObjectDetector.setOptions = function() {};
ObjectDetector.close = function() {};

const PoseLandmarker = {};
PoseLandmarker.POSE_CONNECTIONS = {};
PoseLandmarker.createFromModelBuffer = function() {};
PoseLandmarker.createFromOptions = function() {};
PoseLandmarker.createFromModelPath = function() {};
PoseLandmarker.detect = function() {};
PoseLandmarker.detectForVideo = function() {};
PoseLandmarker.setOptions = function() {};
PoseLandmarker.close = function() {};
