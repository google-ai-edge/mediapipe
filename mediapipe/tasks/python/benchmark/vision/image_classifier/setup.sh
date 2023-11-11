# Install Python dependencies.
python3 -m pip install pip --upgrade
python3 -m pip install mediapipe

wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite
wget -O burger.jpg https://storage.googleapis.com/mediapipe-assets/burger.jpg
