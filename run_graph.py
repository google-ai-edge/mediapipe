import time

import streamlit as st


import cameravtuber2
import cv2

runner = cameravtuber2.GraphRunner("mediapipe/graphs/hand_tracking/hand_face_detection_no_gating.pbtxt")

@st.cache(allow_output_mutation=True)
def get_frames():
    cap = cv2.VideoCapture("hand_motion.mp4")

    frames = []
    
    while(True):
        ret, frame = cap.read()
        if frame is None:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(gray)

    return frames

frames = get_frames()
frame_idx = st.slider('frame idx', 0, len(frames))
# st.image(frames[frame_idx])
# st.write(frames[frame_idx].shape)
frame = frames[frame_idx]
start = time.time()
new_frame = runner.process_frame(frame)
end = time.time()
st.write(end - start)
st.write(new_frame.shape)

new_frame = new_frame.reshape((360,640,4))
st.image(new_frame)


# cameravtuber2.GraphRunner("mediapipe/graphs/hand_tracking/hand_face_tracking_desktop.pbtxt")
