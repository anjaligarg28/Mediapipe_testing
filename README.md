# Mediapipe_testing

MediaPipe: works only for single person
It is a tool to detect hand, body, face movement by predicting the landmarks.

There are ‘solutions’ and ‘tasks’ for pose/hand detection in mediapipe
https://github.com/google/mediapipe/tree/master

Mediapipe ended support for ‘solutions’ on march 1, 2023. Newest versions for detection are
released as ‘tasks’. To run the newest models we have to download .task files from overview
section of particular models in https://developers.google.com/mediapipe/solutions/guide

‘Solutions’ are giving better results in hand detection than tasks, but solutions are slower as they
are not utilizing gpu.

Codes for solutions implementation:
https://github.com/google/mediapipe/tree/master/docs/solutions

In solutions:
Hand detection: Model complexity levels: 0,1
https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
Pose detection: Model complexity levels: 0,1,2
https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

In tasks/new solutions:
Hand detection: Model: full
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
Pose detection: mode: heavy, lite, full
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
