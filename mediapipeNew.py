import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

class DetectPose:
    def __init__(self, model = r'.\models\pose_landmarker_lite.task', mode = 'vid', min_pose_detection_confidence = 0.5):
        self.model = model  
        self.mode = mode   
        self.min_pose_detection_confidence = min_pose_detection_confidence   
        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image
    
    def model_def(self):
    
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path = self.model),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence = self.min_pose_detection_confidence)
        
        return options
    
    def process(self, video_path, output_folder, output_vid_name):
        
        options = self.model_def()
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        with PoseLandmarker.create_from_options(options) as landmarker:
            video = cv2.VideoCapture(video_path)
            fps_original = video.get(cv2.CAP_PROP_FPS)
            # used to record the time when we processed last frame
            prev_frame_time = 0 
            # used to record the time at which we processed current frame
            new_frame_time = 0
            i = 0
            os.makedirs(f'{output_folder}\\temp', exist_ok = True)          
            while video.isOpened():
                frame_exists, curr_frame = video.read()
                if frame_exists:
            
                    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=curr_frame)
                    # time when we finish processing for this frame
                    new_frame_time = time.time()
  
                    fps = str(int(1/(new_frame_time-prev_frame_time)))
                    print("\nProcessing time per frame:", new_frame_time-prev_frame_time)
                    print("Fps: ",fps, "\n")
                    prev_frame_time = new_frame_time
 
                    pose_landmarker_result = landmarker.detect_for_video(mp_image, mp.Timestamp.from_seconds(time.time()).value)
                    print(pose_landmarker_result)
            
                    annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
                    #cv2.imshow('img', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f'{output_folder}\\temp\\{str(i).zfill(5)}.jpg', annotated_image)    
                    i += 1     
                else:
                    if(self.mode == 'vid'):
                        break
                    else:
                        continue
            
            video.release()  
            print("Frames generated successfully in temp folder.")
            convert_frames_to_video(f'{output_folder}\\temp', f'{output_folder}\\{output_vid_name}', fps_original)
            
class DetectHands:
    def __init__(self, model = r'.\models\hand_landmarker.task', mode = 'vid', num_hands = 2, min_hand_detection_confidence = 0.5):
        self.mode = mode
        self.model = model
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        
        MARGIN = 10  # pixels
        FONT_SIZE = 3
        FONT_THICKNESS = 5
        HANDEDNESS_TEXT_COLOR = (100, 210, 70) # vibrant green

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        #  Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
    
    def model_def(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model),
            running_mode=VisionRunningMode.VIDEO,
            num_hands = self.num_hands,
            min_hand_detection_confidence = self.min_hand_detection_confidence)
        
        return options
        
    def process(self, video_path, output_folder, output_vid_name):
        HandLandmarker = mp.tasks.vision.HandLandmarker
        options = self.model_def()
        
        with HandLandmarker.create_from_options(options) as landmarker:
            # Use OpenCV’s VideoCapture to load the input video.
            video = cv2.VideoCapture(video_path)
            # used to record the time when we processed last frame
            prev_frame_time = 0
  
            # used to record the time at which we processed current frame
            new_frame_time = 0
            # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
            fps_original = video.get(cv2.CAP_PROP_FPS)
            i = 0
            os.makedirs(f'{output_folder}\\temp', exist_ok = True) 
            # Loop through each frame in the video using VideoCapture#read()
            while video.isOpened():
                frame_exists, curr_frame = video.read()
                if frame_exists:
                    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = curr_frame)
    
                    # time when we finish processing for this frame
                    new_frame_time = time.time()
  
                    # Calculating the fps
                    fps = str(int(1/(new_frame_time-prev_frame_time)))
                    print("\nProcessing time per frame:", new_frame_time-prev_frame_time)
                    print("Fps: ", fps, "\n")
                    prev_frame_time = new_frame_time
                    
                    hand_landmarker_result = landmarker.detect_for_video(mp_image, mp.Timestamp.from_seconds(time.time()).value)
                    print("hand_landmarker_result: ", hand_landmarker_result)
                    annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
                    #cv2.imshow('img', annotated_image)
                    cv2.imwrite(f'{output_folder}\\temp\\{str(i).zfill(5)}.jpg', annotated_image)   
                    i += 1    
                    
                else:
                    if(self.mode == 'vid'):
                        break
                    else:
                        continue

            video.release()
            print("Frames generated successfully in temp folder.")
            convert_frames_to_video(f'{output_folder}\\temp', f'{output_folder}\\{output_vid_name}', fps_original)
            
def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = os.listdir(pathIn)
    
    #for sorting the file names properly
    files.sort()
    
    for i in range(len(files)):
        filename=pathIn + '\\'+files[i]
        #reading each file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), int(fps), size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
        
    print("Video saved successfully.")    
    out.release()
            
