import cv2
import mediapipe as mp
import time
import os

class DetectHands:
    
    def __init__(self, model = 0, mode = 'vid', num_hands = 2, min_detection_confidence = 0.5):
        self.model = model
        self.mode = mode
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
    
    def process(self, video_path, output_folder, output_vid_name):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        cap = cv2.VideoCapture(video_path)
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        i=0
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        os.makedirs(f'{output_folder}\\temp', exist_ok = True)
        with mp_hands.Hands(model_complexity = self.model, max_num_hands = self.num_hands, min_detection_confidence = self.min_detection_confidence, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    if(self.mode == 'vid'):
                        break
                    else:
                        continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                print(results)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                # time when we finish processing for this frame
                new_frame_time = time.time()
  
                # Calculating the fps
                fps = str(int(1/(new_frame_time-prev_frame_time)))
                print("\nProcessing time per frame: ", new_frame_time-prev_frame_time)
                print("Fps: ", fps, "\n")
                prev_frame_time = new_frame_time
                #cv2.imshow('MediaPipe Hands', image)
                cv2.imwrite(f'{output_folder}\\temp\\{str(i).zfill(5)}.jpg', image)
                i += 1
                
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        print("Frames generated successfully in temp folder.")
        cap.release()
        convert_frames_to_video(f'{output_folder}\\temp', f'{output_folder}\\{output_vid_name}', fps_original)
        

class DetectPose:
    def __init__(self, model = 0, mode = 'vid', min_detection_confidence = 0.5):
        self.model = model
        self.mode = mode
        self.min_detection_confidence = min_detection_confidence
        
    def process(self, video_path, output_folder, output_vid_name):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        
        cap = cv2.VideoCapture(video_path)
        i=0
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        os.makedirs(f'{output_folder}\\temp', exist_ok = True)
        with mp_pose.Pose(
            model_complexity = 1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    if(self.mode == 'vid'):
                        break
                    else:
                        continue
    
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                print(results)
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
                # time when we finish processing for this frame
                new_frame_time = time.time()
  
                # Calculating the fps
  
                fps = str(int(1/(new_frame_time-prev_frame_time)))
                print("\n", new_frame_time-prev_frame_time)
                print(fps, "\n\n\n")
                prev_frame_time = new_frame_time
                # Flip the image horizontally for a selfie-view display.
                #cv2.imshow('MediaPipe Pose', image)
                cv2.imwrite(f'{output_folder}\\temp\\{str(i).zfill(5)}.jpg', image)    
                i += 1 
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        print("Frames generated successfully in temp folder.")
        cap.release()
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
            