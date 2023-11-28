import mediapipeNew as newMp
import mediapipeOld as oldMp


if __name__ == '__main__': 
        
    '''
    while instantiating the class for new models, you can change the model complexity, 
    model files are available in 'models' folder and can be downloaded from
    
    you can also check the meaning of confidence values here
    
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    

    select mode = 'vid' if you want to run the model on input video and select
    mode = 'cam' for live stream mode
    
    give video_path but if using cam mode, give 0
    '''

    object1 = newMp.DetectPose(model = r'.\models\pose_landmarker_lite.task', mode = 'vid', min_pose_detection_confidence = 0.5)
    object2 = newMp.DetectHands(model = r'.\models\hand_landmarker.task', mode = 'cam', num_hands=2, min_hand_detection_confidence = 0.5)
    
    '''
    while instantiating the class for prev models:
        you can select model complexity using model argument
        for pose: 2(heavy), 1(full), 0(lite)
        for hand: 1(full), 0(lite)
        
        mode again: vid/cam
    '''

    object3 = oldMp.DetectPose(model = 0, mode= 'vid', min_detection_confidence=0.5)
    object4 = oldMp.DetectHands(model = 0, mode= 'vid', num_hands=2, min_detection_confidence=0.5)

    '''
    in process function, every argument is necessary, in objects, default values are there
    '''
    object1.process(video_path = r".\\default_input_pose.mp4", output_folder= ".\\output", output_vid_name = 'result1.avi')
    
    #object2.process(video_path = 0, output_folder= ".\\output", output_vid_name = 'result1.avi')
    
    #object3.process(video_path = 0, output_folder= ".\\output", output_vid_name = 'result1.avi')
    
    #object4.process(video_path = r".\\default_input_hand.mp4", output_folder= ".\\output", output_vid_name = 'result1.avi')