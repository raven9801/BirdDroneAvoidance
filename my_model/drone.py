# Import libraries
import cv2
from ultralytics import YOLO
from djitellopy import Tello

import pygame
import time

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("scifiAlarm.wav")  # Make sure this file exists

#45000 good/a couple feet
AREA_THRESHOLD = 30000  # Adjust as needed
COOLDOWN_SECONDS = 3    # Wait time before sound can play again
last_played_time = 0    # Initial time
soundAvoid = True
landAvoid = False


# Initializing the Tello drone
tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()


def getKeyboardInput(key):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 80 
    liftSpeed = 80
    moveSpeed = 85
    rotationSpeed = 100

    # Movement controls
    if key == ord('a'): lr = -speed
    elif key == ord('d'): lr = speed

    if key == ord('w'): fb = moveSpeed
    elif key == ord('s'): fb = -moveSpeed

    if key == ord('r'): ud = liftSpeed
    elif key == ord('f'): ud = -liftSpeed

    if key == ord('j'): yv = -rotationSpeed
    elif key == ord('l'): yv = rotationSpeed

    # Takeoff & Landing
    if key == ord('t') and not tello.is_flying: tello.takeoff()
    elif key == ord('q') and tello.is_flying: tello.land()

    return [lr, fb, ud, yv]


# State the path/location of where the video output file will be saved
# Insert your own path by copy and pasting the file path address from File Explorer
video_path = 'D:/edge/output/movetest1.mp4'   # add video name at the end!

# State what video codec to use (mp4, h.264, av1, etc.)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# video_out = (video_path, fourcc, FPS, (Frame Width, Frame Height), isColor = T/F)
video_out = cv2.VideoWriter(video_path, fourcc, 20, (960,720), isColor=True)

# Load the pretrained YOLOv11 model
# Add the correct path the model location
model = YOLO(r'D:\edge\2classmodelnano40epochdropout02\my_model\train\weights\best.pt', task='detect')

# Get the 'BackgroundFrameRead' object that HOLDS the latest captured frame from the Tello
frame_read = tello.get_frame_read(with_queue=False, max_queue_len=0)

while True:
    # Access the frame from the video using '.frame', so we can USE it and display it
    frame = frame_read.frame

    if frame is not None:   #in other words, if there IS a frame...
        # Run YOLOv11 Object Detection on the frame
        # 'results' saves information about the detected objects

        #confidence of 0.75 worked well for my room, but 0.25 was generally better outside of it,
        #however it increased the false postive rate
        results = model(frame,conf=0.25, imgsz=320)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Convert the annotated frame colors from BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame,cv2.COLOR_BGR2RGB)

        # Write the annotated frame to the video output
        video_out.write(annotated_frame)

        # Display the annotated frame in a window
        cv2.imshow("YOLOv8 Tello Drone Tracking", annotated_frame)


        # controls
        key = cv2.waitKey(1) & 0xFF

        
        keyValues = getKeyboardInput(key)
        tello.send_rc_control(*keyValues)
        

        # get box areas

        current_time = time.time()
        # Extract bounding boxes and compute area
        boxes = results[0].boxes  # YOLOv11 format
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
            area = (x2 - x1) * (y2 - y1)
            print(f"Detection area: {area:.2f} pixels")

            #if sound avoid is active
            if(soundAvoid):
                if area > AREA_THRESHOLD and (current_time - last_played_time) > COOLDOWN_SECONDS:
                    alert_sound.play()
                    last_played_time = current_time

            #if land avoid is active
            if(landAvoid):
                if area > AREA_THRESHOLD:
                    if tello.is_flying: tello.land()

        # Break the loop if 'x' is pressed
        if key == ord("x"):
            break

    else:
        # Indicate no frame was received
        print("No frame received")

video_out.release()     # Closes the video output file
tello.end()             # Ends the tello object (lands tello, turns off stream, stops BackgroundFrameRead)
cv2.destroyAllWindows() # Destroys any open windows, such as the streaming window.
