import cv2
import mediapipe as mp
import time
import queue
import numpy as np
import pyautogui
import mouse

from scipy import signal
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mouse._mouse_event import RIGHT

# Initialize global variables
global left_click_status
global left_click_time
global right_click_status
global right_click_time
left_click_status = False
right_click_status = False
detect = False

#Initialize variables
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#initialize arrays for averaging
global rix_array
global riy_array
rix_array = np.empty(1, dtype=float)
riy_array = np.empty(1, dtype=float)

#initialize the input buffer
detect_buffer = queue.LifoQueue()

# Get the dims of the computer screen
dims = pyautogui.size()

#model_path
model_path = r'./Assets/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('hand landmarker result: {}\n\n'.format(result.hand_landmarks))
    #if a hand is detected, output the result
    if len(result.hand_landmarks)>0:
       detect_buffer.put([result.hand_landmarks, result.handedness])
       analyze_data([result.hand_landmarks, result.handedness])
    #print('hand landmarker image: {}\n\n'.format(output_image))
    #print('hand landmarker timestamp: {}\n\n'.format(timestamp_ms))

#Code to draw landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result[0]
  handedness_list = detection_result[1]
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
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

def analyze_data(data):
  # Parse the data
  hand_landmarks_list = data[0]
  handedness = data[1]
  # Set up a variable to track the number of hands in the image
  hand_count = 0
  for hand in handedness:
    info = hand[0]
    if info.category_name == 'Right':
      right_hand_landmarks = hand_landmarks_list[hand_count]
      right_controls(right_hand_landmarks)
    
    else:
      left_hand_landmarks = hand_landmarks_list[hand_count]
      left_controls(left_hand_landmarks)

    hand_count+=1   
            
def right_controls(right_hand_landmarks):
  # Global vars
  global rix_array
  global riy_array

  #initialize local variable
  sensitivity = 2
  window_size = 12
  poly_order = int(window_size/2)

  # Extract the right index finger tip from the hand landmarks
  right_index_landmark = right_hand_landmarks[8]

  # Get the x and y coordinates of the index finger tip
  right_index_x_coordinates = (1 - right_index_landmark.x)*sensitivity
  right_index_y_coordinates = (right_index_landmark.y)*sensitivity

  rix_array = np.append(rix_array, [right_index_x_coordinates])
  riy_array = np.append(riy_array, [right_index_y_coordinates])

  if len(rix_array) == window_size*sensitivity:
    #print('{}, {}\n\n'.format(rix_array, riy_array))
    smoothed_rix = signal.savgol_filter(rix_array, window_size, poly_order)
    smoothed_riy = signal.savgol_filter(riy_array, window_size, poly_order)
    rix_avg = np.mean(smoothed_rix[int(-window_size/2):-1])
    riy_avg = np.mean(smoothed_riy[int(-window_size/2):-1])
    if rix_avg > sensitivity or riy_avg > sensitivity: 
      rix_array = np.delete(rix_array, 0, 0)
      riy_array = np.delete(riy_array, 0, 0)
    else:
      # Smooth the datapoints
      right_index_x_coordinates = rix_avg
      rix_array = np.delete(rix_array, 0, 0)
      right_index_y_coordinates = riy_avg
      riy_array = np.delete(riy_array, 0, 0)
    
  #print('averaging', right_index_x_coordinates,right_index_y_coordinates)

  # Normalize the finger coordinates to the screen coordinates
  right_index_x_coordinates = int(right_index_x_coordinates * dims[0]-(int(dims[0]/sensitivity)))
  right_index_y_coordinates = int(right_index_y_coordinates * dims[1]-(int(dims[1]/sensitivity)))

  #print(right_index_x_coordinates, right_index_y_coordinates)

  # Move the mouse
  mouse.move(right_index_x_coordinates, right_index_y_coordinates, duration=duration/2)
        
def left_controls(left_hand_landmarks):
  # Initialize global variables to track clicks
  global left_click_status
  global left_click_time
  global right_click_status
  global right_click_time
  
  #Set the distance, in pixels, that the click fingers must be separated by to initiate the click
  finger_dist = 50

  # Extract the left index finger tip from the hand landmarks
  left_index_landmark = left_hand_landmarks[8]
  left_ring_landmark = left_hand_landmarks[16]
  left_thumb_landmark = left_hand_landmarks[4]

  # Get the x and y coordinates of the index finger tip
  left_index_x_coordinates = left_index_landmark.x
  left_index_y_coordinates = left_index_landmark.y

  # Get the x and y coordinates of the index finger tip
  left_ring_x_coordinates = left_ring_landmark.x
  left_ring_y_coordinates = left_ring_landmark.y

  # Get the x and y coordinates of the thumb tip
  left_thumb_x_coordinates = left_thumb_landmark.x
  left_thumb_y_coordinates = left_thumb_landmark.y

  # Normalize the finger coordinates to the screen coordinates
  left_index_x_coordinates = left_index_x_coordinates * dims[0]
  left_index_y_coordinates = left_index_y_coordinates * dims[1]

  # Normalize the finger coordinates to the screen coordinates
  left_ring_x_coordinates = left_ring_x_coordinates * dims[0]
  left_ring_y_coordinates = left_ring_y_coordinates * dims[1]

  # Normalize the finger coordinates to the screen coordinates
  left_thumb_x_coordinates = left_thumb_x_coordinates * dims[0]
  left_thumb_y_coordinates = left_thumb_y_coordinates * dims[1]

  # Compute difference in distance between the index finger and thumb
  delta_it_x = abs(left_index_x_coordinates - left_thumb_x_coordinates)
  delta_it_y = abs(left_index_y_coordinates - left_thumb_y_coordinates)

  # Compute difference in distance between the ring finger and thumb
  delta_rt_x = abs(left_ring_x_coordinates - left_thumb_x_coordinates)
  delta_rt_y = abs(left_ring_y_coordinates - left_thumb_y_coordinates)
  
  if delta_it_x < finger_dist and delta_it_y < finger_dist and not left_click_status:
      mouse.press()
      left_click_status = True
      left_click_time = time.time()
      print("Left Mouse Pressed", left_click_time)
  elif delta_it_x > finger_dist and delta_it_y > finger_dist and left_click_status and time.time()-left_click_time > 0.25:
      mouse.release()
      left_click_status = False
      print("Left Mouse Released")

  if delta_rt_x < finger_dist and delta_rt_y < finger_dist and not right_click_status:
      mouse.press(RIGHT)
      right_click_status = True
      right_click_time = time.time()
      print("Right Mouse Pressed", right_click_time)
  elif delta_rt_x > finger_dist and delta_rt_y > finger_dist and right_click_status and time.time()-right_click_time > 0.25:
      mouse.release(RIGHT)
      right_click_status = False
      print("Right Mouse Released")
   
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result)

if __name__ == "__main__":
  cam = cv2.VideoCapture(0)
  print(cam)

  with HandLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:
        start = time.time()
        ret, frame = cam.read()

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # The hand landmarker must be created with the live stream mode.
        landmarker.detect_async(mp_image, timestamp)
      
        if detect_buffer.qsize():
          # STEP 5: Process the classification result. In this case, visualize it.
          data = detect_buffer.get()
          frame = draw_landmarks_on_image(frame, data)

        cv2.imshow('Video Feed', frame)
          
        timestamp+=1

        cv2.waitKey(1)

        end = time.time()
        global duration
        duration = end-start
        frame_rate = 1/(end-start)
        print(frame_rate)


