import cv2
import mediapipe as mp
import os
import time
import queue
import numpy as np
import pandas as pd
import random

from sklearn import preprocessing
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from npy_append_array import NpyAppendArray

"""
Expressions to capture:

Neutral
Anger
Contempt
Disgust
Fear
Happiness
Sadness
Surprise

"""

#Global vars
global frameCount
global datalist
datalist = []
frameCount = 0
directory = os.getcwd()

#initialize the input buffer
detect_buffer = queue.LifoQueue()

#model path
model_path = r'./Assets/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #if a hand is detected, output the result
    if len(result.face_landmarks)>0:
       detect_buffer.put([result.face_landmarks])
       collect_data([result.face_landmarks])

#Code to draw landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result[0]
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    """
    #Uncomment for face mesh
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style()) 

    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())   

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())
    """
  return annotated_image

def collect_data(data):

  expression = "Anger"

  #data path
  filename = os.path.join(directory, rf'ExpressionDetection\NormData\{expression}.npy')

  coords = []

  #
  for lm in data[0][0]:
    coords.append([lm.x, lm.y, -lm.z])
  
  coords = np.array(coords)

  coords = coords.reshape(478, 3)

  #get the x, y, z coords of landmark 0
  index_0 = coords[0]

  #subtract index zero from all the data array
  coords = np.subtract(coords, index_0)

  #Encode the current expression
  labels = ['Anger','Contempt','Disgust',
              'Fear','Happiness','Neutral',
              'Sadness','Surprise']
  index = labels.index(expression)
  label_array = np.full(coords.shape[0], index).reshape(478,1)

  #add the labels to the dataset
  dataset = np.append(label_array, coords, axis=1).astype('float32')
  dataset = dataset.reshape(1, 478, 4)

  #get global vars
  global frameCount
  global datalist

  #After collecting enough data
  if frameCount % 100 == 0:
    if frameCount:
      if not os.path.exists(filename):
        datalist = np.array(datalist)
        np.save(filename, datalist)
      else:
        with NpyAppendArray(filename) as npaa:
          datalist = np.array(datalist)
          npaa.append(datalist)
    datalist = []
    datalist.append(dataset)
  else:
    datalist.append(dataset)

  frameCount +=1
  print(frameCount, len(datalist))


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

if __name__ == "__main__":
  cam = cv2.VideoCapture(0)

  with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0
    
    while True:
        start = time.time()
        ret, frame = cam.read()
        h, w, depth = frame.shape

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # The hand landmarker must be created with the live stream mode.
        landmarker.detect_async(mp_image, timestamp)
      
        if detect_buffer.qsize()>0:
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
        #print(frame_rate)


