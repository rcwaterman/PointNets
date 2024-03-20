import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import os
import time
import queue
import numpy as np

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from AggregateData import Normalize
from PointNet import PointNet


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

  return annotated_image
#Convert the model output from an index value to a label
def convertOutput(output, confidence):
  values = ['Anger','Contempt','Disgust',
            'Fear','Happiness','Neutral',
            'Sadness','Surprise']
  print('Output:  {}       Prediction:  {}      Confidence:  {}'.format(output, values[output], confidence*100))
#Predict the expression in the image
def pred_exp(data, model, device):
  coords = []

  for lm in data[0][0]:
    coords.append([lm.x, -lm.y, -lm.z])

  point_cloud = np.array(coords).reshape(478, 3)
  norm_pointcloud = Normalize()(point_cloud)

  tensor = torch.tensor(norm_pointcloud.T, dtype=torch.float32).unsqueeze(0)
  tensor = tensor.to(device)
  outputs, __, __ = model(tensor)
  probabilities = torch.exp(outputs)
  confidence, predicted = torch.max(probabilities, 1)
  convertOutput(predicted.item(), confidence.item())
#Set the options for the medaipipe face landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
#Main function
def main():

  if torch.cuda.is_available():
    device = torch.device("cuda")

  modelPath = os.path.join(os.getcwd(), r'ExpressionDetection\Models\pointnet5.pt')
  model = PointNet().to(device)
  model.load_state_dict(torch.load(modelPath))
  model.eval()

  cam = cv2.VideoCapture(0)

  with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0
    
    while True:
      start = time.time()
      ret, frame = cam.read()

      frame = cv2.resize(frame, (960,720), cv2.INTER_AREA)

      # Convert the frame received from OpenCV to a MediaPipe’s Image object.
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

      # The hand landmarker must be created with the live stream mode.
      landmarker.detect_async(mp_image, timestamp)
    
      if detect_buffer.qsize()>0:
        # STEP 5: Process the classification result. In this case, visualize it.
        data = detect_buffer.get()
        pred_exp(data, model, device)
        annotated = draw_landmarks_on_image(frame, data)
        annotated = cv2.flip(annotated, 1)
        cv2.imshow('Annotated Video', annotated)
      else:
        cv2.imshow('Annotated Video', cv2.flip(frame, 1))
        print('Face not detected')

      frame = cv2.flip(frame, 1)  
      cv2.imshow('Raw Video', frame)
    
      timestamp+=1

      cv2.waitKey(1)

      end = time.time()
      global duration
      duration = end-start
      frame_rate = 1/(end-start)
      #print(frame_rate)

if __name__ == "__main__":
  main()
  
