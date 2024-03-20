import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import os
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

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


#Code to draw landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result
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
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())   

  return annotated_image

#Convert the model output from an index value to a label
def convertOutput(output, confidence):
  values = ['Anger','Contempt','Disgust',
            'Fear','Happiness','Neutral',
            'Sadness','Surprise']
  print('Output: {}       Prediction:     {}      Confidence:  {}'.format(output, values[output], confidence*100))
#Predict the expression in the image
def pred_exp(data, model, device):
  coords = []

  if not len(data[0]):
     print("Could not detect a face")
     return

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

#Main function
def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")

    modelPath = os.path.join(os.getcwd(), r'ExpressionDetection\Models\pointnet5.pt')
    model = PointNet().to(device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    dir = os.path.join(os.getcwd(), r'ExpressionDetection\ImageDatasets\Aggregated')
    dir_list = os.listdir(dir)

    #Set the options for the medaipipe face landmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    with FaceLandmarker.create_from_options(options) as landmarker:
        for file in dir_list:
           # Load the input image from an image file.
            file = os.path.join(dir, file)
            file = cv2.imread(file, 0)
            file = cv2.resize(file, (640,480))
            bgr_mp_image = cv2.cvtColor(file, cv2.COLOR_GRAY2BGR)
            rgb_mp_image = cv2.cvtColor(bgr_mp_image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_mp_image)

            # Perform face landmarking on the provided single image.
            # The face landmarker must be created with the image mode.
            face_landmarker_result = landmarker.detect(mp_image)
            annotated = draw_landmarks_on_image(bgr_mp_image, face_landmarker_result.face_landmarks)

            pred_exp([face_landmarker_result.face_landmarks], model, device)
            cv2.imshow('Base Image',file)
            cv2.imshow('Model View',annotated)
            cv2.waitKey(0)

if __name__ == "__main__":
  main()
  
