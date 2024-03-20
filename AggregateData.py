import numpy as np
import glob
import os
import math
import mediapipe as mp
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import plotly.express as px
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mpl_toolkits.mplot3d import Axes3D
from npy_append_array import NpyAppendArray

#Global vars
global imgCount
global datalist
global label_list
datalist = []
label_list = []
imgCount = 0

#directory
directory = os.getcwd()

#folder path and extension
folder = os.path.join(directory, r'ExpressionDetection\NormData')
extension = '/*.npy'

spatial_filename = os.path.join(folder, 'AggregatedSpatialData.npy')
label_filename = os.path.join(folder, 'AggregatedLabelData.npy')

#model path
model_path = r'./Assets/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

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

def get_image_data():
    folder_path = os.path.join(os.getcwd(), r'ExpressionDetection\ImageDatasets\train')
    folders = os.listdir(folder_path)

    #Set the options for the medaipipe face landmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_tracking_confidence=0.8)

    for folder in folders:
        files = os.listdir(os.path.join(folder_path, folder))
        with FaceLandmarker.create_from_options(options) as landmarker:
            for file in files:
                encoded_label = np.zeros(8)
                labels = ['Anger','Contempt','Disgust',
                'Fear','Happiness','Neutral',
                'Sadness','Surprise']
                index = labels.index(folder)
                encoded_label[int(index)] = 1
                encoded_label = encoded_label.reshape(1,8)

            # Load the input image from an image file.
                img = os.path.join(os.path.join(folder_path, folder), file)
                img = cv2.imread(img, 0)
                img = cv2.resize(img, (640,480))
                bgr_mp_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                rgb_mp_image = cv2.cvtColor(bgr_mp_image, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_mp_image)

                # Perform face landmarking on the provided single image.
                # The face landmarker must be created with the image mode.
                face_landmarker_result = landmarker.detect(mp_image)

                data = face_landmarker_result.face_landmarks[0]
                if not len(data):
                    print("Could not find a face")
                    os.remove(os.path.join(os.path.join(folder_path, folder), file))
                else:
                    point = []
                    for pt in data:
                        point.append([pt.x, -pt.y, -pt.z])
                    point= np.array(point).reshape(478,3)
                    x = point[:, 0]
                    y = point[:, 1]
                    z = point[:, 2]
                    point_cloud = np.array((x, y, z)).T
                    norm_pointcloud = Normalize()(point_cloud)
                    rot_pointcloud = RandRotation_z()(norm_pointcloud)
                
                global imgCount
                global datalist
                global label_list
                global aggdata
                global agglabel
            
                #After collecting enough data
                if imgCount % 100 == 0:
                    if imgCount:
                        datalist = np.array(datalist).reshape(100*478, 3)
                        label_list = np.array(label_list).reshape(100, 8)
                        aggdata = np.append(aggdata, datalist, axis = 0)
                        agglabel = np.append(agglabel, label_list, axis = 0)
                        print(aggdata.shape, agglabel.shape)
                    else:
                        aggdata = rot_pointcloud
                        agglabel = encoded_label
                    datalist = []
                    datalist.append(rot_pointcloud)
                    label_list = []
                    label_list.append(encoded_label)
                else:
                    datalist.append(rot_pointcloud)
                    label_list.append(encoded_label)
                imgCount +=1

    return aggdata, agglabel

def main():

    count1=0
    for file in glob.glob(folder + extension):
        if file == spatial_filename or file == label_filename:
            continue
        dataset = np.load(file)
        count2 = 0
        for sample in dataset:
            #One-hot encode the label
            label = np.zeros(8)
            label[int(sample[:,:,0][0][0])]=1
            label=label.reshape(1, 8)
            
            x = sample[:, :, 1][0]
            y = -sample[:, :, 2][0]
            z = sample[:, :, 3][0]

            point_cloud = np.array((x, y, z)).T

            norm_pointcloud = Normalize()(point_cloud)
            rot_pointcloud = RandRotation_z()(norm_pointcloud)
            #print(rot_pointcloud)
            
            #Continuously add to the dataset
            if not count2:
                labels = label
                data = rot_pointcloud
                pcshow(*rot_pointcloud.T)
            else:
                labels = np.append(labels, label, axis = 0)
                data = np.append(data, rot_pointcloud, axis = 0)
                #print(labels.shape, data.shape)
            count2=1

        #Continuously add to the dataset
        if not count1:
            label_arr = labels
            data_arr = data
        else:
            label_arr = np.append(label_arr, labels, axis = 0)
            data_arr = np.append(data_arr, data, axis = 0)
            print(label_arr.shape, data_arr.shape)
        count1=1

    label_arr = np.append(label_arr, labels, axis = 0)
    data_arr = np.append(data_arr, data, axis = 0)

    data, labels = get_image_data()

    label_arr = np.append(label_arr, labels, axis = 0)
    data_arr = np.append(data_arr, data, axis = 0)

    print(data_arr.shape, label_arr.shape)

    np.save(spatial_filename, data_arr)
    np.save(label_filename, label_arr)



if __name__ == '__main__':
    main()