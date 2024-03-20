import cv2
import mediapipe as mp
import time
import queue
import math
import numpy as np
import pyautogui
import mouse
import matplotlib.pyplot as plt

from scipy import signal
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize global variables
global click_status
global click_time
global left_depth_array
global right_depth_array
global top_depth_array
global bottom_depth_array
global min_x
global min_y
global max_x
global max_y
global timestamp

min_x = 0
min_y = 0
max_x = 0
max_y = 0
timestamp = 0

left_depth_array = np.empty(1, dtype=float)
right_depth_array = np.empty(1, dtype=float)
top_depth_array = np.empty(1, dtype=float)
bottom_depth_array = np.empty(1, dtype=float)
click_status = False

#initialize the input buffer
detect_buffer = queue.LifoQueue()
coord_buffer = queue.Queue()

# Store lists of the necessary landmark indices
left_iris_indices = [474, 475, 476, 477]
right_iris_indices = [469, 470, 471, 472]
left_eye_indices = [263, 249, 390, 373, 
                    374, 380, 381, 382, 
                    362, 466, 388, 387, 
                    386, 385, 384, 398]
right_eye_indices = [33, 7, 163, 144, 
                    145, 153, 154, 155, 
                    133, 246, 161, 160, 
                    159, 158, 157, 173]
face_outline = [234, 454, 10, 152, 0]

indices = left_iris_indices + right_iris_indices + left_eye_indices + right_eye_indices + face_outline

# Get the dims of the computer screen
dims = pyautogui.size()

#model_path
model_path = r'./Assets/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('face landmarker result: {}\n\n'.format(result.face_landmarks))
    #if a hand is detected, output the result
    if len(result.face_landmarks)>0:
       detect_buffer.put([result.face_landmarks])
       analyze_data([result.face_landmarks])
    #print('hand landmarker image: {}\n\n'.format(output_image))
    #print('hand landmarker timestamp: {}\n\n'.format(timestamp_ms))

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

def compute_view_angle(data):
  #define vars
  global left_depth_array
  global right_depth_array
  global top_depth_array
  global bottom_depth_array
  global min_x
  global min_y
  global max_x
  global max_y

  #constants
  window_size = 20 #smoothing function window size
  poly_order = 5 #smoothing function polynomial order of magnitude
  hfov = 66.5 #horizontal fov of webcam
  iw = 11.7 #iris width, in millimeters
  hw = 152 #average head width, in millimeters
  fw = 640 #frame width, in pixels
  fh = 480 #frame height, in pixels
  fl = 4.43 #focal length of webcam, in mm
  sw = 5.81 #camera sensor width, in mm
  sh = 3.27 #camera sensor height, in mm
  pd = 57 #pupillary distance, in mm, determined empirically ((lic_x - ric_x)*((rmmpp+lmmpp)/2)*fw). Average for adult is 54 to 74 mm

  #parse the coords of each landmark of interest
  left_iris_coords = data[0:4]
  right_iris_coords = data[4:8]
  left_eye_coords = data[8:24]
  right_eye_coords = data[24:40]
  face_coords = data[40::]

  #get the key face point values
  face_left = face_coords[0]
  face_right = face_coords[1]
  face_top = face_coords[2]
  face_bottom = face_coords[3]
  depth = face_coords[4]

  fl_xy = (face_left[0], face_left[1])
  fr_xy = (face_right[0], face_right[1])
  ft_xy = (face_top[0], face_top[1])
  fb_xy = (face_bottom[0], face_bottom[1])

  #compute the relative left-right face tilt
  midpoint_width = face_left[0]+(face_right[0]-face_left[0])/2
  midpoint_height = face_bottom[1]+(face_top[1]-face_bottom[1])/2

  #Normalize/scale the depth of each facial location to the center of the face
  center_depth = 1/((((abs(face_top[2])+abs(face_bottom[2]))/2)+
                    ((abs(face_left[2])+abs(face_right[2]))/2))/2)
 
  left_depth = (1/abs(face_left[2]))
  right_depth = (1/abs(face_right[2]))
  top_depth = (1/abs(face_top[2]))
  bottom_depth = (1/abs(face_bottom[2]))

  left_depth_array = np.append(left_depth_array, [left_depth])
  right_depth_array = np.append(right_depth_array, [right_depth])
  top_depth_array = np.append(top_depth_array, [top_depth])
  bottom_depth_array = np.append(bottom_depth_array, [bottom_depth])

  #Smooth the data
  if len(left_depth_array) == window_size:
    smoothed_left_depth = signal.savgol_filter(left_depth_array, window_size, poly_order)
    smoothed_right_depth = signal.savgol_filter(right_depth_array, window_size, poly_order)
    smoothed_top_depth = signal.savgol_filter(top_depth_array, window_size, poly_order)
    smoothed_bottom_depth = signal.savgol_filter(bottom_depth_array, window_size, poly_order)
    left_depth_avg = np.mean(smoothed_left_depth[int(-window_size/2):-1])
    right_depth_avg = np.mean(smoothed_right_depth[int(-window_size/2):-1])
    top_depth_avg = np.mean(smoothed_top_depth[int(-window_size/2):-1])
    bottom_depth_avg = np.mean(smoothed_bottom_depth[int(-window_size/2):-1])
    left_depth_array = np.delete(left_depth_array, 0, 0)
    right_depth_array = np.delete(right_depth_array, 0, 0)
    top_depth_array = np.delete(top_depth_array, 0, 0)
    bottom_depth_array = np.delete(bottom_depth_array, 0, 0)
  else:
    left_depth_avg = left_depth
    right_depth_avg = right_depth
    top_depth_avg = top_depth
    bottom_depth_avg = bottom_depth

  #Compute the angle of the face relative to the camera
  LR_midpoint_depth = right_depth_avg+(left_depth_avg-right_depth_avg)/2
  LR_depth_dist = left_depth_avg-LR_midpoint_depth
  LR_theta = 90-math.degrees(math.atan(LR_depth_dist/midpoint_width))
  TB_midpoint_depth = top_depth_avg+(top_depth_avg-bottom_depth_avg)/2
  TB_depth_dist = bottom_depth_avg-TB_midpoint_depth
  TB_theta = 90-math.degrees(math.atan(TB_depth_dist/midpoint_height))
  midpoint_coords = (midpoint_width, midpoint_height)
  tilt_theta = (math.degrees(math.atan((face_top[1]-midpoint_height)/
                                       (face_top[0]-midpoint_width)))) 
  if tilt_theta > 0:
    tilt_theta = 90-tilt_theta
  elif tilt_theta < 0:
    tilt_theta = -(90+tilt_theta)

  #find the key points of the left iris
  lil_index = np.argmin(left_iris_coords[::,0])
  lir_index = np.argmax(left_iris_coords[::,0])
  lit_index = np.argmax(left_iris_coords[::,1])
  lib_index = np.argmin(left_iris_coords[::,1])

  #get the x, y, and z coords of each left iris key point
  lil = left_iris_coords[lil_index]
  lir = left_iris_coords[lir_index]
  lit = left_iris_coords[lit_index]
  lib = left_iris_coords[lib_index]

  #parse the different coordinates into variables
  lil_x, lil_y, lil_z = lil
  lir_x, lir_y, lir_z = lir
  lit_x, lit_y, lit_z = lit
  lib_x, lib_y, lib_z = lib
  liw = (lir_x - lil_x)*1.25 #scale factor, it was empirically determined that lih = 0.75*liw
  lih = lit_y - lib_y
  lidz_x = lil_z - lir_z

  #find the center of the left iris, i.e. the pupil
  lic_x = lil_x + (lir_x - lil_x)/2
  lic_y = lit_y + (lib_y - lit_y)/2
  left_pupil = (lic_x, lic_y)

  #find the key points of the left eye
  lel = np.min(left_eye_coords[::,0])
  ler = np.max(left_eye_coords[::,0])
  let = np.max(left_eye_coords[::,1])
  leb = np.min(left_eye_coords[::,1])

  #find the center of the left eye
  lec_x = lel + (ler - lel)/2
  lec_y = let + (leb - let)/2
  lec = (lec_x, lec_y)

  #this accounts for any angle of the iris
  if liw > lih:
    lid = ((fl*iw*fh)/((liw*fh)*sh))
    lmmpp = iw/(liw*fw)
    lpw = fw*lmmpp*liw #the perceived width, in mm
    lph = lih*lmmpp*fw #the perceived height, in mm
    theta_lix = math.degrees(math.acos(lph/iw))
  elif liw < lih:
    lid = ((fl*iw*fh)/((lih*fh)*sh))
    lmmpp = iw/(lih*fh)
    lph = fh*lmmpp*(lih) #the perceived height, in mm
    lpw = liw*lmmpp*fh #the perceived width, in mm
    theta_lix = math.degrees(math.acos(lpw/iw))
  elif liw == lih:
    lid = (((fl*iw*fw)/((liw*fw)*sw))+
           ((fl*iw*fh)/((lih*fh)*sh)))*0.5 #average the height and width dist measurments
    lmmpp = iw/(((lih*fh)+(liw*fw))*0.5) #average the mmpp calculation
    lph = iw #the perceived height, in mm. This is equal to the actual height
    lpw = iw #the perceived width, in mm. This is equal to the actual height
    theta_lix = 0

  theta_lix = 90 - (90 - theta_lix)
  
  #find the key points of the right iris
  ril_index = np.argmin(right_iris_coords[::,0])
  rir_index = np.argmax(right_iris_coords[::,0])
  rit_index = np.argmax(right_iris_coords[::,1])
  rib_index = np.argmin(right_iris_coords[::,1])

  #get the x, y, and z coords of each right iris key point
  ril = right_iris_coords[ril_index]
  rir = right_iris_coords[rir_index]
  rit = right_iris_coords[rit_index]
  rib = right_iris_coords[rib_index]

  #parse the different coordinates into variables
  ril_x, ril_y, ril_z = ril
  rir_x, rir_y, rir_z = rir
  rit_x, rit_y, rit_z = rit
  rib_x, rib_y, rib_z = rib
  riw = (rir_x - ril_x)*1.25 #scale factor, it was empirically determined that lih = 0.75*liw
  rih = rit_y - rib_y
  ridz_x = ril_z - rir_z

  #find the center of the right iris, i.e. the pupil
  ric_x = ril_x + (rir_x - ril_x)/2
  ric_y = rit_y + (rib_y - rit_y)/2
  right_pupil = (ric_x, ric_y)

  #find the key points of the right eye
  rel = np.min(right_eye_coords[::,0])
  rer = np.max(right_eye_coords[::,0])
  ret = np.max(right_eye_coords[::,1])
  reb = np.min(right_eye_coords[::,1])

  #find the center of the right eye
  rec_x = rel + (rer - rel)/2
  rec_y = ret + (reb - ret)/2
  rec = (rec_x, rec_y)

  #this accounts for any angle of the iris
  if riw > rih:
    rid = ((fl*iw*fh)/((riw*fh)*sh))
    rmmpp = iw/(riw*fw)
    rpw = fw*rmmpp*riw #the perceived width, in mm
    rph = rih*rmmpp*fw #the perceived height, in mm
    theta_rix = math.degrees(math.acos(rph/iw))
  elif riw < rih:
    rid = ((fl*iw*fh)/((rih*fh)*sh))
    rmmpp = iw/(rih*fh)
    rph = fh*rmmpp*(rih) #the perceived height, in mm
    rpw = riw*rmmpp*fh #the perceived width, in mm
    theta_rix = math.degrees(math.acos(rpw/iw))
  elif riw == rih:
    rid = (((fl*iw*fw)/((riw*fw)*sw))+
           ((fl*iw*fh)/((rih*fh)*sh)))*0.5 #average the height and width dist measurments
    rmmpp = iw/(((rih*fh)+(riw*fw))*0.5) #average the mmpp calculation
    rph = iw #the perceived height, in mm. This is equal to the actual height
    rpw = iw #the perceived width, in mm. This is equal to the actual height
    theta_rix = 0
    
  theta_rix = 90 - (90 - theta_rix)

  x = -(right_pupil[0]-rec[0] + left_pupil[0]-lec[0])/2
  y = (right_pupil[1]-rec[1] + left_pupil[1]-lec[1])/2

  aov = hfov*(lic_x-ric_x)

  theta1 = math.asin((rid*math.sin(math.radians(aov)))/(hw*((lmmpp+rmmpp)/2)*fw))

  print(aov,theta1)

  theta_face = 90-(90-math.degrees(math.atan((rid-lid)/((lic_x-ric_x)*fw*((lmmpp+rmmpp)/2)))))

  #make the angle negative
  if theta_face > 0:
    theta_rix = -theta_rix
    ri_proj_x = (rid*math.tan(math.radians(theta_rix+theta_face)))/rmmpp
  else:
    theta_rix = theta_rix
    ri_proj_x = (rid*math.tan(math.radians(theta_rix-theta_face)))/rmmpp


  #make the angle negative
  #if lit_x < lec_x:
  #theta_lix = -theta_lix

  li_proj_x = (lid*math.tan(math.radians(theta_lix-theta_face)))/lmmpp

  avg_proj_x = (ri_proj_x+li_proj_x)/2

  #put the positions in a buffer
  coord_buffer.put([left_pupil, right_pupil, lec, 
                    rec, midpoint_coords, x, 
                    y, center_depth, fl_xy, 
                    fr_xy, ft_xy, fb_xy, ri_proj_x,
                    li_proj_x, avg_proj_x])

def analyze_data(data):
  # Get the data from the input buffer and determine the number of hands
  face_landmarks_list = data[0]

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
      face_landmarks = face_landmarks_list[idx]
      counter = 0
      for i in indices:
        x=face_landmarks[i].x
        y=face_landmarks[i].y
        z=face_landmarks[i].z
        if counter == 0:
          data_array = np.array([[x, y, z]])
        else:
          data_array = np.append(data_array, [[x, y, z]], 0)
        counter += 1
      compute_view_angle(data_array)              
        
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
      
        if detect_buffer.qsize()>0 and coord_buffer.qsize()>0:
          # STEP 5: Process the classification result. In this case, visualize it.
          data = detect_buffer.get()
          frame = draw_landmarks_on_image(frame, data)

          #Get the key coordinates from the buffer
          key_coords = coord_buffer.get()
          left_pupil = (int(key_coords[0][0]*w), int(key_coords[0][1]*h))
          right_pupil = (int(key_coords[1][0]*w), int(key_coords[1][1]*h))
          left_eye = (int(key_coords[2][0]*w), int(key_coords[2][1]*h))
          right_eye = (int(key_coords[3][0]*w), int(key_coords[3][1]*h))
          LR_midpoint = (int(key_coords[4][0]*w), int(key_coords[4][1]*h))
          x = key_coords[5]
          y = key_coords[6]
          center_depth = key_coords[7]
          fl_xy = key_coords[8]
          fr_xy = key_coords[9]
          ft_xy = key_coords[10]
          fb_xy = key_coords[11]
          ri_proj_x = key_coords[12]
          li_proj_x = key_coords[13]
          avg_proj_x = key_coords[14]

          """
          if timestamp < 50:
            mouse.move(int(((x*dims[0]))),
                      int((y*dims[1])), duration=0.01)
          else:
            print(x, y, min_x, max_x, min_y, max_y)
            mouse.move(int((((((x-min_x)/max_x-min_x))*dims[0]))),
                      int((((y-min_y)/(max_y-min_y)))*dims[1]), duration=0.01)
          """

          #Draw the pupil and eye coordinates
          cv2.circle(frame, left_pupil, 1, (0,0,255), 1)
          cv2.circle(frame, right_pupil, 1, (0,0,255), 1)
          cv2.circle(frame, left_eye, 1, (255,255,255), 1)
          cv2.circle(frame, right_eye, 1, (255,255,255), 1)
          cv2.circle(frame, LR_midpoint, 5, (255,255,255), 10)
          cv2.line(frame, (int(fl_xy[0]*w),int(fl_xy[1]*h)), 
                   (int(fr_xy[0]*w),int(fr_xy[1]*h)), (255,0,0), 2)
          cv2.line(frame,(int(ft_xy[0]*w), int(ft_xy[1]*h)),
                   (int(fb_xy[0]*w),int(fb_xy[1]*h)), (255,0,0), 2)
          cv2.line(frame, right_pupil, (int(ri_proj_x),int(h/2)), (0,0,255), 2)
          cv2.line(frame, left_pupil, (int(li_proj_x),int(h/2)), (0,255,0), 2)
          cv2.line(frame, right_pupil, (int(avg_proj_x),int(h/2)), (255,255,0), 2)
          cv2.line(frame, left_pupil, (int(avg_proj_x),int(h/2)), (255,255,0), 2)

          
        cv2.imshow('Video Feed', frame)
          
        timestamp+=1

        cv2.waitKey(1)

        end = time.time()
        global duration
        duration = end-start
        frame_rate = 1/(end-start)
        #print(frame_rate)


