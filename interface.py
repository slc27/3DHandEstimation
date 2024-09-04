import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import PySimpleGUI as sg
import pygame
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import ttk
import yaml

DISTANCE_THRESHOLD = 3
BACKGROUND_COLOR = '#2596be'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
frame_shape = [720, 1280]

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

#direct linear transform
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')

    camera0 = calibration_settings['camera0']
    camera1 = calibration_settings['camera1']

    return camera0, camera1

def read_camera_parameters(camera_id):

    with open('camera_parameters/camera' + str(camera_id) + '_intrinsics.dat', 'r') as inf:

        cmtx = []
        dist = []

        line = inf.readline()
        for _ in range(3):
            line = inf.readline().split()
            line = [float(en) for en in line]
            cmtx.append(line)

        line = inf.readline()
        line = inf.readline().split()
        line = [float(en) for en in line]
        dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):
    #Change name based on origin
    with open(savefolder + 'camera'+ str(camera_id) + '_rot_trans.dat', 'r') as inf:
    #with open(savefolder + 'world_to_camera'+ str(camera_id) + '_rot_trans.dat', 'r') as inf:

        inf.readline()
        rot = []
        trans = []
        for _ in range(3):
            line = inf.readline().split()
            line = [float(en) for en in line]
            rot.append(line)

        inf.readline()
        for _ in range(3):
            line = inf.readline().split()
            line = [float(en) for en in line]
            trans.append(line)

        inf.close()
    return np.array(rot), np.array(trans)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    with open(filename, 'w') as fout: 

        for frame_kpts in kpts:
            for kpt in frame_kpts:
                if len(kpt) == 2:
                    fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
                else:
                    fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

            fout.write('\n')
        fout.close()

def initialize_cameras(input_stream1, input_stream2):
    # Initialize cameras
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # Set up camera resolution
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    return caps

def direction_guide(distance):
    #Dictionary that associates directions with corresponding images of arrows
    direction_map = {
        "KEEP": arrow_image[0],
        "LEFT": arrow_image[1],
        "RIGHT": arrow_image[2],
        "UP": arrow_image[3],
        "DOWN": arrow_image[4],
        "FRONT": arrow_image[5],
        "BACK": arrow_image[6],
    }

    #Save the furthest direction
    if (np.sum(np.abs(distance))>DISTANCE_THRESHOLD):
        absolute_distances = [abs(coordenada) for coordenada in distance]
        max_distancia = max(absolute_distances)
        index_max_distance = absolute_distances.index(max_distancia)
        far_distance = distance[index_max_distance]
        directions = {0:("DOWN", "UP"), 1:("FRONT", "BACK"), 2:("RIGHT", "LEFT")}
        direction = directions[index_max_distance][0] if far_distance < 0 else directions[index_max_distance][1]
    else:
        direction = "KEEP"
    
    image_arrow.config(image=direction_map[direction])   


    return direction


def emit_sound(distance, sound,i,media=0):
    distance_abs = np.abs(distance)
    media = np.round(np.sum(distance_abs))

    # Emit sound depending on the distance
    if int(media)!=0:
        if not (i%int(media)):
            #print("Emitting sound!")
            sound.play() 

def plot_3d(ax, kpts3d, point_stablished, fingers, fingers_colors, sound,i, origin):

    ax.clear() #ax reset
    
    radius = 20
    
    # Set figure colors
    ax.set(facecolor=BACKGROUND_COLOR)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.3))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.7))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.5))

    hand_center = kpts3d[9]  # Get the position of the center of the hand
    
    #Distance between origin and the center of the hand
    distance = np.linalg.norm(hand_center - point_stablished)
    distance_array = hand_center-np.array(point_stablished)

    #Plot the 3D axis and the limits of the 3D figure
    ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2]+5], color='green', linewidth=2, label='z-axis')  # Eje z
    ax.plot([origin[0], origin[0]], [origin[1], origin[1]+5], [origin[2], origin[2]], color='red', linewidth=2, label='y-axis')  # Eje y
    ax.plot([origin[0], origin[0]+5], [origin[1], origin[1]], [origin[2], origin[2]], color='blue', linewidth=2, label='x-axis')  # Eje x
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim3d(origin[0]-radius, origin[0]+radius)
    ax.set_xlabel('x')
    ax.set_ylim3d(origin[1]-radius, origin[1]+radius)
    ax.set_ylabel('y')
    ax.set_zlim3d(origin[2]-radius, origin[2]+radius)
    ax.set_zlabel('z')
    ax.azim = -45
    ax.elev = 30

    #Plot the fingers
    for finger, finger_color in zip(fingers, fingers_colors):
        for _c in finger:
            ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]], 
                    ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]], 
                    zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]], 
                    linewidth=4, c=finger_color)
    
    #Detect strange points
    if not np.array_equal(point_stablished, np.array([-1, -1, -1])) and not np.array_equal(point_stablished, np.array([-1, -1, 1])):
        direction = direction_guide(distance_array)
        
        #Set color based on button pressed
        point_color = 'red' if np.array_equal(point_stablished, np.array(origin)) else 'yellow'
        ax.scatter(point_stablished[0], point_stablished[1], point_stablished[2], c=point_color, marker='o', s=100,label='3DPoint')
        
        if distance < DISTANCE_THRESHOLD:       
            #Plot a line between the hand and the origin     
            ax.plot([point_stablished[0], hand_center[0]], 
                [point_stablished[1], hand_center[1]], 
                [point_stablished[2], hand_center[2]], 
                color='green', linewidth=3, alpha=0.5, label='Line between points')
        elif not np.array_equal(hand_center, np.array([-1, -1, 1])): #Check correct estimation
            ax.plot([point_stablished[0], hand_center[0]], 
                [point_stablished[1], hand_center[1]], 
                [point_stablished[2], hand_center[2]], 
                color='red', linewidth=3, alpha=0.5, label='Line between points')

        if not np.array_equal(hand_center, np.array([-1, -1, 1])): #Check correct estimation
            text_direction.config(text=direction)
            distance_array_form = [round(numero, 1) for numero in distance_array]
            text_distance.config(text=str(distance_array_form))
            #Set the color of 3d hand point
            hand_color = 'green' if distance < DISTANCE_THRESHOLD else 'red'
            ax.scatter(hand_center[0], hand_center[1], hand_center[2], c=hand_color, marker='o', s=800, alpha=0.3)
          
            emit_sound(distance_array, sound,i)
        else:
            image_arrow.config(image=arrow_background)
            text_direction.config(text="Hand out of range")
            text_distance.config(text="NO HAND DETECTED")
            

def draw_keypoints(frame0, frame1, results0, results1, point0, point1):
    # Draw the hand annotations on the image.
    frame0.flags.writeable = True
    frame1.flags.writeable = True
    #Optional conversion to BGR
    #frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
    #frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

    if results0.multi_hand_landmarks:
        for hand_landmarks in results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results1.multi_hand_landmarks:
        for hand_landmarks in results1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if not np.array_equal(point0, np.array([0,0])) and not np.array_equal(point1, np.array([0,0])):
        # Draw an "X" at the point set at frame0
        cv.line(frame0, (point0[0] - 10, point0[1] - 10), (point0[0] + 10, point0[1] + 10), (255, 0, 0), 2)
        cv.line(frame0, (point0[0] - 10, point0[1] + 10), (point0[0] + 10, point0[1] - 10), (255, 0, 0), 2)

        # Draw an "X" at the point set at frame1
        cv.line(frame1, (point1[0] - 10, point1[1] - 10), (point1[0] + 10, point1[1] + 10), (255, 0, 0), 2)
        cv.line(frame1, (point1[0] - 10, point1[1] + 10), (point1[0] + 10, point1[1] - 10), (255, 0, 0), 2)


    frame0_small = cv.resize(frame0, None, fx=1./1.5, fy=1./1.5)
    frame1_small = cv.resize(frame1, None, fx=1./1.5, fy=1./1.5)
    
    # Show frames in the interface
    img_pil1 = Image.fromarray(frame0_small)
    img_tk1 = ImageTk.PhotoImage(image=img_pil1)
    video1.config(image=img_tk1)
    video1.image = img_tk1
    img_pil2 = Image.fromarray(frame1_small)
    img_tk2 = ImageTk.PhotoImage(image=img_pil2)
    video2.config(image=img_tk2)
    video2.image = img_tk2

def keypoint_extraction(caps, hands0, hands1, kpts_cam0, kpts_cam1, kpts_3d, Rz, Rx, point0, point1):
    # Read de frames
    ret0, frame0 = caps[0].read()
    ret1, frame1 = caps[1].read()

    if not ret0 or not ret1:
        return

    # Frame preprocessing
    if frame0.shape[1] != 720:
        frame0 = frame0[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]
        frame1 = frame1[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]

    frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    frame0.flags.writeable = False
    frame1.flags.writeable = False

    # Process frames using hand detection models
    results0 = hands0.process(frame0)
    results1 = hands1.process(frame1)

    # Process keypoints for frame0
    frame0_keypoints = []
    if results0.multi_hand_landmarks:
        for hand_landmarks in results0.multi_hand_landmarks:
            frame0_keypoints.extend([[int(round(frame0.shape[1] * lm.x)), int(round(frame0.shape[0] * lm.y))] for lm in hand_landmarks.landmark])
    else:
        frame0_keypoints = [[-1, -1]] * 21

    kpts_cam0.append(frame0_keypoints)

    # Process keypoints for frame1
    frame1_keypoints = []
    if results1.multi_hand_landmarks:
        for hand_landmarks in results1.multi_hand_landmarks:
            frame1_keypoints.extend([[int(round(frame1.shape[1] * lm.x)), int(round(frame1.shape[0] * lm.y))] for lm in hand_landmarks.landmark])
    else:
        frame1_keypoints = [[-1, -1]] * 21

    kpts_cam1.append(frame1_keypoints)

    # Calculate 3D positions
    frame_p3ds = []
    for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        if -1 in uv1 or -1 in uv2:
            frame_p3ds.append([-1, -1, -1])
        else:
            frame_p3ds.append(DLT(P0, P1, uv1, uv2))

    frame_p3ds = np.array(frame_p3ds)
    kpts_3d.append(frame_p3ds)

    # Rotate keypoints
    frame_kpts_rotated = np.dot(Rz, np.dot(Rx, frame_p3ds.T)).T

    draw_keypoints(frame0, frame1, results0, results1, point0, point1)

    return frame_kpts_rotated, frame0_keypoints, frame1_keypoints

def create_figure():
    # Create figure for visualization
    fig = plt.figure(figsize=(5, 5), facecolor=BACKGROUND_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master = screen_frame)
    canvas.get_tk_widget().pack(side='right')
    return ax

def create_fingers():
    # Fingers definition
    thumb_f = [[0, 1], [1, 2], [2, 3], [3, 4]]
    index_f = [[0, 5], [5, 6], [6, 7], [7, 8]]
    middle_f = [[0, 9], [9, 10], [10, 11], [11, 12]]
    ring_f = [[0, 13], [13, 14], [14, 15], [15, 16]]
    pinkie_f = [[0, 17], [17, 18], [18, 19], [19, 20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['lightsalmon'] * 5

    return fingers, fingers_colors

def run_mp(input_stream1, input_stream2, P0, P1):
    global set_button, set_buttonauto, set_buttonreset #Button definitions

    reset_pressed() #initial GUI reset
    set_button = False
    set_buttonauto = False
    set_buttonreset = False
    caps = initialize_cameras(input_stream1, input_stream2)

    # Hand Object creations
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)

    # Keypoints containers
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    # Define axis rotations
    Rz = np.array(([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]]))
    Rx = np.array(([[1.,  0.,  0.],
                    [0., -1.,  0.],
                    [0.,  0., -1.]]))
    
    ax = create_figure()
    fingers, fingers_colors = create_fingers()

    # Main loop
    flag_spoint = False
    point_stablished = np.array([-1,-1,-1])
    point2D_0 = np.array([0,0])
    point2D_1 = np.array([0,0])
    preset_point = np.loadtxt('files/preset_point.txt')

    # pygame initializations
    pygame.init()
    sound = pygame.mixer.Sound('files/beep.wav')
    sound.set_volume(0.02) 

    i=0

    while True:    
        i+=1
        k = cv.waitKey(1)
        kpts3d, kpts2d_0, kpts2d_1 = keypoint_extraction(caps, hands0, hands1, kpts_cam0, kpts_cam1, kpts_3d, Rz, Rx, point2D_0, point2D_1)
        
        if set_button and not flag_spoint:
            point_stablished = kpts3d[9]
            point2D_0 = kpts2d_0[9]
            point2D_1 = kpts2d_1[9]
            if(not np.array_equal(point_stablished, np.array([-1, -1, 1]))): #Detect strange point declaration
                array_formateado = [round(numero, 2) for numero in point_stablished]
                np.savetxt('files/preset_point.txt', array_formateado)
                preset_point = np.loadtxt('files/preset_point.txt')
                text1.config(text="NEW ORIGIN POINT SET")
                flag_spoint = True
            else:
                tk.messagebox.showinfo("ADVICE", "Please put your hand in the scanner")
                set_button = False
        elif set_buttonauto and not flag_spoint:
            point_stablished = preset_point
            point2D_0 = [0, 0]
            point2D_1 = [0, 0]
            array_formateado = [round(numero, 2) for numero in point_stablished]
            text1.config(text="NEW ORIGIN POINT SET")
            flag_spoint = True

        if not flag_spoint:
            text1.config(text="Press 'SET' to set the origin")
            text_direction.config(text="DIRECTION")
            text_distance.config(text="DISTANCE")

        if set_buttonreset:
            set_button = False
            set_buttonauto = False
            flag_spoint = False
            point_stablished = [-1, -1, -1]
            set_buttonreset = False

        plot_3d(ax, kpts3d, point_stablished, fingers, fingers_colors, sound,i, preset_point)
        plt.draw()
        plt.pause(0.01)
        
        if k & 0xFF == 27: break #27 is ESC key.

    # release resources
    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
    hands0.close()
    hands1.close()

    return

def set_pressed():
    global set_button
    set_button = True

    return set_button

def setauto_pressed():
    global set_buttonauto
    set_buttonauto = True

    return set_buttonauto

def reset_pressed():
    global set_buttonreset
    set_buttonreset = True
    image_arrow.config(image=arrow_background)
    return set_buttonreset

def gui():
    global text1, text_direction, text_distance
    global screen_frame
    global video1, video2
    global button_set, button_autoset
    global image_arrow, arrow_image, arrow_background

    #Create the window
    screen = tk.Tk()
    screen.title("GUI")
    #Create the frame
    screen_frame = tk.Frame(screen, bg=BACKGROUND_COLOR, height=1000, width=1000)
    # pack_propagate prevents the window resizing to match the widgets
    screen_frame.pack_propagate(0)
    screen_frame.pack(fill="both", expand="true")

    screen.geometry("1000x1000")
    screen.resizable(0, 0)

    text_styles = {"font": ("Verdana", 15),
                "background": BACKGROUND_COLOR,
                "foreground": "#E1FFFF"}

    tk.messagebox.showinfo("ORIGIN", "You must set the point manually if the cameras have been calibrated")

    # Create a label
    text1 = tk.Label(screen_frame, text_styles,text="")
    text1.grid(row=0,column=0, padx=[100,0], pady=[75,50], columnspan=3)
    text_direction = tk.Label(screen_frame, text_styles,text="DIRECTION")
    text_direction.grid(row=1,column=0, padx=[50,0], pady=[0,50], columnspan=3)
    text_distance = tk.Label(screen_frame, text_styles,text="DISTANCE")
    text_distance.grid(row=2,column=0, padx=[50,0], pady=[0,50], columnspan=3)

    # Create a button
    button_set = tk.Button(screen_frame, text="SET MANUAL", command=set_pressed)
    button_set.grid(row=3,column=0, padx=0, pady=0)
    button_autoset = tk.Button(screen_frame, text="SET AUTO", command=setauto_pressed)
    button_autoset.grid(row=3,column=1, padx=0, pady=0)
    button_reset = tk.Button(screen_frame, text="RESET", command=reset_pressed)
    button_reset.grid(row=3,column=2, padx=0, pady=0)

    canvas1 = tk.Canvas(screen, width=720, height=480)
    canvas1.pack(side='left',fill="both", expand="true")
    video1 = tk.Label(canvas1, bg=BACKGROUND_COLOR)
    video1.pack_propagate(0)
    video1.pack(fill="both", expand="true")

    canvas2 = tk.Canvas(screen, width=720, height=480)
    canvas2.pack(side='right',fill="both", expand="true")
    video2 = tk.Label(canvas2, bg=BACKGROUND_COLOR)
    video2.pack_propagate(0)
    video2.pack(fill="both", expand="true")

    arrow_none = Image.open("files/arrow_none.png")
    arrow_left = Image.open("files/arrow4.png")
    arrow_right = Image.open("files/arrow2.png")
    arrow_up = Image.open("files/arrow.png")
    arrow_down = Image.open("files/arrow3.png")
    arrow_front = Image.open("files/arrow5.png")
    arrow_back = Image.open("files/arrow6.png")

    arrows = [
        arrow_none,
        arrow_left,
        arrow_right,
        arrow_up,
        arrow_down,
        arrow_front,
        arrow_back
    ]

    arrow_image = []

    for arrow in arrows:
        resized_arrow = arrow.resize((50, 50))
        arrow_image.append(ImageTk.PhotoImage(resized_arrow))


    # Create a photoimage object of the image in the path
    img = Image.open("files/arrow_background.png")
    image1 = img.resize((100, 100))
    arrow_background = ImageTk.PhotoImage(image1)
    image_arrow = tk.Label(image=arrow_background, background=BACKGROUND_COLOR)
    image_arrow.place(x=180, y = 400)
    img = Image.open("files/huro.png")
    image1 = img.resize((75, 75))
    huro = ImageTk.PhotoImage(image1)
    image_huro= tk.Label(image=huro, background=BACKGROUND_COLOR)
    image_huro.place(x=600, y = 435)
    img = Image.open("files/myorehab.png")
    image1 = img.resize((140, 50))
    myorehab = ImageTk.PhotoImage(image1)
    image_myorehab = tk.Label(image=myorehab, background=BACKGROUND_COLOR)
    image_myorehab.place(x=850, y = 455)
    
    # Execute main processing
    run_mp(input_stream1, input_stream2, P0, P1)

    # Release resources
    cv.destroyAllWindows()

    screen.mainloop()

if __name__ == '__main__':
    filename = "calibration_settings.yaml"
    input_stream1, input_stream2 = parse_calibration_settings_file(filename)

    # Get the projection matrix
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    gui()


