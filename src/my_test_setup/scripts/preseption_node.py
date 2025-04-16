#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep
import os

# ROS Imports
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# --- Globals ---
# Shared data between ZED grab thread and YOLO thread (if used)
# If YOLO is in main loop, these might not be needed across threads in *this* node
image_zed = None
image_lock = Lock()
run_signal = False # Signal for separate YOLO thread (optional)
exit_signal = False # Signal for separate YOLO thread (optional)
bridge = CvBridge()

# --- YOLO Function (can be run in thread or main loop) ---
def detect_persons(yolo_model, image_np_bgra, img_size, conf_thres, iou_thres):
    """Runs YOLO detection and returns True if a person is detected."""
    is_person_detected = False
    try:
        img_rgb = cv2.cvtColor(image_np_bgra, cv2.COLOR_BGRA2RGB)
        results = yolo_model.predict(img_rgb, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, classes=0, verbose=False) # classes=0 for person

        if results and results[0].boxes:
            # Check if any detection belongs to class 0 (person)
            detected_classes = results[0].boxes.cls.cpu().numpy()
            if 0 in detected_classes:
                 rospy.logdebug("Person detected by YOLO.")
                 is_person_detected = True
            # else:
            #      rospy.logdebug("Objects detected, but no person.")
        # else:
        #      rospy.logdebug("No objects detected.")

    except Exception as e:
        rospy.logerr(f"Error during YOLO prediction: {e}")

    return is_person_detected

# --- Main Function ---
def main(opt):
    global image_zed, bridge

    rospy.init_node('perception_node', anonymous=True)
    rospy.loginfo("Starting Perception Node...")

    # --- ROS Publishers ---
    image_pub = rospy.Publisher('/perception/image_raw', Image, queue_size=1) # Publish raw image
    person_detected_pub = rospy.Publisher('/perception/person_detected', Bool, queue_size=1) # Publish detection status

    # --- ZED Camera Setup ---
    zed = sl.Camera()
    input_type = sl.InputType()
    if opt.svo is not None:
        rospy.loginfo(f"Using SVO file: {opt.svo}")
        input_type.set_from_svo_file(opt.svo)

    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # ROS standard is usually ENU/FLU, but Y_UP is fine for image processing
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Enable if depth needed later
    init_params.depth_maximum_distance = 15 # Set max depth if using depth mode

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        rospy.logfatal(f"Error opening ZED camera: {repr(status)}")
        sys.exit(1)

    rospy.loginfo("ZED Camera Opened Successfully.")

    # --- YOLO Model Initialization ---
    rospy.loginfo(f"Loading YOLO model: {opt.weights}")
    try:
        yolo_model = YOLO(opt.weights)
        # Optional: Run a dummy inference to warm up the model
        dummy_img = np.zeros((int(opt.img_size), int(opt.img_size), 3), dtype=np.uint8)
        yolo_model.predict(dummy_img, verbose=False)
        rospy.loginfo("YOLO model loaded.")
    except Exception as e:
        rospy.logfatal(f"Failed to load YOLO model: {e}")
        zed.close()
        sys.exit(1)


    image_left_mat = sl.Mat() # Reusable ZED Mat object

    # Get camera info once
    camera_info = zed.get_camera_information()
    cam_res = camera_info.camera_configuration.resolution

    # Optional: Limit loop rate
    rate = rospy.Rate(30) # Target 30 Hz, adjust as needed

    # --- Main Loop ---
    while not rospy.is_shutdown():
        grab_status = zed.grab(runtime_params)
        if grab_status == sl.ERROR_CODE.SUCCESS:
            rospy.loginfo("Got img")
            # Retrieve image
            # Retrieve full resolution BGRA image for YOLO
            zed.retrieve_image(image_left_mat, sl.VIEW.LEFT, sl.MEM.CPU) # No resolution change here
            image_zed_full_res_bgra = image_left_mat.get_data()

            # --- Run YOLO Detection ---
            # Running YOLO in the main loop for simplicity here.
            # If performance is an issue, a separate thread could be used,
            # publishing results asynchronously.
            person_detected = detect_persons(yolo_model, image_zed_full_res_bgra, opt.img_size, opt.conf_thres, opt.iou_thres)

            # --- Publish Results ---
            # 1. Publish Person Detected Status
            person_detected_pub.publish(Bool(data=person_detected))

            # 2. Publish Image (use the same BGRA image)
            # Publish only if someone is subscribed to avoid unnecessary processing
            if image_pub.get_num_connections() > 0:
                try:
                    # It's often better to publish in BGR8 or RGB8 format for compatibility
                    # image_bgr = cv2.cvtColor(image_zed_full_res_bgra, cv2.COLOR_BGRA2BGR)
                    # img_msg = bridge.cv2_to_imgmsg(image_bgr, "bgr8")

                    # Or publish directly as BGRA if the subscriber can handle it
                    img_msg = bridge.cv2_to_imgmsg(image_zed_full_res_bgra, "bgra8")

                    img_msg.header.stamp = rospy.Time.now()
                    img_msg.header.frame_id = "zed_left_camera_frame" # Or your relevant TF frame
                    image_pub.publish(img_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error: {e}")
                except Exception as e:
                    rospy.logerr(f"Error publishing image: {e}")

            # Optional sleep to control rate
            rate.sleep()

        elif grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            rospy.loginfo("End of SVO file reached.")
            break # Exit loop
        else:
            rospy.logwarn_throttle(5.0, f"ZED grab failed: {repr(grab_status)}") # Warn occasionally
            sleep(0.01) # Small sleep on failure


    # --- Cleanup ---
    rospy.loginfo("Shutting down Perception Node...")
    # exit_signal = True # If using a thread, signal it
    # capture_thread.join() # If using a thread, wait for it
    zed.close()
    rospy.loginfo("ZED camera closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/robot/xjobb/yolodetection/yolo11n.pt', help='YOLO model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='Optional path to an SVO file')
    parser.add_argument('--img_size', type=int, default=640, help='YOLO inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='YOLO object confidence threshold') # Adjusted default
    parser.add_argument('--iou_thres', type=float, default=0.5, help='YOLO NMS IOU threshold') # Adjusted default
    # Parse known args allows ROS command line args (like __name, __log)
    opt, unknown = parser.parse_known_args()

    # No torch.no_grad() needed here as predict manages it internally
    main(opt)