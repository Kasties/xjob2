#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, String,Float64,Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import requests
import base64
import json
import os
from time import time
from threading import Lock
from dotenv import load_dotenv
from nav_msgs.msg import Odometry 
import tf.transformations
import math
# Load API Key from .env file in the same directory or parent directories
# Alternatively, use ROS params
load_dotenv()

class GeminiNavigator:
    def __init__(self):
        rospy.init_node('gemini_navigator_node', anonymous=True)
        rospy.loginfo("Starting Gemini Navigator Node...")

        # --- Gemini Configuration ---
        # Use ROS param for API key if preferred (more secure/flexible)
        # self.api_key = rospy.get_param("~google_api_key", os.getenv('GOOGLE_API_KEY'))
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.gemini_api_available = bool(self.api_key)
        # Use the latest appropriate model available via REST API
        self.gemini_model_name = "gemini-2.0-flash" # Check Google AI documentation
        self.gemini_api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_name}:generateContent"

        if not self.gemini_api_available:
            rospy.logwarn("GOOGLE_API_KEY environment variable not set or ROS param not found.")
            rospy.logwarn("Gemini functionality will be disabled.")
        else:
            rospy.loginfo("Gemini API key found. Functionality enabled.")
        self.wl = 1
        self.wa = 1
        # --- State Variables ---
        self.latest_image_msg = None
        self.image_lock = Lock()
        self.person_detected = False
        self.last_gemini_call_time = 0
        # Get interval from ROS param or use default
        self.gemini_call_interval = rospy.get_param("~gemini_interval", 5.0) # seconds
        self.bridge = CvBridge()
        self.odom_lock = Lock()
        # --- ROS Subscribers ---
        rospy.Subscriber('/perception/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24) # Increase buffer size if needed
        rospy.Subscriber('/perception/person_detected', Bool, self.person_detected_callback, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        # --- ROS Publishers ---
        #self.direction_pub = rospy.Publisher('/gemini/direction', float, queue_size=1)
        #self.speed_pub = rospy.Publisher('/gemini/speed', float, queue_size=1)
        self.social_pub = rospy.Publisher('/gemini/social', Int32, queue_size=1)

        # --- Navigation Prompt ---
        # TODO: Get current robot state (direction, velocity) from ROS topics if available
        self.current_heading = 0 # Example placeholder
        self.current_velocity = 0 # Example placeholder

        self.navigation_prompt_template = f"""Task:
            Based *only* on the visual information in the image, how should I adjust my movement to navigate safely and courteously around the person/people visible? Follow standard walking etiquette.

            My Current State (Ego state):
            - Current Heading Direction Estimate: {self.current_heading} degrees (relative to forward)
            - Current Linear Velocity Estimate: {self.current_velocity} m/s

            Instructions:
            - Prioritize safety and maintaining personal space.
            - When passing someone coming towards you or from behind, generally move to your right (relative to your direction of travel).
            - Avoid obstructing pathways.
            - If the path is unclear or too crowded, stopping or slowing down is appropriate.
            - Assume you want to generally continue moving forward unless the situation requires otherwise.
            - Provide a concise action command.


                """

        rospy.loginfo("Gemini Navigator Node Initialized.")


    def image_callback(self, msg):
        # Store the latest image message
        with self.image_lock:
            # rospy.logdebug("Received new image message.")
            self.latest_image_msg = msg

    def person_detected_callback(self, msg):
        # Update person detected status
        if msg.data != self.person_detected: # Log only on change
             rospy.loginfo(f"Person detected status changed to: {msg.data}")
        self.person_detected = msg.data
    def odom_callback(self, msg):
        """Processes odometry messages to update current velocity and heading."""
        with self.odom_lock:
            # Get linear velocity (usually X for non-holonomic robots)
            self.current_velocity = msg.twist.twist.linear.x

            # Get orientation as quaternion
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

            # Update heading in degrees (yaw)
            self.current_heading = math.degrees(yaw)

            rospy.logdebug(f"Odom updated: Vel={self.current_velocity:.2f} m/s, Heading={self.current_heading:.1f} deg")
    def call_gemini_rest_api(self, image_np_bgra, prompt):
        """
        Sends an image (NumPy array BGRA) to Gemini REST API and parses response.
        Returns a tuple (direction, speed) or (None, None) on failure.
        """
        if not self.gemini_api_available:
            rospy.logwarn_throttle(10.0, "Gemini API key not available. Skipping call.")
            return None, None

        rospy.loginfo("Preparing to call Gemini API...")
        direction = None
        speed = None

        try:
            # 1. Convert BGRA to RGB (Gemini generally prefers RGB)
            image_rgb = cv2.cvtColor(image_np_bgra, cv2.COLOR_BGRA2RGB)

            # 2. Encode image to JPEG format in memory
            retval, buffer = cv2.imencode('.jpg', image_rgb)
            if not retval:
                rospy.logerr("--- Error encoding image to JPEG ---")
                return None, None

            # 3. Base64 encode the image bytes
            jpg_base64 = base64.b64encode(buffer).decode('utf-8')

            # 4. Construct the request payload with JSON schema for output
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": jpg_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "direction": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                            "left",
                            "straight",
                            "right"
                            ]
                        },
                        "description": "The intended direction of movement."
                        },
                        "speed": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                            "slow down",
                            "speed up",
                            "constant",
                            "stop"
                            ]
                        },
                        "description": "The intended change in speed or state of motion."
                        }
                    },
                    "required": [
                        "direction",
                        "speed"
                    ],
                }
            }
        }

            # 5. Make the API call using requests
            headers = {'Content-Type': 'application/json'}
            full_url = f"{self.gemini_api_endpoint}?key={self.api_key}"

            rospy.loginfo("--- Calling Gemini REST API ---")
            response = requests.post(full_url, headers=headers, json=payload, timeout=60) # Added timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # 6. Process the response
            response_data = response.json()
            rospy.logdebug(f"--- Gemini Raw Response: {json.dumps(response_data, indent=2)} ---")

            # Safely extract the JSON content
            generated_text = ""
            finish_reason = "UNKNOWN"

            if 'candidates' in response_data and response_data['candidates']:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                    generated_text = candidate['content']['parts'][0].get('text', '') 
                finish_reason = candidate.get('finishReason', 'UNKNOWN')

            if generated_text:
                if json.loads(generated_text).get("direction") != [] and json.loads(generated_text).get("speed") != []:
                    print("\n--- Gemini Response ---")
                    print(generated_text)
                    direction = json.loads(generated_text).get("direction")[0]
                    if direction == "left":
                        direction = 0.5
                    elif direction == "straight":
                        direction = 0
                    elif direction == "right":
                        direction = 1
                    else: direction = 0
                    speed = json.loads(generated_text).get("speed")[0]
                    if speed == "slow down":
                        speed = -0.5
                    elif speed == "speed up":
                        speed = 0.5
                    elif speed == "constant":
                        speed = 0
                    elif speed == "Stop":
                        speed = -1     
                    else: speed = 0  
                    print(f"direction = {direction} speed = {speed}")
                    print("-----------------------\n")
                else:
                    print("\n--- Gemini failed to find path ---")
                    return 0,0
            else:
                # Handle cases where response is empty or blocked
                print("\n--- Gemini Response Blocked or Empty ---")
                print(f"Finish Reason: {finish_reason}")

        except requests.exceptions.RequestException as e:
            rospy.logerr(f"--- Error during Gemini REST API call (Network/Request): {e} ---")
        except requests.exceptions.HTTPError as e:
            rospy.logerr(f"--- Error during Gemini REST API call (HTTP Status): {e.response.status_code} {e.response.reason} ---")
            try:
                 rospy.logerr(f"Response body: {e.response.text}") # Show error details from API if available
            except Exception:
                 pass # Ignore if response body is not readable
        except json.JSONDecodeError as e:
            # This might happen if the initial response.json() fails
            rospy.logerr(f"--- Error decoding initial Gemini API JSON response structure: {e} ---")
            try:
                 rospy.logerr(f"Response text: {response.text}")
            except NameError: # response might not be defined if request failed early
                 pass
        except Exception as e:
            # Catch any other unexpected errors
            import traceback
            rospy.logerr(f"--- An unexpected error occurred during Gemini interaction: {e} ---")
            rospy.logerr(traceback.format_exc()) # Log detailed traceback

        return direction, speed


    def run(self):
        rate = rospy.Rate(2) # Check conditions at 2 Hz (adjust as needed)

        while not rospy.is_shutdown():
            current_time = time() # Use time.time() which is compatible without ROS time sync issues initially

            # Check conditions for calling Gemini
            if self.person_detected and self.gemini_api_available:
                if (current_time - self.last_gemini_call_time >= self.gemini_call_interval):
                    image_to_process = None
                    # Safely get the latest image
                    with self.image_lock:
                        if self.latest_image_msg:
                            try:
                                # Convert ROS Image message to OpenCV format (BGRA8 based on perception node)
                                image_to_process = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding="bgra8")
                                rospy.logdebug(f"Processing image from time: {self.latest_image_msg.header.stamp.to_sec()}")
                            except CvBridgeError as e:
                                rospy.logerr(f"CvBridge Error converting image: {e}")
                                image_to_process = None # Ensure it's None on error
                            except Exception as e:
                                rospy.logerr(f"Error converting image: {e}")
                                image_to_process = None # Ensure it's None on error

                            # Clear the message buffer after processing to avoid reprocessing old images if callbacks lag
                            # self.latest_image_msg = None # Optional: uncomment if you only want the *very* latest

                    if image_to_process is not None:
                        # Update prompt with latest (example) state if needed
                        # Update self.navigation_prompt_template here if self.current_heading/velocity are updated via other ROS topics
                        with self.odom_lock:
                            current_vel = self.current_velocity
                            current_head = self.current_heading
                        # Call Gemini API (potentially in a separate thread if it blocks too long)
                        # For simplicity, calling directly first. Add threading if main loop gets stuck.
                        navigation_prompt_template = f"""Task:
                                    Based *only* on the visual information in the image, how should I adjust my movement to navigate safely and courteously around the person/people visible? Follow standard walking etiquette.

                                    My Current State (Ego state):
                                    - Current Heading Direction Estimate: {self.current_heading} degrees (relative to forward)
                                    - Current Linear Velocity Estimate: {self.current_velocity} m/s

                                    Instructions:
                                    - Prioritize safety and maintaining personal space.
                                    - When passing someone coming towards you or from behind, generally move to your right (relative to your direction of travel).
                                    - Avoid obstructing pathways.
                                    - If the path is unclear or too crowded, stopping or slowing down is appropriate.
                                    - Assume you want to generally continue moving forward unless the situation requires otherwise.
                                    - Provide a concise action command.
                                    - Slowing down or stopping is important if a collision is possible.


                                        """

                        print(navigation_prompt_template)
                        print(f"current heading = {self.current_heading} vel = {self.current_velocity}")
                        direction, speed = self.call_gemini_rest_api(image_to_process, navigation_prompt_template)
                        #print(f"return direction = {direction} speed = {speed}")

                        self.last_gemini_call_time = current_time # Update time even if call failed to prevent rapid retries

                        # Publish results if valid
                        C_social = self.wl * abs(self.current_velocity - speed) + self.wa * abs(self.current_heading - direction)
                        print(f"C_social = {C_social}")
                        if direction != -99 and speed != -99: #NOTE dumb fix
                            #self.direction_pub.publish(float(data=direction))
                            #self.speed_pub.publish(float(data=speed))
                            self.social_pub.publish(Int32(data=1))
                            rospy.loginfo(f"Published Gemini Command: Direction='{direction}', Speed='{speed}'")
                        else:
                            rospy.logwarn("Gemini call did not return a valid direction and speed.")

                    else:
                        rospy.logwarn_throttle(5.0, "Person detected, but no valid image available to send to Gemini.")
                # else: # Optional debug log
                #     rospy.logdebug_throttle(5.0, f"Person detected, but waiting for interval. {current_time - self.last_gemini_call_time:.1f}s / {self.gemini_call_interval}s")


            # Check other conditions or perform other tasks
            # e.g., maybe publish "straight" / "constant" if no person detected for a while?

            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = GeminiNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Gemini Navigator Node Shutting Down.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in Gemini Navigator: {e}")
        import traceback
        rospy.logfatal(traceback.format_exc())