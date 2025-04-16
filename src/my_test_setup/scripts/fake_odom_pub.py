#!/usr/bin/env python

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
import math

def fake_odometry():
    rospy.init_node('fake_odom_publisher', anonymous=True)

    odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)

    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # --- Define the static pose and twist ---
    # Position and Orientation (at origin, no rotation)
    static_x = 1.0
    static_y = 1.0
    static_theta = 0.0

    # Velocities (zero)
    static_vx = 1.0
    static_vyaw = 0.0
    # ---

    odom_msg = Odometry()
    odom_msg.header.frame_id = "odom"       # Standard odom frame
    odom_msg.child_frame_id = "base_link"   # Your robot's base frame

    tf_msg = TransformStamped()
    tf_msg.header.frame_id = "odom"
    tf_msg.child_frame_id = "base_link"

    rate = rospy.Rate(20) # Publish at 20 Hz

    rospy.loginfo("Publishing fake Odometry and TF (odom -> base_link) at origin.")

    while not rospy.is_shutdown():
        current_time = rospy.Time.now()

        # --- Populate Odometry Message ---
        odom_msg.header.stamp = current_time

        # Pose
        odom_msg.pose.pose.position.x = static_x
        odom_msg.pose.pose.position.y = static_y
        odom_msg.pose.pose.position.z = 0.0 # Assuming 2D
        # Convert yaw angle (theta) to quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(static_theta / 2.0)
        q.w = math.cos(static_theta / 2.0)
        odom_msg.pose.pose.orientation = q
        # Set covariance (optional, can be zeros for fake data)
        odom_msg.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                    0, 0.1, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0.1] # Example covariance

        # Twist (Velocity)
        odom_msg.twist.twist.linear.x = static_vx
        odom_msg.twist.twist.linear.y = 0.0 # Assuming non-holonomic
        odom_msg.twist.twist.angular.z = static_vyaw
        # Set covariance (optional)
        odom_msg.twist.covariance = [0.1, 0, 0, 0, 0, 0,
                                     0, 0.1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0.1] # Example covariance

        # --- Populate TF Message ---
        tf_msg.header.stamp = current_time
        tf_msg.transform.translation.x = static_x
        tf_msg.transform.translation.y = static_y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = q

        # --- Publish ---
        odom_pub.publish(odom_msg)
        tf_broadcaster.sendTransform(tf_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        fake_odometry()
    except rospy.ROSInterruptException:
        pass

