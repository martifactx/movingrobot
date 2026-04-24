#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

class ReactiveNavigator:
    def __init__(self):
        rospy.init_node('reactive_navigator', anonymous=True)
        self.bridge = CvBridge()
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/m2wr/camera/image_raw', Image, self.camera_callback)
        self.laser_sub = rospy.Subscriber('/m2wr/laser/scan', LaserScan, self.laser_callback)
        
        self.twist = Twist()
        self.state = "FOLLOW_LINE"
        self.line_detected = False
        self.cx = 400
        self.last_cx = 400
        self.clear_ticks = 0
        
        # 5-Zone Spatial Awareness
        self.laser_right = 2.0
        self.laser_fright = 2.0
        self.laser_front = 2.0
        self.laser_fleft = 2.0
        self.laser_left = 2.0
        
        # Dynamic Decision Variables
        self.evasion_direction = "RIGHT"
        self.hug_side = "LEFT"
        
    def laser_callback(self, msg):
        ranges = [r if not np.isinf(r) and not np.isnan(r) else 2.0 for r in msg.ranges]
        
        if len(ranges) >= 720:
            # Divide the 180-degree view into 5 highly specific zones
            self.laser_right  = min(ranges[0:160]) if ranges[0:160] else 2.0
            self.laser_fright = min(ranges[160:300]) if ranges[160:300] else 2.0
            self.laser_front  = min(ranges[300:420]) if ranges[300:420] else 2.0
            self.laser_fleft  = min(ranges[420:560]) if ranges[420:560] else 2.0
            self.laser_left   = min(ranges[560:719]) if ranges[560:719] else 2.0
            
        # Trigger evasion if ANY front zone is blocked
        if self.state == "FOLLOW_LINE" and (self.laser_front < 0.6 or self.laser_fleft < 0.5 or self.laser_fright < 0.5):
            
            # DYNAMIC DECISION: Calculate open volume on both sides
            left_space = self.laser_fleft + self.laser_left
            right_space = self.laser_fright + self.laser_right
            
            if left_space > right_space:
                self.evasion_direction = "LEFT"
                self.hug_side = "RIGHT"
            else:
                self.evasion_direction = "RIGHT"
                self.hug_side = "LEFT"
                
            self.state = "TURN_AWAY"
            rospy.loginfo("Obstacle! More space on the {}. Evading {}.".format(self.evasion_direction, self.evasion_direction))

    def camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return

        height, width, _ = cv_image.shape
        crop_img = cv_image[height//2:height, 0:width]
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))

        M = cv2.moments(mask)
        if M['m00'] > 0:
            self.cx = int(M['m10']/M['m00'])
            self.line_detected = True
            cv2.circle(crop_img, (self.cx, 50), 20, (0, 255, 0), -1)
        else:
            self.line_detected = False

        cv2.imshow("Robot View", crop_img)
        cv2.waitKey(3)

    def control_loop(self):
        rate = rospy.Rate(20) 
        
        while not rospy.is_shutdown():
            if self.state == "FOLLOW_LINE":
                if self.line_detected:
                    error = self.cx - 400
                    self.twist.linear.x = 0.35
                    self.twist.angular.z = -float(error) / 400.0
                    self.last_cx = self.cx 
                else:
                    self.twist.linear.x = 0.1 
                    self.twist.angular.z = -0.4 if self.last_cx > 400 else 0.4 
                    
            elif self.state == "TURN_AWAY":
                self.twist.linear.x = 0.0
                # Spin in the dynamically chosen direction
                self.twist.angular.z = 0.8 if self.evasion_direction == "LEFT" else -0.8
                
                # Wait for the front to clear
                if self.laser_front > 0.8 and self.laser_fleft > 0.6 and self.laser_fright > 0.6:
                    self.state = "PASS_OBSTACLE"
                
            elif self.state == "PASS_OBSTACLE":
                # DYNAMIC MIRRORING: Wall Hugging logic adapts to which side the wall is on
                if self.hug_side == "LEFT":
                    if self.laser_front < 0.5:
                        self.twist.linear.x = 0.0; self.twist.angular.z = -0.8
                    elif self.laser_fleft < 0.45:
                        self.twist.linear.x = 0.2; self.twist.angular.z = -0.6
                    elif self.laser_left < 0.4:
                        self.twist.linear.x = 0.3; self.twist.angular.z = -0.3
                    elif self.laser_left > 0.65:
                        self.twist.linear.x = 0.3; self.twist.angular.z = 0.3
                    else:
                        self.twist.linear.x = 0.4; self.twist.angular.z = 0.0
                        
                    if self.laser_left > 1.2 and self.laser_fleft > 1.2:
                        self.state = "CLEAR_TAIL"; self.clear_ticks = 0

                elif self.hug_side == "RIGHT":
                    if self.laser_front < 0.5:
                        self.twist.linear.x = 0.0; self.twist.angular.z = 0.8
                    elif self.laser_fright < 0.45:
                        self.twist.linear.x = 0.2; self.twist.angular.z = 0.6
                    elif self.laser_right < 0.4:
                        self.twist.linear.x = 0.3; self.twist.angular.z = 0.3
                    elif self.laser_right > 0.65:
                        self.twist.linear.x = 0.3; self.twist.angular.z = -0.3
                    else:
                        self.twist.linear.x = 0.4; self.twist.angular.z = 0.0
                        
                    if self.laser_right > 1.2 and self.laser_fright > 1.2:
                        self.state = "CLEAR_TAIL"; self.clear_ticks = 0
                    
            elif self.state == "CLEAR_TAIL":
                self.twist.linear.x = 0.4; self.twist.angular.z = 0.0
                self.clear_ticks += 1
                if self.clear_ticks >= 25: 
                    self.state = "RETURN_TO_LINE"
                
            elif self.state == "RETURN_TO_LINE":
                self.twist.linear.x = 0.4
                # Turn back in the opposite direction of the original evasion
                self.twist.angular.z = 0.25 if self.hug_side == "LEFT" else -0.25 
                
                # Active re-scanning: If another obstacle appears, recalculate evasion!
                if self.laser_front < 0.6 or self.laser_fleft < 0.5 or self.laser_fright < 0.5:
                    self.state = "FOLLOW_LINE" # Let the main loop handle the new math
                
                if self.line_detected:
                    self.state = "FOLLOW_LINE"
                    rospy.loginfo("Line Locked!")

            self.cmd_vel_pub.publish(self.twist)
            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = ReactiveNavigator()
        navigator.control_loop()
    except rospy.ROSInterruptException:
        pass