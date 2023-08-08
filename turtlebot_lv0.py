from .base_env import RobotGazeboEnv
import rospy
import subprocess
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import gymnasium as gym
import tf
import time
import os


class TurtlebotLv0(RobotGazeboEnv):
    def __init__(self, ):
        port = "11311"
        launcher_file = os.path.dirname(__file__)+"/launch/TurtlebotLv0.launch"
        world_file = os.path.dirname(__file__)+"/world/custom.world"
        subprocess.Popen(["roscore", "-p", port])
        rospy.init_node("ROSGymnasium")
        subprocess.Popen(["roslaunch","-p", port, launcher_file, "world_name:="+world_file,])

        super(TurtlebotLv0, self).__init__(start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()

        # set up some publishers ans subscribers
        print("Set up publisher and subscriber")
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        self._check_all_systems_ready()
        self.gazebo.pauseSim()

        # Action: linear speed, angular speed
        self.action_space = gym.spaces.Box(np.array([-1, -1]),
                                           np.array([1, 1]),
                                           (2,), np.float32)
        
        # Observation: {"odom":[goal x, goal y, v, w], "lidar"}
        odom_obs_space = gym.spaces.Box(np.array([-np.inf, -np.inf, 0, -1]),
                                        np.array([np.inf, np.inf, 1, 1]),
                                        (4,), np.float32)
        lidar_obs_space = gym.spaces.Box(self.laser_scan.range_min,
                                         self.laser_scan.range_max,
                                         (len(self.laser_scan.ranges),), np.float32)
        
        self.observation_space = gym.spaces.Dict({"odom":odom_obs_space,
                                                  "lidar":lidar_obs_space,})
        
        # scaling for the robot
        self.linear_speed = 1
        self.angular_speed = 1


    def _odom_callback(self, data):
        self.odom = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _check_all_systems_ready(self):
        self.odom = None
        self.laser_scan = None
        try:
            self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
            rospy.logdebug("/odom READY=>")
        except:
            rospy.logerr("/odom not ready yet, retrying for getting odom")

        try:
            self.laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=5.0)
            rospy.logdebug("/scan READY=>")
        except:
            rospy.logerr("/scan not ready yet, retrying for getting laser scan")
        return
    
    def _set_init_pose(self):
        _init_cmd_vel = Twist()
        self.vel_pub.publish(_init_cmd_vel)

        return
    
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        self.reset_goal()

        # Environment Specific variables
        self.collision_distance = 0.15
        self.goal_reach_threshold = 0.15
        self.collided = False
        self.goal_reached = False
        self.last_goal_dist = self.cal_goal_r()

        return 

    def _get_obs(self):
        '''
        Get the observation, which include
        1. Coordinate of the goal in the robot's reference frame
        2. linear and angular speed of the robot
        3. lidar readings
        Return in gymnasium.spaces.Dict with keys "odom" and "lidar"
        '''


        # Get the desired position in the bot's frame (Cartesian)
        rel_x, rel_y = self.desired_point.x-self.odom.pose.pose.position.x,\
                       self.desired_point.y-self.odom.pose.pose.position.y
        
        r = np.sqrt(rel_x**2+rel_y**2)
        theta = np.arctan2(rel_y, rel_x)
        quaternion = self.odom.pose.pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
        theta -= yaw

        goal_x, goal_y = r*np.cos(theta), r*np.sin(theta)


        odom = np.array([goal_x, goal_y, self.odom.twist.twist.linear.x, self.odom.twist.twist.angular.z],
                        dtype=np.float32)
        lidar = np.array(self.laser_scan.ranges, dtype=np.float32)
        # clip all ranges beyond range_max
        lidar[lidar>=self.laser_scan.range_max] = self.laser_scan.range_max

        return {"odom":odom, "lidar":lidar}
    
    def _set_action(self, action):
        """
        publish the action to the robot through cmd_vel
        then forcing the simulator to run for 0.1s
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = (action[0]+1)/2.*self.linear_speed
        cmd_vel_value.angular.z = action[1]*self.angular_speed
        self.vel_pub.publish(cmd_vel_value)
        time.sleep(1./60.)

        return
    
    def _compute_reward(self, observations, done):
        """
        Reward function
        r = 2.5*(Dist(t-1)-Dist(t)) if not collided and not goal_reached
        r = 10 if goal_reached
        r = -10 if collided
        """
        if not done:
            new_goal_dist = self.cal_goal_r()
            reward = 2.5*(self.last_goal_dist - new_goal_dist)
            self.last_goal_dist = new_goal_dist
        elif self.collided:
            reward = -10
        elif self.goal_reached:
            reward = 10
        return reward
    
    def _is_done(self, observations):
        if self.check_collision():
            rospy.logwarn("Collided.")
            return True
        elif self.reach_goal():
            rospy.logwarn("Goal Reached.")
            return True
        else:
            return False
    
    def reach_goal(self):
        bot_pos_vec = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y])
        goal_pos_vec = np.array([self.desired_point.x, self.desired_point.y])
        if np.linalg.norm(bot_pos_vec-goal_pos_vec) <= self.goal_reach_threshold:
            self.goal_reached=True
            return True
        else:
            self.goal_reached=False
            return False
        
    def check_collision(self):
        if np.any(np.array(self.laser_scan.ranges) < self.collision_distance):
            self.collided = True
            return True
        else:
            self.collided = False
            return False
        
    def reset_goal(self):
        # reset goal position
        self.desired_point =  Point()
        theta = np.random.rand()*2*np.pi
        self.desired_point.x = (3*np.random.rand()+1)*np.cos(theta)
        self.desired_point.y = (3*np.random.rand()+1)*np.sin(theta)
        MSG = "New goal position set at " +  str(self.desired_point.x) + " , " + str(self.desired_point.y)
        rospy.logwarn(MSG)

        self.gazebo.unpauseSim()
        self.gazebo.model_config("goal_area", [self.desired_point.x, self.desired_point.y, 0])
        self.gazebo.pauseSim()

        return

    def cal_goal_r(self,):
        return np.sqrt((self.desired_point.x-self.odom.pose.pose.position.x)**2 +\
                       (self.desired_point.y-self.odom.pose.pose.position.y)**2)
