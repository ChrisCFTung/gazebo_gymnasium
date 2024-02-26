from .base_env import RobotGazeboEnv
import rospy
import subprocess
from gazebo_msgs.msg import ModelState, ContactsState
from geometry_msgs.msg import Point, Twist, PoseStamped, PoseWithCovarianceStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.srv import GetPlan
import numpy as np
import gymnasium as gym
import tf
import time
# import bezier
import os


class TurtlebotLv1(RobotGazeboEnv):
    def __init__(self, ):
        port = "11311"
        launcher_file = os.path.dirname(__file__)+"/launch/TurtlebotLv1.launch"
        world_file = os.path.dirname(__file__)+"/world/turtlebot3_house.world"
        map_file = os.path.dirname(__file__)+"/world/turtlebot3_house.yaml"
        subprocess.Popen(["roscore", "-p", port])
        rospy.init_node("ROSGymnasium")
        subprocess.Popen(["roslaunch","-p", port, launcher_file, "world_name:="+world_file,
                          "map_file:="+map_file,])

        super(TurtlebotLv1, self).__init__(start_init_physics_parameters=False,
                                           reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()

        # set up some publishers ans subscribers
        print("Set up publisher and subscriber")
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self._pose_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self._grid_callback)
        rospy.Subscriber("/move_base/local_costmap/costmap_updates", OccupancyGridUpdate, self._gridupdate_callback)
        rospy.Subscriber("/bumper", ContactsState, self._bumper_callback)
        #rospy.topics.Subscriber("/move_base/NavfnROS/plan", Path, self._global_plan_callback)
        self.global_planner = rospy.ServiceProxy("/move_base/NavfnROS/make_plan", GetPlan)

        self._check_all_systems_ready()
        self.gazebo.pauseSim()

        # Action: linear speed, angular speed
        self.action_space = gym.spaces.Box(np.array([-1, -1]),
                                           np.array([1, 1]),
                                           (2,), np.float32)
        
        # Observation: {"goal":[goal x, goal y], "odom":[posx, posy, yaw, v, w], "grid":[width, length]}
        goal_obs_space =  gym.spaces.Box(np.array([-np.inf, -np.inf]),
                                         np.array([np.inf, np.inf]),
                                         (2,), np.float32)
        odom_obs_space = gym.spaces.Box(np.array([-np.inf, -np.inf, -np.pi, 0, -1]),
                                        np.array([np.inf, np.inf, np.pi, 1, 1]),
                                        (5,), np.float32)
        grid_obs_space = gym.spaces.Box(0,1,
                                        (self.grid.info.width, self.grid.info.height), 
                                        np.float32)
        
        self.observation_space = gym.spaces.Dict({"goal":goal_obs_space,
                                                  "odom":odom_obs_space,
                                                  "grid":grid_obs_space,})
        
        # scaling for the robot
        self.linear_speed = 1
        self.angular_speed = 1


    def _odom_callback(self, data):
        self.odom = data

    def _pose_callback(self, data):
        self.pose = data.pose.pose

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _grid_callback(self, data):
        self.grid = data #np.array([float(x) for x in data.data]).reshape((data.info.width, data.info.height))
        #self.grid /= 100.

    def _gridupdate_callback(self, data):
        self.gridupdate = data.data

    def _bumper_callback(self, data):
        if data.states:
            self.collided = True
            self.contact = data

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

        try:
            rospy.wait_for_message("/move_base/local_costmap/costmap", OccupancyGrid, timeout=10.0)
            rospy.logdebug("costmap ready")
        except:
            rospy.logerr("Can't receive costmap")
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
        # self.cumulated_reward = 0.0
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
        1. Coordinate of the goal (map frame)
        2. position and yaw of the robot (map frame) + velocities (local frame)
        3. Local costmap
        Return in gymnasium.spaces.Dict with keys "goal", "odom" and "grid"
        '''


        # Get the goal
        goal = np.array([self.desired_point.x, self.desired_point.y])
        
        # Get the position, yaw, linear velocity and angular velocity
        quaternion = self.pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
        odom = np.array([self.pose.position.x, self.pose.position.y, yaw, 
                         self.odom.twist.twist.linear.x, self.odom.twist.twist.angular.z],
                        dtype=np.float32)
        
        # Get the occupancy grid
        grid = self.get_costmap()
        
        return {"goal":goal, "odom":odom, "grid":grid}
    
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
        # collision is decided by the bumper
        return self.collided
        
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
    
    def get_costmap(self,):
        self.grid.data = self.gridupdate
        return self.grid

    def get_global_plan(self, startp, endp):
        """
        call the service from NavfnROS to generate a global plan
        start_point: numpy array of [x,y, yaw]
        end_point: numpy array of [x,y, yaw]

        the plan will be a path stored in self.global_plan
        """
        start = PoseStamped()
        end = PoseStamped()
        start.pose.position = Point(startp[0], startp[1], 0)
        start.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, startp[2]))
        start.header.frame_id = "map"

        end.pose.position = Point(endp[0], endp[1], 0)
        end.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, endp[2]))
        end.header.frame_id = "map"

        self.global_plan = self.global_planner(start=start, goal=end, tolerance=0.2)

    def get_local_trajectory(self, control):
        """
        find curve to connect robot to the first way point from its position
        """

    def get_waypoint(self, distance=0.3):
        """
        find the first way point, which is at a distance of r from the robot
        in the global plan
        """