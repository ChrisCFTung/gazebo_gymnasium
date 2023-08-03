# A gymnasium environment simulated with GAZEBO

Required installation: ROS, gymnasium

Basic idea:
1. Define the robot and world model in a roslaunch file
2. Launch the file in a python script and create the gymnaisum environment
3. Register the environment to the gymnasium registry
4. create env through gymnasium.make

## base_env.py
define the base class for the gazebo gymnasium environment class

## gazebo_connection.py
copied from openai_ros, provide methods to interact with gazebo

## turtlebot_lv0.py
template environment, task is controlling a turtlebot to reach a random goal position
inside a walled region