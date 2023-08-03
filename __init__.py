from gymnasium.envs.registration import register

register(id="TurtlebotGoal-v0",
         entry_point="gazebo_gymnasium.turtlebot_lv0:TurtlebotLv0",
         max_episode_steps = 1000,
        )

