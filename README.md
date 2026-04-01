# Senior Design for bachelor degree in NPU
## Subject: AUV Path Planning Algorithm Research Based on Reinforcement Learning
In this project, I complement TD3 as my basic algorithm to navigate AUV's trajectory, and simulate current conditions in self-established environment with real current data from CMEMS.
#### base
As my first try to establish a simple instructive environment for a 3D path planning RL algorithm, I realized a 500x500x500 cube space for AUV to explore and avoid obstacles in the "base" directory. Inside this simulation, start, target and obstacles positions can be freely modified as well as radii of obstacles. There are no vibrations or turbulence to affect AUV's performance, though randomization of start and target can be switched(it is better to randomize them in some certain mini cubes of the environment).
In details, th
