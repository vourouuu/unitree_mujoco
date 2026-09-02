The simulation is main.py whith the preexisting scripts imported alongside state.py and footsteps_generator(use python3 -m Walking.main from /simulate_python).
The code implements the pipeline from chapter 10 with comments where things aren't ready.I added a lower frequency thread to handle the slow calculations of the mpc qp more realistically.

For the FootstepGenerator class i implemented the functions generate_footstep,project_kinematics,get_terrain_height,get_measurements .I used Raiberts Heuristic for generate_footsteps which is used online and accounts for current velocity.In project_kinematics i corrected for circular and lateral clamping.
For the StateEstimator class i implemented the function update which gets position,orientation,v,w from mj_data.sensordata and returns them.


Questions:
1 Hip distance restriction
2 Values in FootstepGenerator may need tuning
2 Ml inplementation alongside mpc
