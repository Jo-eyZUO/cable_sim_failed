#%%
# Set up GPU rendering.
import distutils.util
import os
import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU 180:')

# Other imports and helper functions
import time
import itertools
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from IPython.display import clear_output
clear_output()

from ikfastpy import ikfastpy
import mujoco
import mujoco.viewer
# ------------------------------------

# default gravity [ 0.    0.   -9.81]
model = mujoco.MjModel.from_xml_path("ur5_robotiq_f285/UR5_robotiq_f285.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# print('Total number of DoFs in the model:', model.nv)
# print('Generalized positions:', data.qpos)
# print('Generalized velocities:', data.qvel)

# print('Ctrl:', data.ctrl) [0, -1.3, 2., -2.27, -1.57, -1.57, 0] 
# [0, -1.3, 2.1, -2.37, -1.57, -1.57 ]
joint_angles = [0, -1.3, 2., -2.27, -1.57, -1.57, 0] # in radians
data.ctrl = joint_angles

# print('default gravity', model.opt.gravity)

# with mujoco.Renderer(model) as renderer:
#     mujoco.mj_forward(model, data)
#     renderer.update_scene(data, camera = 'top_down')
#     media.show_image(renderer.render())
    
with mujoco.viewer.launch_passive(model, data) as viewer:
    count = 0
    data.qpos[-14:-8] = [0, -1.3, 2.15, -2.42, -1.57, -1.57 ] 
    mujoco.mj_step(model, data)
    while viewer.is_running():
        count += 1
        # =============== pos hard control ================  
        # print('Generalized positions:', data.qpos)
        
        # =============== joint control ================  
        # if count < 1000:
        #     data.ctrl = [0, -1.3, 2.15, -2.42, -1.57, -1.57 , 180] 
        # if count > 1000 and count < 2000:
        #     data.qpos[-14:-8] = [0, -1.3, 1.5, -1.77, -1.57, -1.57] 
        # if count > 2000:
        #     break
        if count < 500:
            data.ctrl = [0, -1.3, 2.15, -2.42, -1.57, -1.57 , 190] 
            mujoco.mj_step(model, data)
            viewer.sync()
        else:
            data.ctrl = [0, -1.3, 1.5, -1.77, -1.57, -1.57, 190] 
            mujoco.mj_step(model, data)
            if count > 1000:
                break
        mujoco.mj_step(model, data)
        viewer.sync()
        

#%%
# =================================================================================
# Initialize kinematics for UR5 robot arm
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

# joint_angles = [0,-1.6,1.6,-1.6,-1.6,0.] # in radians

# # Test forward kinematics: get end effector pose from joint angles
# print("\nTesting forward kinematics:\n")
# print("Joint angles:")
# print(joint_angles)
# ee_pose = ur5_kin.forward(joint_angles)
# ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix

# print("\nEnd effector pose:")
# print(ee_pose)
# print("\n-----------------------------")

ee_pose = np.asarray([0.001, 1. ,-0.029, -0.5, 1,0,0.029,0 , 0.029,-0.029, -1,0.5]).reshape(3,4)
print("\nEnd effector pose:")
print(ee_pose)
# Test inverse kinematics: get joint angles from end effector pose
print("\nTesting inverse kinematics:\n")
joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)

# Check cycle-consistency of forward and inverse kinematics
# assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
# print("\nTest passed!")
# ==================================================================================

#%%
# Set up GPU rendering.
import distutils.util
import os
import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU 180:')

# Other imports and helper functions
import time
import itertools
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from IPython.display import clear_output
clear_output()

from ikfastpy import ikfastpy
import mujoco
import mujoco.viewer
# ------------------------------------

# default gravity [ 0.    0.   -9.81]
model = mujoco.MjModel.from_xml_path("ur5_robotiq_f285/UR5_robotiq_f285.xml")
data = mujoco.MjData(model)

# Episode parameters.
duration = 3       # (seconds)
framerate = 60     # (Hz)
data.qpos[-14:-8] = [0, -1.3, 2.15, -2.42, -1.57, -1.57 ]   # Initial x-y position (m)

# Visual options for the "ghost" model.
vopt2 = mujoco.MjvOption()
vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
# We only want dynamic objects (the humanoid). Static objects (the floor)
# should not be re-drawn. The mjtCatBit flag lets us do that, though we could
# equivalently use mjtVisFlag.mjVIS_STATIC
catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

# Simulate and render.
frames = []
with mujoco.Renderer(model, 480, 640) as renderer:
  while data.time < duration:
    # control signal.
    data.ctrl = [0, -1.3, 2.15, -2.42, -1.57, -1.57 , 190] 
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      # This draws the regular humanoid from `data`.
      renderer.update_scene(data)
      # Render and add the frame.
      pixels = renderer.render()
      frames.append(pixels)

# Render video at half real-time.
media.show_video(frames, fps=framerate/2)


#%%
# Set up GPU rendering.
import distutils.util
import os
import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU 180:')

# Other imports and helper functions
import time
import itertools
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from IPython.display import clear_output
clear_output()

from ikfastpy import ikfastpy
import mujoco
import mujoco.viewer
# ------------------------------------

# default gravity [ 0.    0.   -9.81]
model = mujoco.MjModel.from_xml_path("ur5_robotiq_f285/UR5_robotiq_f285.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# print('Total number of DoFs in the model:', model.nv)
# print('Generalized positions:', data.qpos)
# print('Generalized velocities:', data.qvel)

# print('Ctrl:', data.ctrl) [0, -1.3, 2., -2.27, -1.57, -1.57, 0] 
# [0, -1.3, 2.1, -2.37, -1.57, -1.57 ]
joint_angles = [0, -1.3, 2., -2.27, -1.57, -1.57, 0] # in radians
data.ctrl = joint_angles

# print('default gravity', model.opt.gravity)

# with mujoco.Renderer(model) as renderer:
#     mujoco.mj_forward(model, data)
#     renderer.update_scene(data, camera = 'top_down')
#     media.show_image(renderer.render())
    
with mujoco.viewer.launch_passive(model, data) as viewer:
    count = 0
    data.qpos[-14:-8] = [0, -1.3, 1.5, -1.77, -1.57, -1.57 ] 
    
    mujoco.mj_step(model, data)
    while viewer.is_running():
        # =============== pos hard control ================  
        # print('Generalized positions:', data.qpos)
        
        # =============== joint control ================  
        # if count < 1000:
        #     data.ctrl = [0, -1.3, 2.15, -2.42, -1.57, -1.57 , 180] 
        # if count > 1000 and count < 2000:
        #     data.qpos[-14:-8] = [0, -1.3, 1.5, -1.77, -1.57, -1.57] 
        # if count > 2000:
        #     break
        for i in range(5000):
            data.ctrl = [0, -1.3, 2.15, -2.42, -1.57, -1.57 , 0] 
          
            mujoco.mj_step(model, data)
            viewer.sync()
        for i in range(5000):
            data.ctrl = [0, -1.3, 2.15, -2.42, -1.57, -1.57 , 185] 
          
            mujoco.mj_step(model, data)
            viewer.sync()
        # data.qpos[-14:-8] = [0, -1.3, 1.5, -1.77, -1.57, -1.57 ]
        for j in range(5000):
          data.ctrl = [0, -1.3, 1.5, -1.77, -1.57, -1.57, 190] 
          mujoco.mj_step(model, data)
          viewer.sync()
            # data.ctrl = [0, -1.3, 1.5, -1.77, -1.57, -1.57, 190] 
            # mujoco.mj_step(model, data)
            # viewer.sync()
        break

        
#%%