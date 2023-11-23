"""
same as rnn_ntf_pos_weighted_joints.py
but only for inference and interactive control and sending
"""


import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg

import os, sys, time, subprocess
import numpy as np
import math

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_data_path = "../../../Data/qualisys/bvh/polytopia_fullbody_take2.bvh"
mocap_valid_frame_range = [ 500, 9500 ]

#mocap_data_path = "../../../Data/xsens/bvh/MurielDancing-005.bvh"
#mocap_valid_frame_range = [ 50, 6400 ]

mocap_input_length = 64
mocap_fps = 50

"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

bvh_data = bvh_tools.load(mocap_data_path)
mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

pose_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)

total_sequence_length = pose_sequence.shape[0]
joint_count = pose_sequence.shape[1]
joint_dim = pose_sequence.shape[2]
pose_dim = joint_count * joint_dim


"""
Load Model
"""

motion_model.config = {
    "input_length": 64,
    "data_dim": pose_dim,
    "node_dim": 512,
    "layer_count": 2,
    "device": device,
    "weights_path": "../rnn/results_qualisys_polytopia_fullbody_take2/weights/rnn_weights_epoch_400"
    }

"""
motion_model.config = {
    "input_length": 64,
    "data_dim": pose_dim,
    "node_dim": 512,
    "layer_count": 2,
    "device": device,
    "weights_path": "../rnn/results_xSens_murieldancing_005/weights/rnn_weights_epoch_400"
    }
"""

model = motion_model.createModel(motion_model.config) 

"""
Setup Motion Synthesis
"""

synthesis_config  = motion_synthesis.config
synthesis_config["skeleton"] = mocap_data["skeleton"]
synthesis_config["model"] = model
synthesis_config["motion_seq"] = pose_sequence[1000:1000 + motion_model.config["input_length"], :, :]
synthesis_config["device"] = device

synthesis = motion_synthesis.MotionSynthesis(synthesis_config)


synthesis.setMotionSequence(pose_sequence[2000:2000 + motion_model.config["input_length"], :, :])
    
"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004

osc_sender = motion_sender.OscSender(motion_sender.config)


"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
OSC Control
"""

motion_control.config["motion_seq"] = pose_sequence
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["ip"] = "127.0.0.1"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()