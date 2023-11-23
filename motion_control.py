import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "input_length": 64,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.input_length = config["input_length"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/inputseq", self.setInputSequence)
        self.dispatcher.map("/mocap/jointrot", self.setJointRotation)
    
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def setInputSequence(self, address, *args):

        if len(args) == 1: # start frame index
        
            seq_start_index = args[0]

            total_seq_length = self.motion_seq.shape[0] 
            
            seq_start_index = min(seq_start_index, total_seq_length - self.input_length)
            seq_end_index = seq_start_index + self.input_length
            
            input_seq = self.motion_seq[seq_start_index:seq_end_index, ...]
            
            #self.gui.stop()
            self.synthesis.setMotionSequence(input_seq)
            
        elif len(args) == 2: # frame index, frame count
                
            seq_start_index = args[0]
            seq_frame_count = args[1]
            
            total_seq_length = self.motion_seq.shape[0] 
            
            seq_frame_count = min(seq_frame_count, self.input_length)
            seq_start_index = min(seq_start_index, total_seq_length - seq_frame_count)
            seq_end_index = seq_start_index + seq_frame_count
            
            if seq_frame_count == self.input_length:
                input_seq = self.motion_seq[seq_start_index:seq_end_index, ...]
            else: 
                input_seq = self.motion_seq[seq_start_index:seq_end_index, ...]
                
            #input_seq =  np.concatenate( (self.motion_seq[seq_start_index:seq_end_index, ...], input_seq[seq_frame_count:, ...]), axis=0)
            
            input_seq =  np.concatenate( (input_seq[:seq_frame_count, ...], self.motion_seq[seq_start_index:seq_end_index, ...]), axis=0)
            
            self.synthesis.setMotionSequence(input_seq)
            
        
    def setJointRotation(self, address, *args):
        
        if len(args) == 5: # joint index, rotation axis, rotation angle
        
            joint_index = args[0]
            rot_axis = np.array([args[1], args[2], args[3]])
            rot_angle = args[4]
            
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            self.synthesis.setJointRotation(joint_index, rot_quat, 1)
            
        elif len(args) > 5: # joint indices rotation axis, rotation angle
        
            argcount = len(args)

            joint_index = args[0]
            rot_axis = np.array([args[-4], args[-3], args[-2]])
            rot_angle = args[-1]

            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            for joint_index in args[:-4]:
                
                self.synthesis.setJointRotation(joint_index, rot_quat, 1)