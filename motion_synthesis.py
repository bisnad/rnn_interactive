import torch
from torch import nn
import numpy as np

from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix

config = {"skeleton": None,
          "model": None,
          "device": "cuda",
          "motion_seq": None
          }

class MotionSynthesis():
    
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model = config["model"]
        self.device = config["device"]
        self.motion_seq = torch.from_numpy(config["motion_seq"]).to(self.device)
        
        self.seq_length = self.motion_seq.shape[0]
        self.joint_count = self.motion_seq.shape[1]
        self.joint_dim = self.motion_seq.shape[2]
        self.pose_dim = self.joint_count * self.joint_dim

        self.joint_offsets = self.skeleton ["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton ["parents"]
        self.joint_children = self.skeleton ["children"]
        
        self._create_edge_list()
        
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        
    def _create_edge_list(self):
        
        self.edge_list = []
        
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])
                
    def setMotionSequence(self, motion_seq):
        self.motion_seq = torch.from_numpy(motion_seq).to(self.device)
        
    def setJointRotation(self, joint_index, joint_rot, frame_count):
        
        #print("setJointRotation index ", joint_index, " rot ", joint_rot)
        
        """
        seq_length = self.motion_seq.shape[0]
        joint_rot = np.repeat(np.expand_dims(joint_rot, axis=0), seq_length, axis=0)
        
        joint_rot = torch.from_numpy(joint_rot).to(self.device)
        self.motion_seq[:, joint_index, :] = joint_rot
        """
        
        joint_rot = torch.from_numpy(joint_rot).to(self.device)
        
        if frame_count == 1:
            self.motion_seq[-1, joint_index, :] = joint_rot
        elif frame_count > 1:
            frame_count = min(frame_count, self.seq_length)
            joint_rot = torch.unsqueeze(joint_rot, dim=0).repeat(frame_count, 1)
            self.motion_seq[:frame_count, joint_index, :] = joint_rot
            
    
    def update(self):
        
        self.model.eval()
        
        with torch.no_grad():
            self.pred_pose = self.model(torch.unsqueeze(self.motion_seq.reshape(-1, self.pose_dim), axis=0))
                
        # normalize pred pose
        self.pred_pose = torch.squeeze(self.pred_pose)
        self.pred_pose = self.pred_pose.reshape((-1, 4))
        self.pred_pose = nn.functional.normalize(self.pred_pose, p=2, dim=1)
        self.pred_pose = self.pred_pose.reshape((1, self.joint_count, self.joint_dim))
            
        # append pred pose to sequence
        self.motion_seq = torch.cat([self.motion_seq[1:,:], self.pred_pose], axis=0)

        # convert quaternion pose to position pose
        zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
        zero_trajectory = zero_trajectory.to(self.device)
            
        self.synth_pose_wpos, self.synth_pose_wrot = self._forward_kinematics(torch.unsqueeze(self.pred_pose,dim=0), zero_trajectory)

        self.synth_pose_wpos = self.synth_pose_wpos.detach().cpu().numpy()
        self.synth_pose_wpos = self.synth_pose_wpos.reshape((self.joint_count, 3))
        
        self.synth_pose_wrot = self.synth_pose_wrot.detach().cpu().numpy()
        self.synth_pose_wrot = self.synth_pose_wrot.reshape((self.joint_count, 4))
        
        self.model.train()

                
    def _forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4
        
        toffsets = torch.tensor(self.joint_offsets).to(self.device)
        
        positions_world = []
        rotations_world = []

        expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], self.joint_offsets.shape[0], self.joint_offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for jI in range(self.joint_offsets.shape[0]):
            if self.joint_parents[jI] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self.joint_parents[jI]], expanded_offsets[:, :, jI]) \
                                       + positions_world[self.joint_parents[jI]])
                if len(self.joint_children[jI]) > 0:
                    rotations_world.append(qmul(rotations_world[self.joint_parents[jI]], rotations[:, :, jI]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(torch.Tensor([[[1.0, 0.0, 0.0, 0.0]]]).to(self.device))
                    
        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)

    
        
        

        
    
    