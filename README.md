# Machine Learning for Movement Continuation #

Daniel Bisig - Instituto Stocos, Spain - daniel@stocos.com, Zurich University of the Arts, Switzerland - daniel.bisig@zhdk.ch

### Overview ###

This repository provides the source code to interact with a machine learning model that has been pre-trained on dance movements. The model is an recurrent neural network that is implemented using the Pytorch development framework. Recurrent neural networks belongs to a class of models that can learn temporal sequences of data.  This model can be used to predict how a given dance movement could continue over the next few minutes. 

When running the source code, the model continuously predicts the next dance pose that might follow a given movement sequence. After the next dance pose has been predicted, the movement sequence is updated by removing the oldest pose and appending the predicted pose. Users can interact with the model either by choosing a different movement sequence from a motion capture recording that serves as starting point for the pose predictions. Or they can overwrite some of the joints in a movement sequence before the pose prediction takes place. By doing so, users can combine movements they create on their own with those that the model has learned and thereby create new movement material.

## OSC-Communication

While the model is running,  it communicates with other software applications by using the OSC (Open Sound Control) protocol. By sending OSC messages to the model, the altering of the encodings can be controlled. In turn, the model sends the newly generated movements via OSC to other applications. 

#### The following OSC messages can be sent to the model:

- /mocap/inputseq <Int> : set the start position in a motion capture recording from which the first movement sequence is obtained
- /mocap/jointrot <Int><Int> ... <Int> <Float> <Float> <Float> <Float> : set a new rotation (as quaternion)  for a list of selected joints. 

#### The following OSC messages are send by the model:

- /mocap/joint/pos_world <Float> <Float> .... <Float> : contains the 3D positions of all joints in world coordinates for the currently generated pose
- /mocap/joint/rot_world <Float> <Float> .... <Float> : contains the 3D rotations (as quaternions) of all joints in world coordinates for the currently generated pose



The code has been tested on Windows and MacOS. Anaconda environments for these two operating systems are provided as part of this repository. 