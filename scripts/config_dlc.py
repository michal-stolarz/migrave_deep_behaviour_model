t_steps = 500

raw_frame_height = 320
raw_frame_width = 240
proc_frame_size = 198
state_size_data = 8
state_size_model = 1
t_episodes = 14
actions = ['1', '2', '3', '4']
device = "cpu"  # cuda
minibatch_size = 25
learning_rate = 0.001
optimizer = "adam"
epochs = 12 #6 for dlc1, 20 for dlc8
labels = ['diff2', 'diff3', 'feedback']

# network
noutputs = 2
nfeats = 1
nfeats_full = 8 # for the model taking 8 frames as an input
nstates = [8, 16, 32, 128] # for input of 1 frame
nstates_full = [8, 16, 32, 128] # for input of 8 frames
# nstates_full = [16, 32, 64, 256] # old, for input of 8 frames
kernels = [9, 5]
strides = [3, 1]
poolsize = 2
