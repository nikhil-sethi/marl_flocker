n_agents = 2
minibatch_size = 1024
buffer_size = 1000000
alpha = 0.001 # Actor learning rate
beta = 0.01   # Critic learning rate
episodes_before_train = 50
max_steps = 100
tau = 0.01

Reward function 
 - separate the two drones by maximising velocity difference
 - minimize roll pitch angles
 - maximise velocity norm

Saved policy: Episode 2400 done | reward: 143.01459 | Avg reward: 139.24016 | act loss: -1.49231 | q loss: 0.11581 

Takes okayish time to converge
Now that velocity is also maximised the drones are forced to separate and not stay still. Check zero_vel_separate for more info