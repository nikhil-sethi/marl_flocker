n_agents = 2
minibatch_size = 1024
buffer_size = 1000000
alpha = 0.001 # Actor learning rate
beta = 0.01   # Critic learning rate
episodes_before_train = 50
max_steps = 200
tau = 0.01

Reward function 
 - separate the two drones by giving constant -ve reward under a threshold
 - minimize roll pitch angles
 - maximise velocity norm
 - cohere two drones by minimizing the distance above a certain threshold

Saved policy: Episode 8840 done | reward: 278.31462 | Avg reward: 268.64789 | act loss: -29.47671 | q loss: 10.72358 

Takes long time to converge

It does seem to work but not very good. the drones do come close to each other from their initial positions, and separate when under threshold but only after colliding with each other 
which isn't the best thing
Post colliding they align well without an alginment reward even