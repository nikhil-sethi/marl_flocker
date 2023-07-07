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


Saved policy: Episode 5040 done | reward: -18.49044 | Avg reward: -19.82541 | act loss: 8.81007 | q loss: 0.60087 

Takes a lot of time to converge
because the reward function doesn't penalise the drone magnitude, the result policy just makes both drones have very less velocity
in principle this should not be possible because I set the magnitude to '2' always. But the algo is clever, and finds its way around by having the actions oscillate and make the net velocity positive in discrete time.

Because the drones are almost hovering (1 drone moves but very less though), the roll pitch is minimized and velocity difference is also high enough. (still not maximized though)
It is also possible that internally at each discrete step, the velocity difference is being maximized by oscillation but we can't see it.