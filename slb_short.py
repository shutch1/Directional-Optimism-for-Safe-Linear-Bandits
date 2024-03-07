
#%%
import numpy as np
import math
import safe_linear_bandit
import time

#%%

# seed randomness
prob_rng = np.random.default_rng(seed=0)

# setup trials
print_interval = 2000
num_trials = 30
T = int(5e4)
reg_roful_trials = np.zeros((num_trials, T))
reg_gnop_trials = np.zeros((num_trials, T))
viol_roful_trials = np.zeros(num_trials)
viol_gnop_trials = np.zeros(num_trials)

# run trials
for trial in range(num_trials):
    print('Starting Trial', trial)
    
    # problem parameters
    d = 2
    theta = prob_rng.uniform(-1,1,(d,1))
    a = prob_rng.uniform(-1,1,(d,1))
    xlim = 1
    K = 10
    b = prob_rng.uniform(0.25,1)
    sigma = 1e-1

    # sample directions until some are positive
    invalid = True
    while invalid:
        temp = prob_rng.normal(0,1,size=(K,d)) 
        dirs = temp / np.linalg.norm(temp,axis=1)[:,np.newaxis]
        if np.any(dirs @ theta > 0):
            invalid = False

    # algorithm paramaters
    lamb = 1
    delta = 0.01
    S = math.sqrt(d)
    nu = b/S
    beta = lambda t : sigma*math.sqrt(d*math.log((1 + t/ lamb)/(delta/2))) + math.sqrt(lamb) * S

    # create problems for each algorithm with identical randomness
    noises = prob_rng.standard_normal((T,2))
    noise_seq_t = noises[:,0]
    noise_seq_a = noises[:,1]
    prob_roful = safe_linear_bandit.safe_linear_bandit(d,theta,a,b,T,sigma,noise_seq_t, noise_seq_a, xlim, dirs)
    prob_gnop = safe_linear_bandit.safe_linear_bandit(d,theta,a,b,T,sigma,noise_seq_t, noise_seq_a, xlim, dirs)

    # initialize variables for each simulation
    V_roful = lamb*np.eye(d)
    V_gnop = lamb*np.eye(d)
    Dt_roful = np.zeros((d,1))
    Dt_gnop = np.zeros((d,1))
    Da_roful = np.zeros((d,1))
    Da_gnop = np.zeros((d,1))

    # play problems
    print('Beginning exploration.')
    start_time = time.time()
    loop_start_time = start_time
    for t in range(T):
        # algs choose actions
        beta_t = beta(t)
        x_roful = safe_linear_bandit.roful(V_roful, Dt_roful, Da_roful, beta_t, S, b, xlim, dirs, d)
        x_gnop = safe_linear_bandit.gnop(V_gnop, Dt_gnop, Da_gnop, beta_t, S, b, xlim, dirs, d)

        # play actions and recieve feedback
        (r_roful, c_roful) = prob_roful.play_action(x_roful)
        (r_gnop, c_gnop) = prob_gnop.play_action(x_gnop)

        # update variables
        V_roful = V_roful + x_roful @ x_roful.T
        Dt_roful = Dt_roful + x_roful * r_roful
        Da_roful = Da_roful + x_roful * c_roful

        V_gnop = V_gnop + x_gnop @ x_gnop.T
        Dt_gnop = Dt_gnop + x_gnop * r_gnop
        Da_gnop = Da_gnop + x_gnop * c_gnop

        if t % print_interval == 0:
            print('step:',t,'average loop time:',(time.time() - loop_start_time)/print_interval)
            loop_start_time = time.time()

    final_time = time.time()
    print('Total time:',final_time-start_time)

    reg_roful_trials[trial,:] = prob_roful.true_optim - prob_roful.rew_rec
    reg_gnop_trials[trial,:] = prob_gnop.true_optim - prob_gnop.rew_rec
    viol_roful_trials[trial] = np.sum(prob_roful.viol_rec)
    viol_gnop_trials[trial] = np.sum(prob_gnop.viol_rec)

np.save('short_roful_reg',reg_roful_trials)
np.save('short_gnop_reg',reg_gnop_trials)
np.save('short_roful_viol',viol_roful_trials)
np.save('short_gnop_viol',viol_gnop_trials)

# %%
