
#%%
import numpy as np
import math

#%% Safe Linear Bandit Class

class safe_linear_bandit:
    # Class for the safe linear bandit problem.

    def __init__(self, d, theta, a, b, T, sigma, noise_seq_t, noise_seq_a, xlim, dirs):
        self.d = d
        self.theta = theta
        self.a = a
        self.b = b
        self.T = T
        self.sigma = sigma
        self.noises_t = noise_seq_t*sigma
        self.noises_a = noise_seq_a*sigma
        self.t = 0
        self.dirs = dirs

        # find optimal action and reward
        K = np.shape(dirs)[0]
        max_val = -np.inf
        opt_act = None
        for i in range(K):
            dir = dirs[[i],:].T
            if a.T @ dir > 0:
                scale = min(b / (a.T @ dir), xlim/np.linalg.norm(dir))
            else:
                scale = xlim/np.linalg.norm(dir)
            rew = theta.T @ (scale*dir)
            if rew > max_val:
                max_val = rew
                opt_act = scale*dir

        self.true_optim = max_val
        self.optim_point = opt_act

        # initialize recording functions
        self.rew_rec = np.zeros(T)
        self.x_rec = np.zeros((T,d))
        self.viol_rec = np.zeros(T)

    def play_action(self,x):
        # function for playing action

        # check if time is out
        if self.t >= self.T:
            return None

        # reward and constraint values
        r = self.theta.T @ x
        r_noise = r + self.noises_t[self.t]
        c = self.a.T @ x
        c_noise = c + self.noises_a[self.t]

        # save data
        self.rew_rec[self.t] = r
        self.x_rec[self.t,:] = x.T
        self.viol_rec[self.t] = max(self.a.T @ x - self.b, 0)

        # update round counter
        self.t = self.t + 1

        # return constraint and reward with noises
        return r_noise, c_noise


#%% Algorithm functions

def roful(V, D_theta, D_a, beta_t, S, b, xlim, dirs, d):
    # function for play of roful algorithm
    nu = b/S
    
    V_inv = np.linalg.inv(V)
    theta_hat = V_inv @ D_theta
    a_hat = V_inv @ D_a

    # find best direction
    K = np.shape(dirs)[0]
    max_val = -np.inf
    opt_act = None
    for i in range(K):
        dir = dirs[[i],:].T
        if a_hat.T @ dir - beta_t *math.sqrt(dir.T @ V_inv @ dir) > 0:
            scale = min(b/(a_hat.T @ dir - beta_t *math.sqrt(dir.T @ V_inv @ dir)), xlim/np.linalg.norm(dir))
        else:
            scale = xlim/np.linalg.norm(dir)
        rew = scale*(theta_hat.T @ dir + beta_t*math.sqrt(dir.T @ V_inv @ dir))
        if rew > max_val:
            max_val = rew
            opt_act = scale*dir
    
    # check if zero is best
    if max_val < 0:
        opt_act = np.zeros((d,1))
    
    xtil_t = opt_act

    # find mu
    if a_hat.T @ xtil_t + beta_t *math.sqrt(xtil_t.T @ V_inv @ xtil_t) > 0:
        mu = min(b/(a_hat.T @ xtil_t + beta_t *math.sqrt(xtil_t.T @ V_inv @ xtil_t)), 1)
    else:
        mu = 1

    # find btil
    if np.all(xtil_t == np.zeros((d,1))):
        btil = 1
    else:
        btil = min(nu/np.linalg.norm(xtil_t),1)

    gamma = max(btil, mu)
    x_t = gamma*xtil_t
    return x_t

def gnop(V, D_theta, D_a, beta_t, S, b, xlim, dirs, d):
    # function for play of gnop algorithm
    kappa = 1 + 2*S/b
    V_inv = np.linalg.inv(V)
    theta_hat = V_inv @ D_theta
    a_hat = V_inv @ D_a

    # find best direction
    K = np.shape(dirs)[0]
    max_val = -np.inf
    opt_act = None
    for i in range(K):
        dir = dirs[[i],:].T
        if a_hat.T @ dir + beta_t *math.sqrt(dir.T @ V_inv @ dir) > 0:
            scale = min(b/(a_hat.T @ dir + beta_t *math.sqrt(dir.T @ V_inv @ dir)), xlim/np.linalg.norm(dir))
        else:
            scale = xlim/np.linalg.norm(dir)
        rew = scale*(theta_hat.T @ dir + kappa*beta_t*math.sqrt(dir.T @ V_inv @ dir))
        if rew > max_val:
            max_val = rew
            opt_act = scale*dir
    
    # check if zero is best
    if max_val < 0:
        opt_act = np.zeros((d,1))
    
    x_t = opt_act

    return x_t
