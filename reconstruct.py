import copy, random
import optparse, pathlib
import numpy as np
import pandas as pd
from scipy.special import logit as nplogit
from scipy.stats import dirichlet, halfnorm, binom, norm, invgamma
from sklearn.cluster import AgglomerativeClustering
import itertools

rng = np.random.RandomState(111)
parser = optparse.OptionParser()

parser.add_option('--Y',action = 'store', dest = 'Y_matrix')
parser.add_option('--m',action = 'store', dest = 'haplotype_number')
parser.add_option('--N_samples',action = 'store', dest = 'Samplesnumber')
parser.add_option('--N_burnin',action = 'store', dest = 'Burinnumber')

options, args = parser.parse_args()
N_samples = int(options.Samplesnumber)
N_burnin = int(options.Burinnumber)
m = int(options.haplotype_number)



def HaploSepCombi(Y, m, al = np.array([0,1])):
    """
    Parameters
    ----------
    Y: numpy array
        The observed outcome matrix.
    m: positive int
        Inner dimension parameter.
    al: numpy array
        Finite alphabet set of S.

    Returns
    -------
    numpy array
        The initialization of omega matrix.
 
    -------    
    This function help us to initialize the weight matrix by combinatorial algorithm.
    """
    T = Y.shape[1]
    al = np.array([0,1])
    cluster = AgglomerativeClustering(n_clusters = 2**m, affinity='euclidean', linkage='ward')
    clusterCut = cluster.fit_predict(Y)
    val = np.zeros((2**m, T))
    for i in range(2**m):
        ind_c = np.where(clusterCut == i)[0]
        if len(ind_c) > 1:
            val[i,] = np.mean(Y[ind_c,], axis = 0)
        else:
            val[i,] = Y[ind_c,]
    order_index = np.argsort(np.sum(np.abs(val), axis=1))
    val = val[order_index, :]
    val = val - np.array([val[0,]]*val.shape[0])
    val = np.maximum(val, al[0])
    val = np.minimum(val, al[1])

    omega = np.zeros((m, T))
    omega[0,] = (val[1,] - al[0])/(al[1] - al[0])
    Iter = 1
    for i in range(2,m+1):
        if i == m:
            combs = np.array(list(itertools.product(range(2), repeat=Iter)))
            alM = np.zeros((combs.shape[0], combs.shape[1]+1))
            alM[:,:-1] = combs
        else:
            alM = np.array(list(itertools.product(range(2), repeat=Iter+1)))
        omega_run = np.concatenate((omega[0:Iter, ], (np.ones(T) - np.sum(omega, axis=0)).reshape(1,T)), axis=0)
        val_run = alM.dot(omega_run)
        
        ind = []
        for j in range(0,val_run.shape[0]):
            val_order = np.argsort(np.sum(np.abs(np.array([val_run[j,]]*val.shape[0])-val), axis=1))
            
            
            flag = []
            for i in range(len(val_order)):
                if val_order[i] in ind:
                    flag.append(False)
                else:
                    flag.append(True)
            ind_new = np.array(val_order)[np.array(flag)][0]
            ind.append(ind_new)
        omega[Iter, ] = (np.delete(val, ind, axis=0)[0,] - al[0])/(al[1]-al[0])
        Iter = Iter+1
    
    reorder = np.flip(np.argsort(np.sum(omega, axis=1)))
    omega = omega[reorder,:]/np.sum(omega,axis=0)
    return omega

def S_initialization(Y, W_init):
    """
    Parameters
    ----------
    Y: numpy array
        The observed outcome matrix.
    W_init: numpy array
        The initial weight matrix

    Returns
    -------
    numpy array
        The initial design matrix.
    This function help us to initialize the design matrix by choosing the design
    matrix with the least MSE of obseved matrix and the product of design and 
    parameter matrix. 
    """
    S_init = np.zeros((Y.shape[0], W_init.shape[0]))
    for i in range (Y.shape[0]):
        y = Y[i,:]
        S_list = np.array(list(itertools.product(range(2), repeat=W_init.shape[0])))
        norm = np.mean(abs(y - S_list.dot(W_init)), axis=1)
        ind_s = np.argmin(norm)
        S_init[i,:] = S_list[ind_s, :]
    return S_init
def logit(p):
    """
    Parameters
    ----------
    p: numpy array
        A variable/matrix need to be transformed.
  
    Returns
    -------
    numpy array
    
    The logit function, p / (1 - p).
    """
    return np.log(p / (float(1) - p))
    
def invlogit(p, eps=1e-9):
    """
    Parameters
    ----------
    p: numpy array
        A variable/matrix need to be transformed.
    eps : float, positive value
        A small value for numerical stability in invlogit.

    Returns
    -------
    numpy array
    The inverse of the logit function, 1 / (1 + exp(-x)).
    """
    return (1. - 2. * eps) / (1. + np.exp(-p)) + eps



class StickBreaking():
    """
    Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    Primarily borrowed from the STAN implementation.
    Parameters
    ----------
    eps : float, positive value
        A small value for numerical stability in invlogit.
    """
    name = "stickbreaking"
    def __init__(self, eps=1e-9):
        self.eps = eps
    def forward(self, x_):
        x = x_.T
        # reverse cumsum
        x0 = x[:-1]
        s = np.cumsum(x0[::-1], 0)[::-1] + x[-1]
        z = x0 / s
        Km1 = x.shape[0] - 1
        k = np.arange(Km1)[(slice(None),) + (None,) * (x.ndim - 1)]
        eq_share = nplogit(1. / (Km1 + 1 - k).astype(str(x_.dtype)))
        y = nplogit(z) - eq_share
        return y
    def backward(self, y_):
        y = y_.T
        Km1 = y.shape[0]
        k = np.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
        z = invlogit(y + eq_share, self.eps)
        yl = np.concatenate([z, np.ones(y[:1].shape)])
        yu = np.concatenate([np.ones(y[:1].shape), 1 - z])
        S = np.cumprod(yu, 0)
        x = S * yl
        return x.T
    def jacobian_det(self, y_):
        y = y_.T
        Km1 = y.shape[0]
        k = np.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
        yl = y + eq_share
        yu = np.concatenate([np.ones(y[:1].shape), 1 - invlogit(yl, self.eps)])
        S = np.cumprod(yu, 0)
        return np.sum(np.log(S[:-1]) - np.log(1 + np.exp(yl)) - np.log(1 + np.exp(-yl)), 0)
stick_breaking = StickBreaking()

class Cand_w():
    """
    Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    Primarily borrowed from the STAN implementation.
    Parameters
    ----------
    eps : float, positive value
        A small value for numerical stability in invlogit.
    """
    name = "cand_w"
    def __init__(self, eps=1e-9):
        self.eps = eps
    def forward(self, x_, point=None):
        y = np.zeros(shape=(x_.shape[0]-1, x_.shape[1]))
        for ind in range(x_.shape[1]):
            y[:,ind] = stick_breaking.forward(x_[:,ind])
        return y
    def backward(self, y_):
        x = np.zeros(shape=(y_.shape[0] + 1, y_.shape[1]))
        for ind in range(y_.shape[1]):
            x[:,ind] = stick_breaking.backward(y_[:,ind])
        return x
    def jacobian_det(self, y_):
        result = 0
        for ind in range(y_.shape[1]):
            result = result + stick_breaking.jacobian_det(y_[:,ind])
        return result
cand_w = Cand_w()

class logit_transform():
    """
    Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    Primarily borrowed from the STAN implementation.
    Parameters
    ----------
    eps : float, positive value
        A small value for numerical stability in invlogit.
    """
    name = "logit"
    def __init__(self, eps=1e-9):
        self.eps = eps
    def forward(self, x):
        return np.log(x / (float(1) - x))
    def backward(self, y):
        return (1. - 2. * self.eps) / (1. + np.exp(-y)) + self.eps
    def jacobian_det(self, y):
        return np.log(self.backward(y)) + np.log(1-self.backward(y))
logit_transform = logit_transform()


#########################Log Posterior###########################

def logNorm(S, W, Y, sigma2):
    """
    The function used to calculate the log probability of Y conditional on S, 
    omega and sigma.
    """
    return np.sum(norm.logpdf(Y.flatten()-np.dot(S,W).flatten(), scale=np.sqrt(sigma2)))
"""
The following functions are log probability functions for design matrix, 
parameter matrix and the hyperparameter alpha.
"""
def S_logp(S, W, Y, p, sigma2):
    return logNorm(S, W, Y, sigma2) + np.sum(binom.logpmf(np.sum(S, axis=1), S.shape[1], p))

def w_logp(S, W, Y, alpha, sigma2):
    rho = np.sum(W, axis=1) / np.sum(np.sum(W))
    return logNorm(S, W, Y, sigma2) + dirichlet.logpdf(rho, alpha)

def alpha_logp(alpha, rho, delta_cum_prod, phi):
    result = dirichlet.logpdf(rho, alpha)
    for ind in range(len(alpha)):
        result = result + halfnorm.logpdf(alpha[ind] / np.sqrt(1/(phi[ind] * delta_cum_prod[ind])))
    return result

##########################Import dataset##########################

Y_true = np.array(pd.read_csv(options.Y_matrix, sep=" ", header = None))

def Gibbs(Y_true, m, path, N_samples = 3000, N_burnin = 2000):
    """
    Parameters
    ----------
    Y_true : numpy array
        The observed outcome matrix.
    m : positive int
        The inner dimension parameter.
    path : string
        The path to the data set.
    N_samples : positive int, optional
        Number of iterations after burnin. The default is 3000.
    N_burnin : positive int, optional
        Number of iterations as burnin. The default is 2000.

    """    
    N, T, m = Y_true.shape[0], Y_true.shape[1], m #Define shape parameters
    Y = Y_true
#########################Initialization##########################
#### parameters: S， W and sigma ####
    W_init = HaploSepCombi(Y, m)
    order_index_init = np.flip(np.argsort(np.sum(W_init, axis=1)))
    W_init = W_init[order_index_init, :] #order the inital omega by the row sum
    for row_ind in range(W_init.shape[0]):
        zero_ind = (W_init[row_ind,:] == 0)
        small_value = np.min(W_init[row_ind,W_init[row_ind,]>0])
        W_init[row_ind,zero_ind] = small_value/10
        if row_ind > 0:
            W_init[0,zero_ind] = W_init[0,zero_ind] - small_value/10
        else:
            W_init[1,zero_ind] = W_init[1,zero_ind] - small_value/10
    S_init = S_initialization(Y_true, W_init)
#### Hyperparameters #####
    a_beta = 2
    b_beta = 2
    a_gamma, b_gamma = 3, 3
    sigma_alpha = 1.5
    sigma_w = 0.01
    phi_gamma = 3
    a_1, a_2 = 0.5, 4
    sigma2 = np.var(Y.flatten() - np.dot(S_init, W_init).flatten())
    p = rng.beta(a_beta,b_beta,N)
    alpha =  np.sum(W_init,axis=1)
    delta0 = rng.gamma(a_1, 1)
    delta1 = rng.gamma(a_2, 1, size= m-1)
    delta = np.concatenate((delta0,delta1), axis=None)
    phi = rng.gamma(3, 3, size= m)
    rho = np.sum(W_init,axis=1) / np.sum(np.sum(W_init))
    
    
    Samples = {'m': m, 'sigma2': sigma2, 'S': S_init, 'p': p, 'W': W_init, 'rho': rho,
               'alpha': alpha, 'delta': delta, 'phi': phi}
    All_samples  = []
#########################Gibbs Sampling###############################################
    for t in range(N_samples+N_burnin):
        print('Step: '+ str(t))
        All_samples.append(copy.deepcopy(Samples))
        
        # Step 1: sample/update S using BinaryGibbsMetropolis
        S_shape = Samples['S'].shape
        s_curr = np.copy(Samples['S'])
        s_list = np.array(list(itertools.product(range(2), repeat=S_shape[1])))
        order = list(range(S_shape[0]))
        random.shuffle(order)
        for row in order:
            prob_s = []
            for i in range(s_list.shape[0]):
                s_cand_i = np.copy(s_curr)
                s_cand_i[row,:] = s_list[i,:]
                log_norm = np.sum(norm.logpdf(Y[row,:]-np.dot(s_cand_i,Samples['W'])[row,:], scale=np.sqrt(Samples['sigma2'])))
                prob_i = np.exp(log_norm + binom.logpmf(np.sum(s_list[i,:]), S_shape[1], p[row]))
                prob_s.append(prob_i)
     
            s_cdf = np.cumsum(prob_s)/(np.sum(prob_s))
            u_s = rng.rand()
            if u_s< s_cdf[0]:
                s_ind = 0
            else:
                s_ind = len(np.where(s_cdf<u_s)[0])
            s_curr[row,:] = s_list[s_ind,:]            
        Samples['S'] = s_curr
    
        # Step 2: sample p
        rowsum_S = np.sum(Samples['S'], axis=1)
        Samples['p'] = rng.beta(a_beta + rowsum_S, b_beta + Samples['m'] - rowsum_S)
    
        # Step 3: sample/update omega in a Gibbs manner each column as they are independent
        for ind_w in range(5):
            W_curr = np.copy(Samples['W'])
            logp_curr_w = w_logp(Samples['S'], W_curr, Y, Samples['alpha'], Samples['sigma2'])
            Y_curr = cand_w.forward(W_curr)
            Y_prop = Y_curr + rng.normal(loc=0, scale=sigma_w, size=Y_curr.shape)
            jac_det_curr, jac_det_prop = cand_w.jacobian_det(Y_curr), cand_w.jacobian_det(Y_prop)
            W_prop = cand_w.backward(Y_prop)
            logp_prop_w = w_logp(Samples['S'], W_prop, Y, Samples['alpha'], Samples['sigma2'])
            if np.log(rng.rand()) < (logp_prop_w + jac_det_prop) - (logp_curr_w + jac_det_curr):
                W_curr = W_prop
                logp_curr_w = logp_prop_w
            Samples['W'] = W_curr
            Samples['rho'] = np.sum(W_curr, axis=1) / np.sum(np.sum(W_curr))

        # Step 4: sample alpha
        delta_cum_prod = np.cumprod(Samples['delta'])
        Alpha = np.log(np.copy(Samples['alpha']))
        Alpha_cand = Alpha + rng.multivariate_normal(np.zeros(Samples['m']), sigma_alpha * np.diag(np.ones(Samples['m'])))
        alpha_cand = np.exp(Alpha_cand)
        logp_alpha = alpha_logp(Samples['alpha'], Samples['rho'], delta_cum_prod, Samples['phi'])
        logp_alpha_cand = alpha_logp(alpha_cand, Samples['rho'], delta_cum_prod, Samples['phi'])
        u_alpha = rng.uniform(0, 1)
        if np.log(u_alpha) < (logp_alpha_cand + np.log(np.prod(alpha_cand))) - (logp_alpha + np.log(np.prod(Samples['alpha']))):
            Samples['alpha'] = alpha_cand

        # Step 5: sample delta
        delta, alpha_sq, phi = np.copy(Samples['delta']), np.copy(np.square(Samples['alpha'])), np.copy(Samples['phi'])
        tau = np.cumprod(delta)
        delta_up = np.zeros(shape=(len(delta),))
        for ind in range(len(delta)):
            if ind == 0:
                delta_up[ind] = rng.gamma(a_1 + 0.5 * (len(delta) - (ind+1) + 1),
                                                1 / (1 + 0.5 * np.sum(np.array(tau[ind:] / delta[ind]) * phi[ind:] * alpha_sq[ind:])) )
            else:
                delta_up[ind] = rng.gamma(a_2 + 0.5 * (len(delta) - (ind+1) + 1),
                                                1 / (1 + 0.5 * np.sum(np.array(tau[ind:] / delta[ind]) * phi[ind:] * alpha_sq[ind:])) )
        Samples['delta'] = delta_up
    
        # Step 6: sample phi
        Samples['phi'] = rng.gamma(.5 * (phi_gamma + 1), 1 / (.5 * (phi_gamma + np.cumprod(Samples['delta']) * np.square(Samples['alpha']))))

        # Step 7: sample sigma square
        Samples['sigma2'] = invgamma.rvs(a=a_gamma + np.prod(Y.shape)/2, scale=b_gamma + np.sum(np.square(Y.flatten()-Samples['S'].dot(Samples['W']).flatten()))/2)
    np.savez(path+'PosteriorSamples',Final=All_samples)
 

file_name = str(pathlib.Path(options.Y_matrix).parent.absolute())+'/'
if __name__ == '__main__':
    a = Gibbs(Y_true, m, file_name, N_samples, N_burnin)

