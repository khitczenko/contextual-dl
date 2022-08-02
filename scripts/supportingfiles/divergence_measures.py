import numpy as np
import math

def compute_probs(data, durationrange, n=10): 
    h, e = np.histogram(data, bins=n, range=durationrange)
    h = h.astype(np.float32) + 1e-5
    p = h/np.sum(h)
    return e, p

def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def compute_kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

# def compute_js_divergence(p, q):
#     m = (1./2.)*(p + q)
#     return (1./2.)*compute_kl_divergence(p, m) + (1./2.)*compute_kl_divergence(q, m)

# def compute_hellinger_distance(p, q):
#     """Hellinger distance between two discrete distributions.
#        Same as original version but without list comprehension
#     """
#     list_of_squares = []
#     for p_i, q_i in zip(p, q):

#         # caluclate the square of the difference of ith distr elements
#         s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

#         # append 
#         list_of_squares.append(s)

#     # calculate sum of squares
#     sosq = sum(list_of_squares)    

#     return sosq / math.sqrt(2)

def kl_divergence(train_sample, test_sample, durationrange, n_bins=10): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, durationrange, n=n_bins)
    _, q = compute_probs(test_sample, durationrange, n=e)

    # list_of_tuples = support_intersection(p, q)
    # p, q = get_probs(list_of_tuples)
    
    return compute_kl_divergence(p, q)

# Symmetric KL-divergence by adding KL(P|Q) + KL(Q|P)
def symmetric_kl_divergence(c1_dur, c2_dur, durationrange, n_bins=10):
    kl = kl_divergence(np.array(c1_dur).reshape(-1,1), np.array(c2_dur).reshape(-1,1), durationrange, n_bins) + kl_divergence(np.array(c2_dur).reshape(-1,1), np.array(c1_dur).reshape(-1,1), durationrange, n_bins)
    return(kl)

# def js_divergence(train_sample, test_sample, n_bins=10): 
#     """
#     Computes the JS Divergence using the support 
#     intersection between two different samples
#     """
#     e, p = compute_probs(train_sample, n=n_bins)
#     _, q = compute_probs(test_sample, n=e)
    
#     # list_of_tuples = support_intersection(p,q)
#     # p, q = get_probs(list_of_tuples)
    
#     return compute_js_divergence(p, q)

# def hellinger_distance(train_sample, test_sample, n_bins=10): 
#     """
#     Computes the JS Divergence using the support 
#     intersection between two different samples
#     """
#     e, p = compute_probs(train_sample, n=n_bins)
#     _, q = compute_probs(test_sample, n=e)
    
#     # list_of_tuples = support_intersection(p,q)
#     # p, q = get_probs(list_of_tuples)
    
#     return compute_hellinger_distance(p, q)