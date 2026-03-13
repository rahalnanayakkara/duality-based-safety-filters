import numpy as np
from scipy import sparse
import osqp

def phi(x1, x2, a1, a2):
    return -a2*x1*x1 - 2*a2*x2*x2 - 3*a2*x1*x2 - a1*x1 - 2*a1*x2 + a2

def gamma_1(k1, x2_hat, d2, a1, a2):
    '''
    Computes maximum over x2 in [x2_hat-d2, x2_hat+d2], of when x1=k1
    '''
    phi_2 = lambda x : phi(k1, x, a1, a2)
    if a2>0:
        x_opt = -(3*a2*k1+2*a1)/(4*a2)
        if x2_hat-d2 <= x_opt <= x2_hat+d2:
            return phi_2(x_opt)
    return max(phi_2(x2_hat-d2), phi_2(x2_hat+d2))

def gamma_2(k2, x1_hat, d1, a1, a2):
    '''
    Computes maximum over x1 in [x1_hat-d1, x1_hat+d1], of when x2=k2
    '''
    phi_1 = lambda x : phi(x, k2, a1, a2)
    if a2>0:
        x_opt = -(3*a2*k2+a1)/(2*a2)
        if x1_hat-d1 <= x_opt <= x1_hat+d1:
            return phi_1(x_opt)
    return max(phi_1(x1_hat-d1), phi_1(x1_hat+d1))

def sup(x1_hat, x2_hat, d1, d2, a1, a2):
    '''
    Computes maximum of a^T Phi over rectangle [x1_hat-d1, x1_hat+d1]*[x2_hat-d2, x2_hat+d2]
    '''
    vals = [gamma_1(x1_hat-d1, x2_hat, d2, a1, a2),
            gamma_1(x1_hat+d1, x2_hat, d2, a1, a2),
            gamma_2(x2_hat-d2, x1_hat, d1, a1, a2),
            gamma_2(x2_hat+d2, x1_hat, d1, a1, a2)]
    return max(vals)

def polytope_hull(x1_hat, x2_hat, d1, d2, N_planes=16):
    '''
    Over approximate convex hull of using N-planes
    output - (C, d) for Cx <= d representation
    '''
    thetas = np.linspace(-np.pi, np.pi, N_planes, endpoint=False)
    res = []
    C = []
    for theta in thetas:
        res.append(sup(x1_hat, x2_hat, d1, d2, np.cos(theta), np.sin(theta)))
        C.append([np.cos(theta), np.sin(theta)])
    d = np.array(res)
    C = np.array(C)
    return (C, d)

def duality_filter(x_hat, delta, u_nom=0, N_planes=16):
    '''
    Computes a safe control input for the double integrator using estimate x_hat and uncertainty delta.
    The over approximation of the convex hull is computed using N_planes number of hyper-planes.
    '''
    C,d = polytope_hull(x_hat[0], x_hat[1], delta[0], delta[1], N_planes=N_planes)

    A = np.vstack([np.hstack([C.T, np.array([[1], [0]])]), 
                np.concat([d, np.array([0])]), 
                np.hstack([np.eye(N_planes), np.zeros((N_planes, 1))])])
    l = np.concat([[0, -1, -np.inf], np.zeros(N_planes)])
    u = np.concat([[0, -1, 0], np.ones(N_planes)*np.inf])

    P = np.zeros((N_planes+1, N_planes+1))
    P[-1, -1] = 1
    q = np.zeros(N_planes+1)
    q[-1] = - u_nom

    P = sparse.csc_matrix(P)
    A = sparse.csc_matrix(A)
    
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, alpha=1.0, verbose=False)
    res = prob.solve()
    
    return res.x[-1]