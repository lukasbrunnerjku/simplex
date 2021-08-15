import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

"""
Package installation guideline:

>> pip install numpy, tqdm, matplotlib

Run python file as:

>> python simplex.py

"""


def simplex_two_phase(A, b, c):
    """
    Implements simplex method using the 2-phrase strategy outlined in 
    Nocedal & Wright (Ch. 13, S 378 -> STARTING THE SIMPLEX METHOD)

    Solves following problem:
        min c'x s.t. Ax=b, x >= 0
    """
    pass


def simplex(A, b, c, x, B_, N_, verbose: bool = False):
    """
    Implements simplex method using the algorithm 13.1 outlined in 
    Nocedal & Wright (Ch. 13)

    Solves following problem:
        min c'x s.t. Ax=b, x >= 0
    """

    costs = []
    pbar = tqdm(desc='Find optimal vertex of polytope')
    while True:

        # Select the colums of the constraint matrix to build the matrices
        B = A[:, B_]  # mxm
        N = A[:, N_]  # mx(n-m)

        Binv = np.linalg.inv(B)

        # Are given by choice of input x
        # x[B_] = Binv @ b
        # x[N_] = 0
        
        # Solve B.T @ λ = cB for λ
        lambda_ = Binv.T @ c[B_]  # λ

        # Compute sN = cN − N.T @ λ
        sN = c[N_] - N.T @ lambda_

        # Optionally monitor the cost function over iterations
        if verbose:
            cost = c.T @ x
            costs.append(cost)

        # Keep track of algorithm progress via progress bar updates
        pbar.set_postfix_str(f'c.T@x={cost}')
        pbar.update()

        if np.all(sN >= 0):
            pbar.close()
            break  # found cost minimizing polytope vertex x

        # Select q ∈ N with sq < 0 as the entering index
        q = N_[np.argmin(sN[sN < 0])]  # take the most negative one, q ∈ {1,2,...n}

        # Solve Bd = Aq for d
        d = Binv @ A[:, q]

        if np.all(d <= 0):
            pbar.close()
            raise RuntimeError('Problem is unbounded')
    
        # Calculate xq+ = min i | di>0 (xB)i/di and use p to denote the minimizing i
        search_vector = x[B_] / d  # (xB)i/di ; m,
        pos_entries = np.argwhere(d > 0).flatten()  # i | di>0 ; with i ∈ {1,2,...m}
        p = np.argmin(search_vector[pos_entries])  # index p ∈ {1,2,...m}
        xq = search_vector[pos_entries][p]  # xq+  WARNING: xq ≠ x[q]
        p = B_[p]  # index p ∈ {1,2,...n}

        # Update xB+ = xB − d * xq+
        x[B_] = x[B_] - d * xq  # xB+ -> now vertex x is 0 at the leaving index p

        # Update xN+ = (0,..., 0, xq+ , 0,..., 0).T
        x[q] = xq  # xN+ -> previously vertex x was 0 at entering index q but now xq

        # Change B by adding q and removing the basic variable corresponding to column p of B
        # Note: q and p are "global" (i.e. q, p ∈ {1,2,...n}) indices 
        # 
        # q -> entering index
        # p -> leaving index  (leaving and entering w.r.t. the basis B_)
        B_[B_ == p] = q
        N_[N_ == q] = p

        #import pdb; pdb.set_trace()

    if verbose:
        plt.plot(costs)
        ax = plt.gca()
        plt.title('cost function over iterations')
        ax.set_ylabel('c.T @ x')
        ax.set_xlabel('simplex iterations')
        plt.savefig('costs.png')

    return x  # cost minimizing vertex


if __name__ == '__main__':

    """
    example 13.9)

    min −5x1 − x2 subject to
        x1 + x2 ≤ 5,
        2x1 + (1/2)x2 ≤ 8,
        x ≥ 0.

    we first need to define equality constraints using slack variables x3, x4
    
    min −5x1 − x2 subject to
        x1 + x2 + x3 = 5,
        2x1 + (1/2)x2 + x4 = 8,
        x ≥ 0. ( where the new vector x is (x1, x2, x3, x4).T )
    """
    A = np.array([  # mxn
        [1, 1, 1, 0],
        [2, 1/2, 0, 1]
    ])
    b = np.array([5, 8])  # m,
    c = np.array([-5, -1, 0, 0])  # n,  -> min c.T @ x

    x = np.array([0, 0, 5, 8])  # initial feasible point x,
    # which can be trivially found when each constraint has
    # introduced a slack variable and b ≥ 0.

    # build the initial basis with inactive constraints B_
    # and active constraints N_ (w.r.t. x ≥ 0)
    B_ = np.array([i for i in range(len(x)) if x[i] > 0])
    N_ = np.array([i for i in range(len(x)) if x[i] == 0])
    
    x = simplex(A, b, c, x, B_, N_, verbose=True)
    print('Found optimal x:', x)
