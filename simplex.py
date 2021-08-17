import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

"""
Package installation guideline:

>> pip install numpy, tqdm, matplotlib

Run python file as:

>> python simplex.py 0

>> python simplex.py 1

...in the example 2 and 3 cycles are encountered thus the algorithm
would run endlessly, to stop one can press CTRL+C or CTRL+PAUSE,
the cycle avoidance strategy is partially implemented (bug somewhere)
 
>> python simplex.py 2 

...in order to see the algorithm cycling use the verbose mode
>> python simplex.py 2 --verbose

>> python simplex.py 3

the command line argument (0, 1, 2, 3) specifies which example to solve!

"""


def simplex_two_phase(A, b, c, verbose = False):
    """
    Implements simplex method using the 2-phrase strategy outlined in 
    Nocedal & Wright (Ch. 13, S 378 -> STARTING THE SIMPLEX METHOD)

    Solves following problem:
        min c'x s.t. Ax=b, x >= 0

    Subdivide into two phases,

    PHASE I
    min e'z s.t. Ax + Ez  b, (x,z) ≥ 0

    PHASE II
    min c'x s.t. Ax + z  b, x ≥ 0, 0 ≥ z ≥ 0
    
    """
    
    m, n = A.shape  # A is mxn matrix
    e = np.ones((m, ))  # e = (1, 1,..., 1).T
    E = np.zeros((m, m))

    # E_jj = +1 if bj ≥ 0, E_jj = −1 if bj < 0
    for j in range(m):  # b is of shape m,
        E[j, j] = 1 if b[j] >= 0 else -1

    # this way a first feasible point is trivially given by
    x = np.zeros((n, ))
    z = np.abs(b)  # zj = |bj|, j = 1, 2,..., m

    # we need to re-write the phase I problem in order
    # to apply the simplex method below
    p1_A = np.concatenate([A, E], -1)  # mx(n+m) matrix
    p1_c = np.concatenate([np.zeros((n, )), e], 0)  # (n+m), vector
    p1_x = np.concatenate([x, z], 0)  # (n+m), vector

    # by construction the first n entries of x are 0 (active constraints w.r.t. x ≥ 0)
    p1_B_ = np.arange(n, n+m)  # the inactive set {n, n+1,... n+m-1}
    p1_N_ = np.arange(n)  # the active set {1, 2,... n-1}

    # solve the phase I problem to get an initial feasible solution for phase II
    p1_x, p1_B_, p1_N_ = simplex(p1_A, b, p1_c, p1_x, p1_B_, p1_N_, verbose)

    if verbose:
        print('PHASE I yielded initial solution of:', p1_x)

    # if e'z is positive at this solution the original problem is infeasible
    if not np.isclose(np.sum(p1_x[n:]), 0):  # e'z is equivalent to the sum over z components
        raise RuntimeError(f'Problem is infeasible ({np.sum(p1_x[n:])})')

    if np.all(p1_B_ < n):  # PHASE II
        # if all indices of the inactive set belong to some x of the original problem
        # we have the case that no artificial elements of z are remaining in the basis
        x = p1_x[:n]  # initial feasible solution
        B_ = p1_B_  # initial inactive set

        # take those indices i ∈ {1,2,...n} not already in the inactive set
        N_ = np.array(list({i for i in range(n)} - set(p1_B_)))  # initial active set

        return simplex(A, b, c, x, B_, N_, verbose)

    else:
        # from the book:
        # "...the final basis B for the Phase II problem may still contain components of
        # z, making it unsuitable as an optimal basis for..." the original problem
        print('...')


def simplex(A, b, c, x, B_, N_, verbose: bool = False, eps: float = 0.001):
    """
    Implements simplex method using the algorithm 13.1 outlined in 
    Nocedal & Wright (Ch. 13)

    Solves following problem:
        min c'x s.t. Ax=b, x >= 0
    """
    assert 1 > eps > 0

    costs = []
    keep_track = None
    encountered_cycle = False
    pbar = tqdm(desc='Find optimal vertex of polytope')
    while True:

        if verbose:
            print(f'B_ = {B_}; N_ = {N_}')
            input('PRESS ENTER TO CONTINUE')

        if keep_track is None:
            keep_track = [B_.copy()]
        else:
            if not encountered_cycle:  # once we get rid of the degenerated basis we should never attempt one again
                for j in range(len(keep_track)):
                    if np.array_equal(B_, keep_track[j]):  # found the cycle entry
                        # import pdb; pdb.set_trace()
                        veps = np.array([eps ** (k+1) for k in range(len(b))])
                        original_b = b.copy()  # save b for resetting with final basis
                        b = b + B @ veps
                        x[B_] = x[B_] + veps
                        encountered_cycle = True
                        print('CYCLE ENCOUNTERED')
                        break
                else:  # the else clause executes after the loop completes normally
                    keep_track.append(B_.copy())

        # Select the colums of the constraint matrix to build the matrices
        B = A[:, B_]  # mxm
        N = A[:, N_]  # mx(n-m)

        Binv = np.linalg.inv(B)

        # Are given by choice of input x
        # x[B_] = Binv @ b
        # x[N_] = 0

        if verbose:
            print(f'xB = {x[B_]} (xi > 0 if xi == 0 degeneracy case)')
        
        # Solve B.T @ λ = cB for λ
        lambda_ = Binv.T @ c[B_]  # λ

        # Compute sN = cN − N.T @ λ
        sN = c[N_] - N.T @ lambda_

        # Optionally monitor the cost function over iterations
        if verbose:
            costs.append(c.T @ x)

        # Keep track of algorithm progress via progress bar updates
        pbar.set_postfix_str(f'c.T@x={c.T @ x}')
        pbar.update()

        if np.all(sN >= 0):
            pbar.close()

            if encountered_cycle:  # post processing step required
                x[B_] = Binv @ original_b

            break  # found cost minimizing polytope vertex x

        # Select q ∈ N with sq < 0 as the entering index
        q = N_[np.argmin(sN[sN < 0])]  # take the most negative one, q ∈ {1,2,...n}

        # Solve Bd = Aq for d
        d = Binv @ A[:, q]

        if np.all(d <= 0):
            pbar.close()
            raise RuntimeError('Problem is unbounded')
    
        # Calculate xq+ = min i | di>0 (xB)i/di and use p to denote the minimizing i
        pos_entries = np.argwhere(d > 0).flatten()  # i | di>0 ; with i ∈ {1,2,...m}
        search_vector = x[B_][pos_entries] / d[pos_entries]  # (xB)i/di ; m,
        p = np.argmin(search_vector)  # index p ∈ {1,2,...m}
        xq = search_vector[p]  # xq+  WARNING: xq ≠ x[q]
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

    if verbose:
        plt.plot(costs)
        ax = plt.gca()
        plt.title('cost function over iterations')
        ax.set_ylabel('c.T @ x')
        ax.set_xlabel('simplex iterations')
        plt.savefig('costs.png')

    return x, B_, N_  # cost minimizing vertex


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('nr', type=int, help='Example Nr.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.nr == 0:
        """
        example 13.9) from Chapter 13 in the book

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
        
        x, *_ = simplex(A, b, c, x, B_, N_, verbose=args.verbose)
        print('Found optimal x:', x)

        """
        optimal x = [4 0 1 0] (with c.T @ x = -20)

        ==> the ultimate solution is x1 = 4 and x2 = 0, they fulfil all
        of the given inequality constraints
            x1 + x2 ≤ 5,
            2x1 + (1/2)x2 ≤ 8,
            x ≥ 0.
        """

    elif args.nr == 1:
        """
        example 13.9) from Chapter 13 in the book

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
            [2, 1 / 2, 0, 1]
        ])
        b = np.array([5, 8])  # m,
        c = np.array([-5, -1, 0, 0])  # n,  -> min c.T @ x

        x, *_ = simplex_two_phase(A, b, c, verbose=args.verbose)
        print('Found optimal x:', x)
        """
        Find optimal vertex of polytope: 3it [00:00, 1508.20it/s, c.T@x=0.0]
        PHASE I yielded initial solution of: [3.66666667 1.33333333 0. 0. 0. 0.]
        -> here we can see that a feasible solution is found in the first
        phase of the algorithm which is used as the initial feasible point
        of the second phase, and indeed the second phase is able to find the optimal point
        from there on.
        
        Find optimal vertex of polytope: 2it [00:00, ?it/s, c.T@x=-20.0]
        Found optimal x: [4. 0. 1. 0.]
        """

    elif args.nr == 2:
        """
        example 13.2) from Chapter 13 in the book
        
        min 3x1 + x2 + x3 subject to
            2x1 + x2 + x3 ≤ 2,
            x1 − x2 − x3 ≤ −1,
            x ≥ 0.
            
        we transform the inequality constraints to equality constraints x4, x5

        min 3x1 + x2 + x3 subject to
            2x1 + x2 + x3 + x4 = 2,
            x1 − x2 − x3 + x5 = −1,
            x ≥ 0. ( where the new vector x is (x1, x2, x3, x4. x5).T )
        """

        A = np.array([  # mxn
            [2,  1,  1, 1, 0],
            [1, -1, -1, 0, 1]
        ])
        b = np.array([2, -1])  # m,
        c = np.array([3, 1, 1, 0, 0])  # n,  -> min c.T @ x

        x, *_ = simplex_two_phase(A, b, c, verbose=args.verbose)
        print('Found optimal x:', x)
        """
        solution 
        """
    elif args.nr == 3:
        """

        min -3x1 - 2x2 subject to
            -x1 + x2 ≤ 4,
            3x1 − 2x2 ≤ 14,
            x1 - x2 ≤ 3
            x ≥ 0.

        we transform the inequality constraints to equality constraints x3, x4. x5

        min -3x1 - 2x2 subject to
            -x1 +  x2 + x3 = 4,
            3x1 − 2x2 + x4 = 14,
             x1 - x2  + x5 = 3
            x ≥ 0. ( where the new vector x is (x1, x2, x3, x4. x5).T )
        """

        A = np.array([  # mxn
            [-1, 1, 1, 0, 0],
            [3, -2, 0, 1, 0],
            [1, -1, 0, 0, 1]
        ])
        b = np.array([4, 14, 3])  # m,
        c = np.array([-3, -2, 0, 0, 0])  # n,  -> min c.T @ x

        x = np.array([0, 0, 4, 14, 3])  # initial feasible point x,
        # which can be trivially found when each constraint has
        # introduced a slack variable and b ≥ 0.

        # build the initial basis with inactive constraints B_
        # and active constraints N_ (w.r.t. x ≥ 0)
        B_ = np.array([i for i in range(len(x)) if x[i] > 0])
        N_ = np.array([i for i in range(len(x)) if x[i] == 0])

        x, *_ = simplex(A, b, c, x, B_, N_, verbose=args.verbose)
        print('Found optimal x:', x)

        # x, *_ = simplex_two_phase(A, b, c, verbose=args.verbose)
        # print('Found optimal x:', x)

    else:
        raise argparse.ArgumentError
