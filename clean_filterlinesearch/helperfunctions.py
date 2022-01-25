import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import dolfin as dl

def csr_fenics2scipy(A_fenics):
    ai, aj, av = dl.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy


def power_iteration(op, n, maxiter=400, tol=1.e-8):
    x0 = np.random.randn(n)
    lam0 = 0.
    for i in range(maxiter):
        x1 = op(x0)
        lam1 = np.linalg.norm(x1)/np.linalg.norm(x0)
        x1 = x1 / np.linalg.norm(x1)
        if abs(lam0 - lam1)/ max(lam0, lam1) < tol:
            break
        x0 = x1
        lam0 = lam1
    return x1, lam1, i

# ---- gradient check

def grad_check(F, gradF, x0, xhat):
    epss = np.logspace(1, 40, base=0.5)
    F0   = F(x0)
    gradF0   = gradF(x0)
    grad_err = np.zeros(len(epss))
    for j, eps in enumerate(epss):
        Fplus = F(x0 + eps*xhat)
        if len(gradF0.shape) > 1:
            grad_err[j] = np.linalg.norm((Fplus - F0)/ eps - gradF0.dot(xhat), 2)
        else:
            grad_err[j] = np.abs((Fplus - F0) / eps - np.inner(gradF0, xhat))
    plt.plot(epss, grad_err, '-ob')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.title('finite difference check')
    plt.show()



class Krylov_convergence:
    """
    This class allows for tracking the residual history of 
    a Krylov solve.
    """
    def __init__(me, A, b):
        me.counter = [0,]
        me.residuals = list()
        me.A = A
        me.b = b
    def callback(me, xk):
        k = me.counter[0]
        me.counter[0] = k+1
        res = np.linalg.norm(me.A.dot(xk) - me.b) / np.linalg.norm(me.b)
        me.residuals.append(res)
    def reset(me):
        me.counter = [0,]
        me.residuals = list()



"""
MultigridHierarchy will be used to manage the construction
of fine and coarse grid operators, projection/restriction operators and
other necessary infastructure needed to from a fine grid point X = x, lam, z
construct the associated interior-point Newton systems for various levels etc...
needed to ultimately use two_grid_action and Uzawa smoothing to solve
IP-Newton system via Multigrid...
"""

class multigridHierarchy:
    def __init__(me, problems):
        me.problems = problems
        if not len(me.problems) == 2:
            print("EXPECTING TWO PROBLEMS!!!!")

        me.P_state     = csr_fenics2scipy(\
                      dl.PETScDMCollection.create_transfer_matrix(\
                      problems[0].Vh2, problems[1].Vh2))
        me.P_rho       = csr_fenics2scipy(\
                      dl.PETScDMCollection.create_transfer_matrix(\
                      problems[0].Vh1, problems[1].Vh1))
        me.R_state     = me.P_state.transpose()
        me.R_rho       = me.P_rho.transpose()

        me.P        = sps.bmat([[me.P_state, None, None],\
                                [None, me.P_rho, None],\
                                [None, None, me.P_state]], format="csr")
        me.R        = sps.bmat([[me.R_state, None, None],\
                                [None, me.R_rho, None],\
                                [None, None, me.R_state]], format="csr")

        me.Px       = sps.bmat([[me.P_state, None],\
                                 [None, me.P_rho]])
        me.Rx       = sps.bmat([[me.R_state, None],\
                                [None, me.R_rho]], format="csr")

        me.Lfine   = None
        me.Lcoarse = None
        me.S       = None
    # from a fine grid point X = x, lam, z
    # construct the Uzawa smoothers...
    # construct and output two_grid action operator...
    def constructPreconditioner(me, X, smoother='Uzawa', IPsys=True):
        x, lam, z = X[:]
        n1 = me.problems[-1].n1
        rho = x[n1: ]
        Hk  = me.problems[-1].DxxL(X)
        Jk  = me.problems[-1].Dxc(x)
        JkT = Jk.transpose()
        dHk = sps.diags(z / (rho - me.problems[-1].rhol))
        if IPsys:
            Wk  = sps.bmat([[Hk[:n1, :n1], Hk[:n1, n1:]],\
                        [Hk[n1:, :n1], Hk[n1:, n1:] + dHk]], format="csr")
        else:
            Wk = Hk

        # fine grid operator
        me.Lfine = sps.bmat([[Wk, JkT],\
                             [Jk, None]], format="csr")


        # determine the smoother
        n = me.problems[-1].n
        m = me.problems[-1].m
        if smoother == 'Uzawa':
            me.S = Uzawa(Wk, JkT, Jk, n + m, n, 0.0)
            w, its = power_iteration(me.S.Schur_mult, m)[1:]
            print("KKT system Schur complement max eigenvalue = {0:1.2e} converged in {1:d} iterations".format(w, its))
            me.S.w = 0.5 / w
        else:
            me.S = ILUsmoother(me.Lfine)


        # coarse grid operator
        n1coarse  = me.problems[0].n1
        xcoarse   = me.Rx.dot(x)
        rhocoarse = xcoarse[n1coarse:]
        lamcoarse = me.R_state.dot(lam)

        zcoarse   = me.problems[0].Mm.dot(me.R_rho.dot(np.linalg.solve(me.problems[-1].Mm, z)))
        #zcoarse = me.R_rho.dot(z)
        Xcoarse = [xcoarse, lamcoarse, zcoarse]

        Hkcoarse  = me.problems[0].DxxL(Xcoarse)
        Jkcoarse  = me.problems[0].Dxc(xcoarse)
        JkTcoarse = Jkcoarse.transpose()
        dHkcoarse = sps.diags(zcoarse / (rhocoarse - me.problems[0].rhol))
        if IPsys:
            Wkcoarse  = sps.bmat([[Hkcoarse[:n1coarse, :n1coarse], Hkcoarse[:n1coarse, n1coarse:]],\
                              [Hkcoarse[n1coarse:, :n1coarse], Hkcoarse[n1coarse:, n1coarse:] + dHkcoarse]],\
                            format="csr")
        else:
            Wkcoarse = Hkcoarse

        # coarse grid operator
        me.Lcoarse = sps.bmat([[Wkcoarse, JkTcoarse],\
                               [Jkcoarse, None]], format="csr")



class Uzawa:
    def __init__(me, A11, A12, A21, n, idx0, w, M=None):
        # size of operator
        me.n     = n
        
        # indicies for the block splitting defined by the Uzawa
        me.idx0 = idx0
        
        """
        The Uzawa smoother is for saddle point systems, with system
        matrices
        A = [[A11, A12],
             [A21,  0 ]]
        """
        me.A11 = A11
        me.A12 = A12
        me.A21 = A21
        
        me.idx0 = A11.shape[0]
        """
        Define a `mass-matrix' preconditioner for the A11 subblock.
        """        
        me.M   = M
        """
        The splitting parameter \omega, which defines the Uzawa smoother.
        """
        me.w = w

    def dot(me, x):
        rhs1 = x[ :me.idx0]
        rhs2 = x[me.idx0: ]
        z    = np.zeros(len(x))
        if me.M is None:
            z[:me.idx0] = spla.spsolve(me.A11, rhs1)
        else:
            z[:me.idx0], info = spla.cg(me.A11, rhs1, M=me.M, tol=1.e-15)
            if info != 0:
                print("Error with Uzawa smoother action")
        z[me.idx0:] = me.A21.dot(z[:me.idx0])
        z[me.idx0:] = me.w*(z[me.idx0:] - rhs2[:])
        return z
    def Schur_mult(me, x):
        r1 = me.A12.dot(x)
        r2 = spla.spsolve(me.A11, r1)
        y  = me.A21.dot(r2)
        return y
    def Schur_rhs(me, x):
        r1 = spla.spsolve(me.A11, x[:me.idx0])
        r2 = me.A21.dot(r1)
        return r2 - x[me.idx0:]




class ILUsmoother:
    def __init__(me, A):
        me.ILUsolver = spla.spilu(A.tocsc(), drop_tol=1.e-4, fill_factor=6.0)
        me.n         = A.shape[0]
    def dot(me, x):
        return me.ILUsolver.solve(x)


class ILUaction(spla.LinearOperator):
    def __init__(me, A):
        me.shape = A.shape
        me.dtype = A.dtype
        me.ILUsolver = spla.spilu(A.tocsc(), drop_tol=1.e-4, fill_factor=6.0)
    def _matvec(me, b):
        return me.ILUsolver.solve(b)

class two_grid_action(spla.LinearOperator):
    """
    Inherit from spla.LinearOperator class so that
    the action defined by the _matvec method can be
    used as a preconditioner for Krylov solves.

    Inputs:
    Lfine   -- fine grid operator
    Lcoarse -- coarse grid operator
    S       -- fine grid smoother
    R       -- restriction operator (fine to coarse grid)
    P       -- projection operator  (coarse to fine grid)
    m       -- number of pre and post smoothing steps
    """
    def __init__(me, Lfine, Lcoarse, S, P, R, m, M = None, coarsegridcorrection = True):
        me.Lcoarse = Lcoarse #coarse grid operator
        me.Lfine   = Lfine   #fine grid operator
        me.S       = S       #smoother
        me.P       = P       #projection (coarse to fine)
        me.R       = R       #restriction (fine to coarse)
        me.m       = m       #number of pre and post smoothing steps
        me.shape   = Lfine.shape
        me.M       = M       # smoother for coarse grid linear solve
        me.dtype   = Lcoarse.dtype
        me.coarsegridcorrection = coarsegridcorrection
    def _matvec(me, b):
        x  = np.zeros(me.shape[0])
        r  = b.copy()
        """
        Use the smoother to smoothen the
        high frequency components of the error e,
        by e = S r
        """
        for i in range(me.m):
            e  = me.S.dot(r)
            x  = x + e
            r  = b - me.Lfine.dot(x)
        if me.coarsegridcorrection:
            """
            Restrict the fine grid residual to the coarse
            grid.
            Use a direct solver on the coarse grid to obtain an
            accurate solution of the error equation L e = r
            """
            rc = me.R.dot(r)
            if me.M is None:
                ec = spla.spsolve(me.Lcoarse, rc)
            else:
                ec, info = spla.cg(me.Lcoarse, rc, tol=1.e-14, M=me.M)
                if info != 0:
                    print("error in coarse grid solve")
            e = me.P.dot(ec)
        else:
            e = spla.spsolve(me.Lfine, r)
        x = x + e
        """
        If the smoother was symmetric, post-smoothing is necessary
        to ensure that the two-grid action is symmetric and so
        then could be to precondition a MINRES solve.
        """
        r = b - me.Lfine.dot(x)
        for i in range(me.m):
            e = me.S.dot(r)
            x = x + e
            r = b - me.Lfine.dot(x)
        return x  
