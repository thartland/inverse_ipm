import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla


try:
    import dolfin as dl
except:
    Warning("FEniCS not found in current Python environment.")

def csr_fenics2scipy(A_fenics):
    ai, aj, av = dl.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy


def make_pairs(data, filename):
    n     = len(data)
    pairs = [(j+1, data[j]) for j in range(n)]
    np.savetxt(filename+".dat", pairs)


def doublepass(A, k):
    n = A.shape[0]
    Omega = np.random.randn(n, k)
    Y     = np.zeros((n,k))
    for i in range(k):
        Y[:, i] = A._matvec(Omega[:,i])
    QY, _ = np.linalg.qr(Y)
    Z    = np.zeros((n,k))
    for i in range(k):
        Z[:, i] = A._rmatvec(QY[:,i])
    QZ, RZ = np.linalg.qr(Z)
    Vhat, Sig, Uhat = np.linalg.svd(RZ)
    V = QZ.dot(Vhat)
    U = QY.dot(Uhat.T)
    return U, Sig, V 
    
    
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

def grad_check(F, gradF, x0, xhat, save=False, title='myfig.png'):
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
    if save:
        plt.savefig(title)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()



class Krylov_convergence:
    """
    This class allows for tracking the residual history of 
    a Krylov solve.
    """
    def __init__(me, A, b, residual_callback=True):
        me.counter = [0,]
        me.residuals = list()
        me.A = A
        me.b = b
        me.residual_callback = residual_callback
    def callback(me, xk):
        k = me.counter[0]
        me.counter[0] = k+1
        if me.residual_callback:
            # xk is residual
            res = np.linalg.norm(xk) / np.linalg.norm(me.b)
        else:
            # xk is not residual
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
        me.Spre    = None
        me.Spost   = None
    # from a fine grid point X = x, lam, z
    # construct the Uzawa smoothers...
    # construct and output two_grid action operator...
    def constructPreconditioner(me, Wk, JkT, Jk, n1):#X):
        #x, lam, z = X[:]
        #n1 = me.problems[-1].n1
        #rho = x[n1: ]
        #Hk  = me.problems[-1].DxxL(X)
        #Jk  = me.problems[-1].Dxc(x)
        #JkT = Jk.transpose()
        #dHk = sps.diags(z / (rho - me.problems[-1].rhol))
        #Wk  = sps.bmat([[Hk[:n1, :n1], Hk[:n1, n1:]],\
        #            [Hk[n1:, :n1], Hk[n1:, n1:] + dHk]], format="csr")
        
        # fine grid operator
        me.Lfine = sps.bmat([[Wk, JkT],\
                             [Jk, None]], format="csr")


        me.Spre  = ConstrainedPreSmoother(Wk, JkT, Jk, n1, Mgrid=True, P = me.P_state, R = me.R_state)
        me.Spost = ConstrainedPostSmoother(Wk, JkT, Jk, n1, Mgrid=True, P = me.P_state, R=me.R_state)

        me.Lcoarse = me.R.dot(me.Lfine).dot(me.P)
        


        # coarse grid operator
        #n1coarse  = me.problems[0].n1
        #xcoarse   = me.Rx.dot(x)
        #rhocoarse = xcoarse[n1coarse:]
        #lamcoarse = me.R_state.dot(lam)

        #zcoarse   = me.problems[0].Mm.dot(me.R_rho.dot(np.linalg.solve(me.problems[-1].Mm, z)))
        #zcoarse = me.R_rho.dot(z)
        #Xcoarse = [xcoarse, lamcoarse, zcoarse]

        #Hkcoarse  = me.problems[0].DxxL(Xcoarse)
        #Jkcoarse  = me.problems[0].Dxc(xcoarse)
        #JkTcoarse = Jkcoarse.transpose()
        #dHkcoarse = sps.diags(zcoarse / (rhocoarse - me.problems[0].rhol))
        #if IPsys:
        #    Wkcoarse  = sps.bmat([[Hkcoarse[:n1coarse, :n1coarse], Hkcoarse[:n1coarse, n1coarse:]],\
        #                      [Hkcoarse[n1coarse:, :n1coarse], Hkcoarse[n1coarse:, n1coarse:] + dHkcoarse]],\
        #                    format="csr")
        #else:
        #    Wkcoarse = Hkcoarse

        # coarse grid operator
        #me.Lcoarse = me.R.dot(me.Lfine).dot(me.P)
        #me.Lcoarse = sps.bmat([[Wkcoarse, JkTcoarse],\
        #                       [Jkcoarse, None]], format="csr")



class Uzawa:
    def __init__(me, A11, A12, A21, w, M=None):
        # size of operator
        me.n     = A11.shape[0] + A21.shape[0]
        me.idx0 = A11.shape[0]
        
        """
        The Uzawa smoother is for saddle point systems, with system
        matrices
        A = [[A11, A12],
             [A21,  0 ]]
        """
        me.A11 = A11
        me.A12 = A12
        me.A21 = A21
        
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

class ConstrainedSmoother:
    def __init__(me, W, JT, J, n1, lumped=False):
        # size of operator
        me.n     = W.shape[0] + J.shape[0]
        # splitting index
        me.idx0  = W.shape[0]
        me.n1 = n1

        me.lumped=lumped
        
        """
        This constrained smoother is for symmetric saddle point systems,
        with block system structure
        matrices
        A = [[W, J^T],
             [J,  0 ]]
        """
        me.J  = J
        me.JT = JT
        me.W  = W

        if me.lumped:
            Wblock = sps.bmat([[W[:me.n1,:me.n1], None],\
                               [None, W[me.n1:,me.n1:]]], format='csr')
            one = np.ones(me.idx0)
            me.DWinv = sps.diags( 1. / (Wblock.dot(one)))
        else:
            me.DWinv = sps.diags(1. / me.W.diagonal())
        

    def dot(me, b):
        f = b[ :me.idx0]
        g = b[me.idx0: ]

        z = np.zeros(me.n)
        # rhs to determine y
        rhs = me.J.dot(me.DWinv.dot(f)) - g
        sysmat = me.J.dot(me.DWinv.dot(me.JT))
        y = spla.spsolve(sysmat, rhs)
        z[me.idx0:] = y[:]

        # determine x
        x = me.DWinv.dot(f - me.JT.dot(y))
        z[:me.idx0] = x[:]
        z[:] = z[:]
        return z


"""
This describes the action of a smoother
S, wherein

S^-1 = [[Wuu  0   JuT]
        [0   Wmm   0 ]
        [Ju   0    0 ]]

this smoother is for a saddle point system

A =    [[W  JT]
        [J   0]]

where

W =    [[Wuu Wum]
        [Wmu Wmm]]

J      = [Ju Jm] 
"""

class ApproximateConstrainedSmoother:
    def __init__(me, W, JT, J, n1):
        me.W  = W
        me.JT = JT
        me.J  = J

        me.n1 = n1
        me.idx0 = W.shape[0]
        me.n    = W.shape[0] + J.shape[0]

        me.Wuu = W[:n1,:n1]
        me.Wmm = W[n1:,n1:]
        me.Ju  = J[:,:n1]
        me.Jm  = J[:,n1:]
        me.JuT = me.Ju.transpose()
        me.JmT = me.Jm.transpose()

    def dot(me, R):
        R1 = R[:me.n1]
        R2 = R[me.n1:me.idx0]
        R3 = R[me.idx0:]

        # S R = z
        z  = np.zeros(me.n)
        
        du = spla.spsolve(me.Ju, R3)
        dm = spla.spsolve(me.Wmm, R2)
        dy = spla.spsolve(me.JuT, R1 - me.Wuu.dot(du))

        z[:me.n1]        = du[:]
        z[me.n1:me.idx0] = dm[:]
        z[me.idx0:]      = dy[:]

        return z



"""
Cumulative Smoother, combine smoother actions to yield

S_3 = S_1 + S_2 - S_2 A S_1 (strategy 1)
S_3 = S_2 A S_1            
"""
class CumulativeSmoother(spla.LinearOperator):
    def __init__(me, S1, S2, A, strategy=1):
        me.dtype = A.dtype
        me.shape = A.shape
        me.S1 = S1
        me.S2 = S2
        me.A  = A
        me.strategy = strategy
    def dot(me, x):
        y = me.S1.dot(x)
        z = me.A.dot(y)
        if me.strategy == 1:
            w = me.S2.dot(x- z)
            return y + w
        else:
            w = me.S2.dot(z)
            return w
    def _matvec(me, b):
        return me.dot(b)








"""
ConstrainedPreSmoother
This describes the action of a smoother
S, wherein

S^-1 = [[Wuu     0      JuT]
        [Wmu diag(Wmm)  JmT]
        [Ju      0       0 ]]

this smoother is for a saddle point system

A =    [[W  JT]
        [J   0]]

where

W =    [[Wuu Wum]
        [Wmu Wmm]]

J      = [Ju Jm] 
"""


class ConstrainedPreSmoother(spla.LinearOperator):
    def __init__(me, W, JT, J, n1, Mgrid=False, P=None, R=None):
        me.W  = W
        me.JT = JT
        me.J  = J

        me.n1 = n1
        me.idx0 = W.shape[0]
        me.n    = W.shape[0] + J.shape[0]
        me.shape = (me.n, me.n)
        me.dtype = W.dtype 

        me.Wuu = W[:n1, :n1]
        me.Wmm = W[n1:, n1:]
        #me.Wmm = sps.diags(W[n1:, n1:].diagonal(), format="csr")
        me.Wmu = W[n1:, :n1]
        me.Ju  = J[:,:n1]
        me.Jm  = J[:,n1:]
        me.JuT = me.Ju.transpose()
        me.JmT = me.Jm.transpose()
        
        me.Mgrid = Mgrid
        me.P     = P
        me.R     = R

        if me.Mgrid and (P is None or R is None):
            print("ERROR!!!, Must supply Projection/Restriction when using Mgrid solves in smoother!")
        
        if me.Mgrid:
            w = 2. / 3. # relaxation factor
            m = 10      # Jacobi pre/post smoothing steps for Ju solves
            Jucoarse  = me.R.dot(me.Ju.dot(me.P))
            SJu       = sps.diags(w / me.Ju.diagonal(), format="csr")
            JuTcoarse = me.R.dot(me.JuT.dot(P))
            SJuT      = sps.diags(w / me.JuT.diagonal(), format="csr")
            me.Ju_twogrid_P = two_grid_action(me.Ju, Jucoarse, SJu, P, R, m)
            me.JuT_twogrid_P = two_grid_action(me.JuT, JuTcoarse, SJuT, P, R, m)

    def dot(me, R):
        R1 = R[:me.n1]
        R2 = R[me.n1:me.idx0]
        R3 = R[me.idx0:]

        # S R = z
        z  = np.zeros(me.n)
        
        if me.Mgrid:
            krylov_convergence = Krylov_convergence(me.Ju, R3, residual_callback=False)
            du, info = spla.cg(me.Ju, R3, tol=1.e-12, maxiter=100, M = me.Ju_twogrid_P, callback=krylov_convergence.callback)
            if info > 0:
                print("CG solve failure!!!")
            else:
                print("CG converged in {0:d} iterations".format(len(krylov_convergence.residuals)))
            dy, info = spla.cg(me.JuT, R1 - me.Wuu.dot(du), tol=1.e-12, maxiter=100, M=me.JuT_twogrid_P)
            if info > 0:
                print("CG solve failure!!!")
        else:
            du = spla.spsolve(me.Ju, R3)
            dy = spla.spsolve(me.JuT, R1 - me.Wuu.dot(du))
        dm = spla.spsolve(me.Wmm, R2 - me.Wmu.dot(du) - me.JmT.dot(dy))

        z[:me.n1]        = du[:]
        z[me.n1:me.idx0] = dm[:]
        z[me.idx0:]      = dy[:]

        return z
    def _matvec(me, b):
        return me.dot(b)


"""
ConstrainedPostSmoother
This describes the action of a smoother
S, wherein

S^-1 = [[Wuu    Wum     JuT]
        [0   diag(Wmm)   0 ]
        [Ju     Jm       0 ]]

this smoother is for a saddle point system

A =    [[W  JT]
        [J   0]]

where

W =    [[Wuu Wum]
        [Wmu Wmm]]

J      = [Ju Jm] 
"""


class ConstrainedPostSmoother:
    def __init__(me, W, JT, J, n1, Mgrid=False, P=None, R=None):
        me.W  = W
        me.JT = JT
        me.J  = J

        me.n1 = n1
        me.idx0 = W.shape[0]
        me.n    = W.shape[0] + J.shape[0]

        me.Wuu = W[:n1, :n1]
        #me.Wmm = sps.diags(W[n1:, n1:].diagonal(), format="csr")
        me.Wmm = W[n1:, n1:]
        me.Wum = W[:n1, n1:]
        me.Ju  = J[:,:n1]
        me.Jm  = J[:,n1:]
        me.JuT = me.Ju.transpose()
        me.JmT = me.Jm.transpose()
        
        me.Mgrid = Mgrid
        me.P     = P
        me.R     = R

        if me.Mgrid and (P is None or R is None):
            print("ERROR!!!, Must supply Projection/Restriction when using Mgrid solves in smoother!")
        
        if me.Mgrid:
            w = 2. / 3. # relaxation factor
            m = 10      # Jacobi pre/post smoothing steps for Ju solves
            Jucoarse  = me.R.dot(me.Ju.dot(me.P))
            SJu       = sps.diags(w / me.Ju.diagonal(), format="csr")
            JuTcoarse = me.R.dot(me.JuT.dot(P))
            SJuT      = sps.diags(w / me.JuT.diagonal(), format="csr")
            me.Ju_twogrid_P = two_grid_action(me.Ju, Jucoarse, SJu, P, R, m)
            me.JuT_twogrid_P = two_grid_action(me.JuT, JuTcoarse, SJuT, P, R, m)

    def dot(me, R):
        R1 = R[:me.n1]
        R2 = R[me.n1:me.idx0]
        R3 = R[me.idx0:]

        # S R = z
        z  = np.zeros(me.n)
        
        dm = spla.spsolve(me.Wmm, R2)
        
        if me.Mgrid:
            du, info = spla.cg(me.Ju, R3 - me.Jm.dot(dm), tol=1.e-12, maxiter=100, M = me.Ju_twogrid_P)
            if info > 0:
                print("CG solve failure!!!")
            dy, info = spla.cg(me.JuT, R1 - me.Wuu.dot(du) - me.Wum.dot(dm), tol=1.e-12, maxiter=100, M=me.JuT_twogrid_P)
            if info > 0:
                print("CG solve failure!!!")
        else:
            du = spla.spsolve(me.Ju,  R3 - me.Jm.dot(dm))
            dy = spla.spsolve(me.JuT, R1 - me.Wuu.dot(du) - me.Wum.dot(dm))

        z[:me.n1]        = du[:]
        z[me.n1:me.idx0] = dm[:]
        z[me.idx0:]      = dy[:]
        return z

    
"""
Warning::: A and B assumed to both be symmetric!!!
"""
class productOperator(spla.LinearOperator):
    def __init__(me, A, B):
        me.shape = A.shape
        me.A = A
        me.B = B
    def dot(me, x):
        y = me.B.dot(x)
        return me.A.dot(y)
    def dotT(me, x):
        y = me.A.dot(x)
        return me.B.dot(y)
    def _matvec(me, x):
        return me.dot(x)
    def _rmatvec(me, x):
        return me.dotT(x)
    

class reducedHessian(spla.LinearOperator):
    def __init__(me, W, JT, J, n1, regularization=True):
        me.n1  = n1
        me.W   = W 
        me.n   = W.shape[0]-n1
        me.shape = (me.n, me.n)
        me.Wuu   = W[:me.n1, :me.n1]
        me.Wum   = W[:me.n1, me.n1:]
        me.Wmu   = W[me.n1:, :me.n1]
        me.Wmm   = W[me.n1:, me.n1:]
        me.Ju    = J[:, :me.n1]
        me.Jm    = J[:, me.n1:]
        me.JuT   = JT[:me.n1, :]
        me.JmT   = JT[me.n1:, :]
        me.r1    = None
        me.r2    = None
        me.r3    = None
        me.regularization = regularization
    def preprhs(me, r):
        me.r1 = r[:me.n1]
        me.r2 = r[me.n1:me.W.shape[0]]
        me.r3 = r[me.W.shape[0]:]
        Juinvr3 = spla.spsolve(me.Ju, me.r3)
        return me.r2 - me.Wmu.dot(Juinvr3) - me.JmT.dot(spla.spsolve(me.JuT, me.r1 - me.Wuu.dot(Juinvr3)))

    def dot(me, mhat):
        uhat   = -spla.spsolve(me.Ju, me.Jm.dot(mhat))
        lamhat = -spla.spsolve(me.JuT, me.Wuu.dot(uhat) + me.Wum.dot(mhat))
        if me.regularization:
            return me.Wmu.dot(uhat) + me.Wmm.dot(mhat) + me.JmT.dot(lamhat)
        else:
            return me.Wmu.dot(uhat) + me.JmT.dot(lamhat)

    def _matvec(me, x):
        return me.dot(x)
    def _rmatvec(me, x):
        return me.dot(x)
    def backsolve(me, mhat):
        uhat   = spla.spsolve(me.Ju, me.r3 - me.Jm.dot(mhat))
        lamhat = spla.spsolve(me.JuT, me.r1 - me.Wuu.dot(uhat) - me.Wum.dot(mhat))
        return np.concatenate([uhat, mhat, lamhat])

class regularizationSmoother(spla.LinearOperator):
    def __init__(me, Wmm):
        me.shape = Wmm.shape
        me.Wmm   = Wmm
    def dot(me, x):
        return spla.spsolve(me.Wmm, x)
    def _matvec(me, x):
        return me.dot(x)



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
    def __init__(me, Lfine, Lcoarse, S, P, R, m, M = None, coarsegridcorrection = True, Spost=None):
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
        me.Spost = Spost 
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
            if me.Spost is None:
                e = me.S.dot(r)
            else:
                e = me.Spost.dot(r)
            x = x + e
            r = b - me.Lfine.dot(x)
        return x


class multi_grid_action(spla.LinearOperator):
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
    def __init__(me, Ls, Ss, Ps, Rs, m):
        me.Ls      = Ls[::-1]      # sequence of operators ordered from fine to coarse
        me.Ss      = Ss[::-1]      # sequence of smoothers
        me.Ps      = Ps[::-1]      # projection (coarse to fine)
        me.Rs      = Rs[::-1]      # restriction (fine to coarse)
        me.m       = m       # number of pre and post smoothing steps
        me.lvl     = len(me.Ls) # depth of the multigrid hierarchy
        me.shape   = me.Ls[0].shape
        me.dtype   = me.Ls[0].dtype
    def single_level_action(me, b, lvl):
        n = len(b)
        x = np.zeros(n)
        r = b.copy()
        if lvl == me.lvl:
            # direct solve on coarsest level
            x = spla.spsolve(me.Ls[lvl-1], b)
            return x
        else:
            # pre smoothing            
            for i in range(me.m):
                e = me.Ss[lvl-1].dot(r)
                x = x + e
                r = b - me.Ls[lvl-1].dot(x)
            # request an update to the error from a coarser level
            rcoarse = me.Rs[lvl-1].dot(r)
            ecoarse = me.single_level_action(rcoarse, lvl+1)
            e       = me.Ps[lvl-1].dot(ecoarse)
            x       = x + e
            # post smoothing
            for i in range(me.m):
                r = b - me.Ls[lvl-1].dot(x)
                e = me.Ss[lvl-1].dot(r)
                x = x + e
            return x
    def _matvec(me, b):
        return me.single_level_action(b, 1)