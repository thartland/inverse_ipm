import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import dolfin as dl
# False - natural ordering; True - interlace
dl.parameters['reorder_dofs_serial'] = False

from helperfunctions import csr_fenics2scipy

"""
An object which will internally handle the finite-element
discretization of the infinite-dimensional inverse problem
and set up the linear KKT system.
"""

class inverseDiffusion:
    def __init__(me, Vh, Vh1, Vh2, beta, gamma1, gamma2, ud, g, rhol):

        me.Vh    = Vh
        me.Vh1   = Vh1
        me.Vh2   = Vh2
        me.rhol  = rhol

        # n1 dofs for state
        # n2 dofs for parameter
        me.n1    = me.Vh.sub(0).dim()
        me.n2    = me.Vh.sub(1).dim()
        me.n     = me.n1 + me.n2
        me.m     = me.n1

        # data that we are fitting the model to
        me.ud    = ud

        # rhs
        me.g = g

        me.beta   = beta
        me.gamma1 = gamma1
        me.gamma2 = gamma2
        utest,  mtest = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        Rform = (dl.Constant(gamma1) * mtest * mtrial +\
                 dl.Constant(gamma2) * dl.inner(dl.grad(mtest), dl.grad(mtrial))) * dl.dx(me.Vh.mesh())
        Mform = (utest * utrial + mtest * mtrial) * dl.dx(me.Vh.mesh())

        me.R_fenics  = dl.assemble(Rform)
        me.R         = me.R_fenics.array()[me.n1:, me.n1:]
        me.M_fenics  = dl.assemble(Mform)
        me.Mx        = me.M_fenics.array()
        me.Mu        = me.Mx[:me.n1,:me.n1]
        me.Mm        = me.Mx[me.n1:,me.n1:]

    """
    In what follows x will be a function on the state-parameter product
    finite-element space
    """

    """
    objective -- return the value of the regularized data-misfit functional at x
    """
    def f(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        return dl.assemble((dl.Constant(0.5) * (X.sub(0) - me.ud)**2. + \
                           dl.Constant(me.gamma1/2.)*X.sub(1)*X.sub(1) + \
                           dl.Constant(me.gamma2/2.)*dl.inner(\
                           dl.grad(X.sub(1)), dl.grad(X.sub(1))))*dl.dx(me.Vh.mesh()))
    """
    gradient -- return the variational derivative of J with respect to x
    """
    def Dxf(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        utest, rhotest = dl.TestFunctions(me.Vh)
        return dl.assemble(((X.sub(0) - me.ud) * utest + \
                            dl.Constant(me.gamma1) * X.sub(1) * rhotest + \
                            dl.Constant(me.gamma2) * \
                            dl.inner(dl.grad(X.sub(1)), dl.grad(rhotest)))*dl.dx(me.Vh.mesh())).get_local()
    def Dxxfform(me, x):
        utest, mtest = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        return (utrial*utest + \
                                dl.Constant(me.gamma1)*\
                                mtrial*mtest + \
                                dl.Constant(me.gamma2)*\
                                dl.inner(dl.grad(mtest), dl.grad(mtrial)))*dl.dx(me.Vh.mesh())
    """
    return the second variational derivative of J with respect to x, that is
    a linear mapping from primal to dual
    """
    def Dxxf(me, x):
        return csr_fenics2scipy(dl.assemble(me.Dxxfform(x)))
    """
    constraints -- evaluate the PDE-constraint at x, returning a dual-vector
    """
    def c(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        utest, mtest = dl.TestFunctions(me.Vh)
        return dl.assemble((X.sub(1)*dl.inner(dl.grad(X.sub(0)), dl.grad(utest)) + \
                            dl.Constant(me.beta)*X.sub(0)*utest - me.g*utest)*dl.dx(me.Vh.mesh()))\
                            .get_local()[:me.n1]
    def theta(me, x):
        return np.linalg.norm(me.c(x), 2)
    """
    evaluate the variational derivative of the PDE-constraint with respect
    to x,
    return a linear mapping from the primal to the dual
    """
    def Dxc(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        p, _   = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        J = csr_fenics2scipy(\
                  dl.assemble((X.sub(1)*dl.inner(dl.grad(utrial), dl.grad(p)) + \
                                mtrial*dl.inner(dl.grad(X.sub(0)), dl.grad(p))+ dl.Constant(me.beta)*utrial*p)*\
                               dl.dx(me.Vh.mesh())))
        J = J[:me.n1,:]
        return J
    # constraint Jacobian
    def Dxxcpform(me, x, p):
        X = dl.Function(me.Vh)
        P = dl.Function(me.Vh)
        P.vector().vec()[:me.n1] = p[:]
        X.vector().set_local(x)
        utest, mtest   = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        return (mtest * dl.inner(dl.grad(utrial), dl.grad(P.sub(0))) + \
                                             mtrial * dl.inner(dl.grad(utest), dl.grad(P.sub(0))))*\
                                            dl.dx(me.Vh.mesh())
    def Dxxcp(me, x, p):
        return dl.assemble(me.Dxxcpform(x, p))
    def phi(me, x, mu):
        return me.f(x) - mu * sum(np.log(x[me.n1:] - me.rhol))
    def Dxphi(me, x, mu):
        y = np.zeros(me.n)
        y += me.Dxf(x)
        y[me.n1:] += -mu / (x[me.n1:] - me.rhol)
        return y
    def L(me, X):
        x, lam, z = X[:]
        return (me.f(x) + np.inner(lam, me.c(x)) - np.inner(z, x[me.n1:]-me.rhol))
    def DxL(me, X):
        x, lam, z = X[:]
        y = np.zeros(me.n)
        y = me.Dxf(x) + me.Dxc(x).transpose().dot(lam)#np.dot(lam.T, me.Dxc(x))
        y[me.n1:] -= z[:]
        return y
    def DxxL(me, X):
        x, lam, z = X[:]
        #y = np.zeros((me.n, me.n))
        #y[:, :] += me.Dxxf(x)
        #y[:, :] += me.Dxxcp(x, lam).array()[:,:]
        y = csr_fenics2scipy(dl.assemble(me.Dxxfform(x) + me.Dxxcpform(x, lam)))
        return y
    def E(me, X, mu, smax):
        x, lam, z = X[:]
        rho = x[me.n1:]
        E1 = np.linalg.norm(me.DxL(X), np.inf)
        E3 = np.linalg.norm((rho - me.rhol)*z-mu, np.inf)
        laml1 = np.linalg.norm(lam, 1)
        zl1   = np.linalg.norm(z,   1)
        sd    = max(smax, (laml1 + zl1) / (me.m + me.n)) / smax
        sc    = max(smax, zl1 / me.n) / smax
        E2    = np.linalg.norm(me.c(x), np.inf)
        return max(E1 / sd, E2, E3 / sc), E1, E2, E3
    def restore_feasibility(me, x):
        u = x[:me.n1]
        rho = x[me.n1:]
        utest  = dl.TestFunction(me.Vh2)
        utrial = dl.TrialFunction(me.Vh2)
        rhofunc = dl.Function(me.Vh1)
        rhofunc.vector()[:] = rho[:]
        aform = (dl.inner(dl.grad(utest), dl.grad(utrial)) + me.beta * utest * utrial) * dl.dx(me.Vh2.mesh())
        Lform = me.g * utest * dl.dx(me.Vh2.mesh())
        usol  = dl.Function(me.Vh2)
        A, b  = dl.assemble_system(aform, Lform, [])
        dl.solve(A, usol.vector(), b)
        return usol.vector()[:]




"""
An object which will internally handle the finite-element
discretization of the infinite-dimensional inverse problem
and set up the linear KKT system.
"""    

class inverseRHS:
    def __init__(me, Vh, Vh1, Vh2, beta, gamma1, gamma2, ud, rhol):
        
        me.Vh    = Vh
        me.Vh1   = Vh1
        me.Vh2   = Vh2
        me.rhol  = rhol
        
        # n1 dofs for state
        # n2 dofs for parameter
        me.n1    = me.Vh.sub(0).dim()
        me.n2    = me.Vh.sub(1).dim()
        me.n     = me.n1 + me.n2
        me.m     = me.n1
        
        # data that we are fitting the model to
        me.ud    = ud

              
        me.beta   = beta
        me.gamma1 = gamma1
        me.gamma2 = gamma2
        utest,  mtest = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        Rform = (dl.Constant(gamma1) * mtest * mtrial +\
                 dl.Constant(gamma2) * dl.inner(dl.grad(mtest), dl.grad(mtrial))) * dl.dx(me.Vh.mesh())
        Mform = (utest * utrial + mtest * mtrial) * dl.dx(me.Vh.mesh())
       
        me.R_fenics  = dl.assemble(Rform)
        me.R         = me.R_fenics.array()[me.n1:, me.n1:]
        me.M_fenics  = dl.assemble(Mform)
        me.Mx        = me.M_fenics.array()
        me.Mu        = me.Mx[:me.n1,:me.n1]
        me.Mm        = me.Mx[me.n1:,me.n1:]

    """
    In what follows x will be a function on the state-parameter product
    finite-element space
    """
    
    """
    objective -- return the value of the regularized data-misfit functional at x
    """
    def f(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        return dl.assemble((dl.Constant(0.5) * (X.sub(0)-me.ud)**2. + \
                           dl.Constant(me.gamma1/2.)*X.sub(1)*X.sub(1) + \
                           dl.Constant(me.gamma2/2.)*dl.inner(\
                           dl.grad(X.sub(1)), dl.grad(X.sub(1))))*dl.dx(me.Vh.mesh()))
    """
    gradient -- return the variational derivative of J with respect to x
    """
    def Dxf(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        utest, rhotest = dl.TestFunctions(me.Vh)
        return dl.assemble(((X.sub(0) - me.ud) * utest + \
                            dl.Constant(me.gamma1) * X.sub(1) * rhotest + \
                            dl.Constant(me.gamma2) * \
                            dl.inner(dl.grad(X.sub(1)), dl.grad(rhotest)))*dl.dx(me.Vh.mesh())).get_local()
    def Dxxfform(me, x):
        utest, mtest = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        return (utrial*utest + \
                                dl.Constant(me.gamma1)*\
                                mtrial*mtest + \
                                dl.Constant(me.gamma2)*\
                                dl.inner(dl.grad(mtest), dl.grad(mtrial)))*dl.dx(me.Vh.mesh())
    """ 
    return the second variational derivative of J with respect to x, that is 
    a linear mapping from primal to dual 
    """
    def Dxxf(me, x):
        return dl.assemble(me.Dxxfform(x)).array()
    """
    constraints -- evaluate the PDE-constraint at x, returning a dual-vector 
    """    
    def c(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        utest, mtest = dl.TestFunctions(me.Vh)
        return dl.assemble((dl.inner(dl.grad(X.sub(0)), dl.grad(utest)) + \
                            dl.Constant(me.beta)*X.sub(0)*utest - X.sub(1)*utest)*dl.dx(me.Vh.mesh()))\
                            .get_local()[:me.n1]
    def theta(me, x):
        return np.linalg.norm(me.c(x), 2)
    """
    evaluate the variational derivative of the PDE-constraint with respect
    to x, 
    return a linear mapping from the primal to the dual
    """
    def Dxc(me, x):
        X = dl.Function(me.Vh)
        X.vector().set_local(x)
        p, _   = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        J = csr_fenics2scipy(\
                  dl.assemble((dl.inner(dl.grad(utrial), dl.grad(p)) + \
                                dl.Constant(me.beta)*utrial*p - mtrial*p)*\
                               dl.dx(me.Vh.mesh())))
        J = J[:me.n1,:]
        return J
    # constraint Jacobian
    def Dxxcpform(me, x, p):
        X = dl.Function(me.Vh)
        P = dl.Function(me.Vh)
        P.vector().vec()[:me.n1] = p[:]
        X.vector().set_local(x)
        utest, mtest   = dl.TestFunctions(me.Vh)
        utrial, mtrial = dl.TrialFunctions(me.Vh)
        return (dl.Constant(0.0)*(mtest * utrial + mtrial * utest))*\
                                            dl.dx(me.Vh.mesh())
    def Dxxcp(me, x, p):
        return dl.assemble(me.Dxxcpform(x, p))
    def phi(me, x, mu):
        return me.f(x) - mu * sum(np.log(x[me.n1:] - me.rhol))
    def Dxphi(me, x, mu):
        y = np.zeros(me.n)
        y += me.Dxf(x)
        y[me.n1:] += -mu / (x[me.n1:] - me.rhol)
        return y
    def L(me, X):
        x, lam, z = X[:]
        return (me.f(x) + np.inner(lam, me.c(x)) - np.inner(z, x[me.n1:]-me.rhol))
    def DxL(me, X):
        x, lam, z = X[:]
        y = np.zeros(me.n)
        J = me.Dxc(x)
        JT = J.transpose()
        y = me.Dxf(x) + JT.dot(lam)#np.dot(lam.T, me.Dxc(x))
        y[me.n1:] -= z[:]
        return y
    def DxxL(me, X):
        x, lam, z = X[:]
        y = csr_fenics2scipy(dl.assemble(me.Dxxfform(x) + me.Dxxcpform(x, lam)))
        return y
    def E(me, X, mu, smax):
        x, lam, z = X[:]
        rho = x[me.n1:]
        E1 = np.linalg.norm(me.DxL(X), np.inf)
        E3 = np.linalg.norm((rho - me.rhol)*z-mu, np.inf)
        laml1 = np.linalg.norm(lam, 1)
        zl1   = np.linalg.norm(z,   1)
        sd    = max(smax, (laml1 + zl1) / (me.m + me.n)) / smax
        sc    = max(smax, zl1 / me.n) / smax
        E2    = np.linalg.norm(me.c(x), np.inf)
        return max(E1 / sd, E2, E3 / sc), E1, E2, E3
    def restore_feasibility(me, x):
        u = x[:me.n1]
        rho = x[me.n1:]
        utest  = dl.TestFunction(me.Vh2)
        utrial = dl.TrialFunction(me.Vh2)
        rhofunc = dl.Function(me.Vh1)
        rhofunc.vector()[:] = rho[:]
        aform = (dl.inner(dl.grad(utest), dl.grad(utrial)) + me.beta * utest * utrial) * dl.dx(me.Vh2.mesh())
        Lform = rhofunc * utest * dl.dx(me.Vh2.mesh())
        usol  = dl.Function(me.Vh2)
        A, b  = dl.assemble_system(aform, Lform, [])
        dl.solve(A, usol.vector(), b)
        return usol.vector()[:]

