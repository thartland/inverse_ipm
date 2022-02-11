import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla
try:
    import dolfin as dl
    # False - natural ordering; True - interlace
    dl.parameters['reorder_dofs_serial'] = False
    from helperfunctions import csr_fenics2scipy
except:
    Warning("FEniCS not found in current Python environment.")

"""
An object which will internally handle the finite-element
discretization of the infinite-dimensional inverse problem
and set up the linear KKT system.
"""

class inverseDiffusion:
    def __init__(me, Vh, Vh1, Vh2, beta, gamma1, gamma2, ud, g, rhol):
        me.sparse_struct = True
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
        me.sparse_struct = True
        
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




"""
An object which will internally handle the finite-element
discretization of the infinite-dimensional inverse problem
and set up the linear KKT system.
"""    

class inverseDiffusionDirichlet:
    def __init__(me, Vh, Vh1, Vh2, bc, gamma1, gamma2, ud, g, rhol, idxs):
        me.sparse_struct = True
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
        
        # Dirichlet condition indicies
        me.idxs = idxs
        
        # data that we are fitting the model to
        me.ud    = ud

        # rhs
        me.g = g
        
        # bc
        me.bc = bc
        
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
        PDEform = dl.assemble((X.sub(1)*dl.inner(dl.grad(X.sub(0)), dl.grad(utest)) - \
                            me.g*utest)*dl.dx(me.Vh.mesh()))
        me.bc.apply(PDEform)
        PDEform = PDEform.get_local()
        PDEform[me.idxs] = x[me.idxs]
        return PDEform[:me.n1]
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
        PDEform = dl.assemble((X.sub(1)*dl.inner(dl.grad(utrial), dl.grad(p)) + \
                                mtrial*dl.inner(dl.grad(X.sub(0)), dl.grad(p)))*\
                               dl.dx(me.Vh.mesh()))
        me.bc.apply(PDEform)
        J = csr_fenics2scipy(PDEform).todense()
        J = np.array(J[:me.n1,:])
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
        PDEform = dl.assemble(me.Dxxcpform(x,p))
        me.bc.apply(PDEform)
        PDEform = PDEform.array()
        PDEform[me.idxs,:] *= 0.
        return PDEform
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
        y = me.Dxf(x) + np.dot(lam.T, me.Dxc(x))
        y[me.n1:] -= z[:]
        return y
    def DxxL(me, X):
        x, lam, z = X[:]
        y = np.zeros((me.n, me.n))
        y[:, :] += me.Dxxf(x)
        y[:, :] += me.Dxxcp(x, lam)[:,:]
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
        aform = (rhofunc * dl.inner(dl.grad(utest), dl.grad(utrial)) + me.beta * utest * utrial) * dl.dx(me.Vh2.mesh())
        Lform = me.g * utest * dl.dx(me.Vh2.mesh())
        usol  = dl.Function(me.Vh2)
        A, b  = dl.assemble_system(aform, Lform, [])
        dl.solve(A, usol.vector(), b)
        return usol.vector()[:]



"""
In this example problem we seek to solve

min_(x1, x2, x3, x4) f(x1, x2, x3, x4) := x1 * x4 * (x1 + x2 + x3) + x_2

s.t.
1 <= xj <=5, j = 1,2,3,4
      c1(x1, x2, x3, x4) = 0
25 <= C2(x1, x2, x3, x4)

c1(x1, x2, x3, x4) := sum_j xj^2 - 40
C2(x1, x2, x3, x4) := x1 * x2 * x3 * x4

slack variables are then introduced
x5 = 5 - x1
x6 = 5 - x2
x7 = 5 - x3
x8 = 5 - x4
x9 = C2(x1, x2, x3, x4) - 25

so that we then seek the solution of

min_(x1, x2, x3, x4, x5, x6, x7, x8, x9) f(x1, x2, x3, x4)

s.t.
1 <= xj, j = 1,2,3,4
0 <= xj, j = 5,6,7,8,9
cj(x) = 0, j=1,2,..,6
c1(x) := sum_(1<=j<=4) xj^2 - 40
c2(x) := x9 + C2(x1, x2, x3, x4) - 25
c3(x) := x1 + x5 - 5
c4(x) := x2 + x6 - 5
c5(x) := x3 + x7 - 5
c6(x) := x4 + x8 - 5
"""

class hs071:
    def __init__(self):
        self.sparse_struct = False
        self.n1   = 0
        self.n2   = 9
        self.n    = self.n1 + self.n2
        self.m    = 6
        self.rhol = np.zeros(self.n2)
        self.rhol[:4] = 1.
    def f(self, x):
        return x[0]*x[3]*sum(x[:3]) + x[2]
    def Dxf(self, x):
        g = np.zeros(self.n)
        g[0] = x[3]*sum(x[:3]) + x[0] * x[3]
        g[1] = x[0]*x[3]
        g[2] = 1. + x[0]*x[3]
        g[3] = x[0]*sum(x[:3])
        return g
    def Dxxf(self, x):
        H    = np.zeros((self.n, self.n))
        H[0, 0] = 2.*x[3]
        H[0, 1:3] = x[3]
        H[0, 3]   = 2.*x[0] + x[1] + x[2]
        H[1:3, 0] = x[3]
        H[3, 0]   = 2.*x[0] + x[1] + x[2]
        H[3, 1:3] = x[0]
        H[1:3, 3] = x[0]
        return H
    def c(self, x):
        y = np.zeros(self.m)
        y[0] = np.inner(x[:4], x[:4]) - 40.
        y[1] = x[8] - np.prod(x[:4]) + 25.
        y[2:] = x[:4] + x[4:8] - 5.
        return y
    def theta(self, x):
        return np.linalg.norm(self.c(x), 2)
    def Dxc(self, x):
        J = np.zeros((self.m, self.n))
        J[0,:4] =  2. * x[:4]
        J[1, 0] = -1. * np.prod(x[1:4])
        J[1, 1] = -1. * x[0]*np.prod(x[2:4])
        J[1, 2] = -1. * np.prod(x[:2])*x[3]
        J[1, 3] = -1. * np.prod(x[:3])
        J[1, 8] =  1.
        for j in range(4):
            J[j + 2, j]     = 1.
            J[j + 2, j + 4] = 1.
        return J
    def Dxxc(self, x):
        y = np.zeros((self.m, self.n, self.n))
        y[0][:4,:4] = 2. * np.identity(4)
        y[1][:4, :4]  = -1.*np.array([[0.       , x[2]*x[3], x[1]*x[3], x[1]*x[2]],\
                                [x[2]*x[3], 0.       , x[0]*x[3], x[0]*x[2]],\
                                [x[1]*x[3], x[0]*x[3], 0.       , x[0]*x[1]],\
                                [x[1]*x[2], x[0]*x[2], x[0]*x[1], 0.]]) 
        return y
    def phi(self, x, mu):
        return self.f(x) - mu * sum(np.log(x[self.n1:] - self.rhol))
    def Dxphi(self, x, mu):
        return self.Dxf(x) - mu / (x[self.n1:] - self.rhol)
    def L(self, X):
        x, lam, z = X[:]
        return (self.f(x) + np.inner(lam, self.c(x)) - np.inner(z, x[self.n1:]-self.rhol))
    def DxL(self, X):
        x, lam, z = X[:]
        y = np.zeros(self.n)
        y = self.Dxf(x) + np.dot(lam.T, self.Dxc(x))
        y[self.n1:] -= z[:]
        return y
    def DxxL(self, X):
        x, lam, z = X[:]
        y = np.zeros((self.n, self.n))
        y[:, :] += self.Dxxf(x)
        y[:, :] += np.tensordot(self.Dxxc(x), lam, axes=([0,0]))
        return y
    def E(self, X, mu, smax):
        x, lam, z = X[:]
        rho = x[self.n1:]
        E1 = np.linalg.norm(self.DxL(X), np.inf)
        E3 = np.linalg.norm((rho - self.rhol)*z-mu, np.inf)
        laml1 = np.linalg.norm(lam, 1)
        zl1   = np.linalg.norm(z,   1)
        sd    = max(smax, (laml1 + zl1) / (self.m + self.n)) / smax
        sc    = max(smax, zl1 / self.n) / smax
        if self.m > 0:
            E2 = np.linalg.norm(self.c(x), np.inf)
            return max(E1 / sd, E2, E3 / sc), E1, E2, E3
        else:
            return max(E1 / sd, E3 / sc)
    def restore_feasibility(self, X):
        x, lam, z = X[:]
        A = np.zeros((self.n+self.m, self.n+self.m))
        A[:self.n, :self.n] = np.identity(self.n)
        J = self.Dxc(x)
        A[self.n:self.n+self.m, :self.n] = J[:,:]
        A[:self.n, self.n:self.n+self.m] = (J.T)[:,:]
        r = np.zeros(self.n+self.m)
        r[self.n:self.n+self.m] = self.c(x)[:]
        sol = np.linalg.solve(A, -r)
        xhat = sol[:self.n]
        lamhat = sol[self.n:self.n+self.m]
        return x+xhat, lam+lamhat, z




"""
Here we seek to solve

min_(x in R^n) f(x) := 1/2 x^T A x
s.t. 
0 <= xj <= 1, j=1,2,...,n
C(x) = 0
C(x) := sum_(1<=j<=n)(xj) - 1

we accomodate the upper-bound constraints by introducing slack
variables
yj = 1 - xj, j=1,2,...n

so that we then solve the following equivalent reformulated problem
min_((x, y) in R^n x R^n) f(x, y) := 1/2 x^T A x
s.t.
0 <= xj,    j=1,2,...,n
0 <= yj,    j=1,2,...,n
cj(x) = 0,  j=1,2,...,n+1
cj(x)     := xj + yj - 1, j=1,2,...,n
c(n+1)(x) := sum_(1<=j<=n)(xj) - 1

It is important to remark that the matrix A, which defines the objective function, need not be
positive definite. However, if it is positive definite then we know it's solution.
"""


class example2:
    def __init__(self, nx, A):
        self.sparse_struct = False
        self.A    = A
        self.nx   = nx
        self.n1   = 0
        self.n2   = 2 * self.nx
        self.n    = self.n1 + self.n2
        self.m    = self.nx + 1
        self.rhol = np.zeros(self.n2)
    def f(self, x):
        return 0.5 * np.inner(x[:self.nx], np.dot(self.A, x[:self.nx]))
    def Dxf(self, x):
        y = np.zeros(self.n)
        y[:self.nx] = np.dot(self.A, x[:self.nx])
        return y
    def Dxxf(self, x):
        y = np.zeros((self.n, self.n))
        y[:self.nx, self.nx:] = self.A[:,:]
        return y
    def c(self, x):
        y = np.zeros(self.m)
        for i in range(self.nx):
            y[i] = x[i] + x[i+self.nx] - 1.
        y[self.nx] = sum(x[:self.nx]) - 1.
        return y
    def theta(self, x):
        return np.linalg.norm(self.c(x), 2)
    def Dxc(self, x):
        J = np.zeros((self.m, self.n))
        for i in range(self.nx):
            J[i, i]     = 1.
            J[i, i + self.nx] = 1.
        J[self.nx, :self.nx] = 1.
        return J
    def Dxxc(self, x):
        y = np.zeros((self.m, self.n, self.n))
        return y
    def phi(self, x, mu):
        return self.f(x) - mu * sum(np.log(x[self.n1:] - self.rhol))
    def Dxphi(self, x, mu):
        return self.Dxf(x) - mu / (x[self.n1:] - self.rhol)
    def L(self, X):
        x, lam, z = X[:]
        rho = x[self.n1:]
        return (self.f(x) + np.inner(lam, self.c(x)) - np.inner(z, rho - self.rhol))
    def DxL(self, X):
        x, lam, z = X[:]
        y = np.zeros(self.n)
        y = self.Dxf(x) + np.dot(lam.T, self.Dxc(x))
        y[self.n1:] -= z[:]
        return y
    def DxxL(self, X):
        x, lam, z = X[:]
        y = np.zeros((self.n, self.n))
        y[:, :] += self.Dxxf(x)
        y[:, :] += np.tensordot(self.Dxxc(x), lam, axes=([0,0]))
        return y
    def E(self, X, mu, smax):
        x, lam, z = X[:]
        rho = x[self.n1:]
        E1 = np.linalg.norm(self.DxL(X), np.inf)
        E3 = np.linalg.norm((rho - self.rhol)*z-mu, np.inf)
        laml1 = np.linalg.norm(lam, 1)
        zl1   = np.linalg.norm(z,   1)
        sd    = max(smax, (laml1 + zl1) / (self.m + self.n2)) / smax
        sc    = max(smax, zl1 / self.n2) / smax
        if self.m > 0:
            E2 = np.linalg.norm(self.c(x), np.inf)
            return max(E1 / sd, E2, E3 / sc), E1, E2, E3
        else:
            return max(E1 / sd, E3 / sc)


class example3:
    def __init__(self, nx, K, M, rhol, Omegaf, rho0, rho1):
        self.sparse_struct = False
        self.K    = K
        self.M    = M
        self.nx   = nx
        self.Omegaf = Omegaf
        self.rho0   = rho0
        self.rho1   = rho1
        self.n1   = 0
        self.n2   = self.nx + 1
        self.n    = self.n1 + self.n2
        self.m    = 3
        self.rhol = np.append(rhol, 0.) # add extra constraint for slack variable
    def f(self, x):
        return 0.5 * np.inner(x[:self.nx], np.dot(self.K, x[:self.nx]))
    def Dxf(self, x):
        y = np.zeros(self.n)
        y[:self.nx] = np.dot(self.K, x[:self.nx])
        return y
    def Dxxf(self, x):
        y = np.zeros((self.n, self.n))
        y[:self.nx, :self.nx] = self.K[:, :]
        return y
    def c(self, x):
        y = np.zeros(self.m)
        y[0] = sum(self.M.dot(x[:self.nx])) + x[-1] - self.Omegaf
        y[1] = x[0] - self.rho0
        y[2] = x[self.nx-1] - self.rho1
        return y
    def theta(self, x):
        return np.linalg.norm(self.c(x), 2)
    def Dxc(self, x):
        J = np.zeros((self.m, self.n))
        one = np.ones(self.nx)
        J[0, :self.nx]    = self.M.dot(one)
        J[0, self.nx]     = 1.
        J[1, 0]           = 1.
        J[2, self.nx - 1] = 1.
        return J
    def Dxxc(self, x):
        y = np.zeros((self.m, self.n, self.n))
        return y
    def phi(self, x, mu):
        return self.f(x) - mu * sum(np.log(x[self.n1:] - self.rhol))
    def Dxphi(self, x, mu):
        rho = x[self.n1:]
        return self.Dxf(x) - mu / (rho - self.rhol)
    def L(self, X):
        x, lam, z = X[:]
        rho = x[self.n1:]
        return (self.f(x) + np.inner(lam, self.c(x)) - np.inner(z, rho - self.rhol))
    def DxL(self, X):
        x, lam, z = X[:]
        y = np.zeros(self.n)
        y = self.Dxf(x) + np.dot(lam.T, self.Dxc(x))
        y[self.n1:] -= z[:]
        return y
    def DxxL(self, X):
        x, lam, z = X[:]
        y = np.zeros((self.n, self.n))
        y[:, :] += self.Dxxf(x)
        y[:, :] += np.tensordot(self.Dxxc(x), lam, axes=([0,0]))
        return y
    def E(self, X, mu, smax):
        x, lam, z = X[:]
        rho = x[self.n1:]
        E1 = np.linalg.norm(self.DxL(X), np.inf)
        E3 = np.linalg.norm((rho - self.rhol)*z-mu, np.inf)
        laml1 = np.linalg.norm(lam, 1)
        zl1   = np.linalg.norm(z,   1)
        sd    = max(smax, (laml1 + zl1) / (self.m + self.n2)) / smax
        sd    = 1.
        sc    = max(smax, zl1 / self.n2) / smax
        if self.m > 0:
            E2 = np.linalg.norm(self.c(x), np.inf)
            return max(E1 / sd, E2, E3 / sc), E1, E2, E3
        else:
            return max(E1 / sd, E3 / sc)



"""
Here we seek to solve

min_(x in R^n) f(x) := 0.5 x^T x
s.t.
0<(xl)j <= xj, j=1,...,n
c1(x) = x1 - C, C > (xl)1
"""



class example4:
    def __init__(self, nx):
        self.sparse_struct = False
        self.nx   = nx
        self.n1   = 0
        self.n2   = self.nx
        self.n    = self.n1 + self.n2
        self.m    = 1
        self.rhol = 0.1*np.ones(self.n2)
    def f(self, x):
        return 0.5 * np.inner(x[:self.nx], x[:self.nx])
    def Dxf(self, x):
        y = np.zeros(self.n)
        y[:self.nx] = x[:self.nx]
        return y
    def Dxxf(self, x):
        y = np.zeros((self.n, self.n))
        y[:self.nx, :self.nx] = np.identity(self.nx)
        return y
    def c(self, x):
        y = np.zeros(self.m)
        y[0] = x[0] - (self.rhol[0] + 1.0)
        return y
    def theta(self, x):
        return np.linalg.norm(self.c(x), 2)
    def Dxc(self, x):
        J = np.zeros((self.m, self.n))
        J[0,0] = 1.
        return J
    def Dxxc(self, x):
        y = np.zeros((self.m, self.n, self.n))
        return y
    def phi(self, x, mu):
        rho = x[self.n1:]
        return self.f(x) - mu * sum(np.log(rho - self.rhol))
    def Dxphi(self, x, mu):
        rho = x[self.n1:]
        return self.Dxf(x) - mu / (rho - self.rhol)
    def L(self, X):
        x, lam, z = X[:]
        rho = x[self.n1:]
        return (self.f(x) + np.inner(lam, self.c(x)) - np.inner(z, rho - self.rhol))
    def DxL(self, X):
        x, lam, z = X[:]
        y = np.zeros(self.n)
        y = self.Dxf(x) + np.dot(lam.T, self.Dxc(x))
        y[self.n1:] -= z[:]
        return y
    def DxxL(self, X):
        x, lam, z = X[:]
        y = np.zeros((self.n, self.n))
        y[:, :] += self.Dxxf(x) # objective Hessian
        y[:, :] += np.tensordot(self.Dxxc(x), lam, axes=([0,0])) # constraint Hessian
        return y
    def E(self, X, mu, smax):
        x, lam, z = X[:]
        rho = x[self.n1:]
        E1 = np.linalg.norm(self.DxL(X), np.inf)
        E3 = np.linalg.norm((rho - self.rhol)*z - mu, np.inf)
        laml1 = np.linalg.norm(lam, 1)
        zl1   = np.linalg.norm(z,   1)
        sd    = max(smax, (laml1 + zl1) / (self.m + self.n2)) / smax
        sc    = max(smax, zl1 / self.n2) / smax
        if self.m > 0:
            E2 = np.linalg.norm(self.c(x), np.inf)
            return max(E1 / sd, E2, E3 / sc), E1, E2, E3
        else:
            return max(E1 / sd, E3 / sc), E1, E3