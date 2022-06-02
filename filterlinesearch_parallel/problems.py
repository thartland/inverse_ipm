import numpy as np


try:
    import dolfin as dl
    # False - natural ordering; True - interlace
    dl.parameters['reorder_dofs_serial'] = False
except:
    Warning("FEniCS not found in current Python environment.")

"""
An object which will internally handle the finite-element
discretization of the infinite-dimensional inverse problem
and set up the linear KKT system.
"""

class inverseDiffusion:
    def __init__(me, Vh, Vhm, Vhu, beta, gamma1, gamma2, ud, g, rhol, B=None):
        me.sparse_struct = True
        me.Vh    = Vh
        me.Vhm   = Vhm
        me.Vhu   = Vhu
        me.rhol  = rhol

        me.comm = me.Vh.mesh().mpi_comm()

        # ---- primal variable x in R^n, n = n1 + n2
        # ---- n2 number of components which have an inequality constraint
        # ---- m is the number of equality constraints
        me.n1    = me.Vh.sub(0).dim()
        me.n2    = me.Vh.sub(1).dim()
        me.n     = me.n1 + me.n2
        me.m     = me.n1

        # ---- data that the model is fit to
        me.ud    = ud

        # ---- data defining the partial differential equality constraint
        # ---- -\nabla^2 u + \beta u = g in \Omega
        # ---- du / dn               = 0 on \partial\Omega
        me.g = g
        me.beta   = beta
        
        # ---- define mass-matrices
        utes   = dl.TestFunction(me.Vhu)
        utri   = dl.TrialFunction(me.Vhu)
        MuForm = (utes * utri) * dl.dx(me.Vhu.mesh())
        me.Mu  = dl.assemble(MuForm)
        
        mtes   = dl.TestFunction(me.Vhm)
        mtri   = dl.TrialFunction(me.Vhm)
        MmForm = (mtes * mtri) * dl.dx(me.Vhm.mesh())
        me.Mm  = dl.assemble(MmForm)

        # ---- lumped mass-matrix
        one    = dl.Vector()
        me.MlumpedVec = dl.Vector()
        me.Mm.init_vector(one, 1)
        me.Mm.init_vector(me.MlumpedVec, 0)
        one.set_local(np.ones(one.local_size()))
        one.apply("insert")
        me.Mm.mult(one, me.MlumpedVec)
        me.MlumpedVec.apply("insert")

        # ---- regularization
        me.gamma1 = gamma1
        me.gamma2 = gamma2

        Rform = (dl.Constant(gamma1) * mtes * mtri +\
                 dl.Constant(gamma2) * dl.inner(dl.grad(mtes), dl.grad(mtri))) * dl.dx(me.Vhm.mesh())
        me.R  = dl.assemble(Rform)

        if B is None:
            me.sparse_obs = False
        else:
            me.B = B
            me.sparse_obs = True
        #me.d   = me.ud.vector()
        
        # ---- helper vectors
        me.u      = dl.Vector()
        me.help1u = dl.Vector()
        me.help2u = dl.Vector()
        me.Mu.init_vector(me.help1u, 1)
        me.Mu.init_vector(me.help2u, 0)
        if me.sparse_obs:
            me.Bu     = dl.Vector()
            me.B.init_vector(me.Bu, 0)
        me.Mu.init_vector(me.u, 1)
        me.help1m = dl.Vector()
        me.help2m = dl.Vector()
        me.rho    = dl.Vector()
        me.Mm.init_vector(me.rho, 1)
        me.Mm.init_vector(me.help1m, 1)
        me.Mm.init_vector(me.help2m, 0)


    """
    In what follows x will be a function on the state-parameter product
    finite-element space
    """

    """
    objective -- return the value of the regularized data-misfit functional at x
    """
    def f(me, x):
        # ---- need a generic routine, how to take a primal-vector x
        # ---- distributed according to the state-parameter product space

        # ---- partition the data
        X = me.comm.bcast(x.gather_on_zero())
        me.u.set_local(X[::2][me.u.local_range()[0]: me.u.local_range()[1]])
        me.u.apply("insert")
        me.rho.set_local(X[1::2][me.rho.local_range()[0]: me.rho.local_range()[1]])
        me.rho.apply("insert")

        # ---- regularization
        me.help1m.zero()
        me.help1m.axpy(1.0, me.rho)
        me.R.mult(me.help1m, me.help2m)

        # ---- data-discrepancy
        me.help1u.zero()
        me.help1u.axpy(1.0, me.u)
        me.help1u.axpy(-1.0, me.ud)
        me.help1u.apply("insert")
        # ---- misfit component of objective
        if me.sparse_obs:
            me.B.mult(me.help1u, me.Bu)
            me.Bu.apply("insert")
            return 0.5 * me.Bu.norm('l2')**2.  + me.help2m.inner(me.help1m) / 2.
        else:
            me.Mu.mult(me.help1u, me.help2u)
            me.help2u.apply("insert")
            return 0.5 * me.help2u.inner(me.help1u) + me.help2m.inner(me.help1m) / 2.
    def checkMass(me, u):
        me.u.set_local(u)
        me.u.apply("insert")
        me.help1u.zero()
        me.help1u.axpy(1.0, u)
        me.help1u.apply("insert")
        me.Mu.mult(me.help1u, me.help2u)
        me.help2u.apply("insert")
        return me.help1u.inner(me.help2u)

    """
    gradient -- return the variational derivative of J with respect to x
    """
    def Dxf(me, x):
        # ---- partition the data
        X = me.comm.bcast(x.gather_on_zero())
        me.u.set_local(X[::2][me.u.local_range()[0]: me.u.local_range()[1]])
        me.u.apply("insert")
        me.rho.set_local(X[1::2][me.rho.local_range()[0]: me.rho.local_range()[1]])
        me.rho.apply("insert")
        
        # ---- gradient computation
        # ------ misfit component
        me.help1u.zero()
        me.help1u.axpy(1.0, me.u)
        me.help1u.axpy(-1.0, me.ud)
        me.help1u.apply("insert")
        if me.sparse_obs:
            # B^T B (u - d)
            me.B.mult(me.help1u, me.Bu)
            me.Bu.apply("insert")
            me.B.transpmult(me.Bu, me.help1u)
            me.help1u.apply("insert")
        else:
            # M (u - d)
            me.Mu.mult(me.help1u, me.help2u)
            me.help2u.apply("insert")
            me.help1u.zero()
            me.help1u.axpy(1.0, me.help2u)
            me.help1u.apply("insert")
        # ------ regularization component
        me.help1m.zero()
        me.R.mult(me.rho, me.help1m)
        me.help1m.apply("insert")

        # ---- concatenate the data
        G = np.zeros(me.Vh.dim())
        G[::2]  = me.comm.bcast(me.help1u.gather_on_zero())
        G[1::2] = me.comm.bcast(me.help1m.gather_on_zero())
        g = x.copy()
        g.zero()
        g.set_local(G[g.local_range()[0]: g.local_range()[1]])
        g.apply("insert")
        return g
    #"""
    #return the second variational derivative of J with respect to x, that is
    #a linear mapping from primal to dual
    #"""
    #def Dxxf(me, x):
    #    if me.sparse_obs:
    #        y = sps.bmat([[me.BT.dot(me.B)/me.sig**2., None], [None, me.R]], format="csr")
    #    else:
    #        y = sps.bmat([[me.Mu, None], [None, me.R]], format="csr")
    #    return y
    #"""
    #constraints -- evaluate the PDE-constraint at x, returning a dual-vector
    #"""
    #def c(me, x):
    #    X = dl.Function(me.Vh)
    #    X.vector().set_local(x)
    #    utest, mtest = dl.TestFunctions(me.Vh)
    #    return dl.assemble((X.sub(1)*dl.inner(dl.grad(X.sub(0)), dl.grad(utest)) + \
    #                        dl.Constant(me.beta)*X.sub(0)*utest - me.g*utest)*dl.dx(me.Vh.mesh()))\
    #                        .get_local()[:me.n1]
    #def theta(me, x):
    #    return np.linalg.norm(me.c(x), 2)
    #"""
    #evaluate the variational derivative of the PDE-constraint with respect
    #to x,
    #return a linear mapping from the primal to the dual
    #"""
    #def Dxc(me, x):
    #    X = dl.Function(me.Vh)
    #    X.vector().set_local(x)
    #    p, _   = dl.TestFunctions(me.Vh)
    #    utrial, mtrial = dl.TrialFunctions(me.Vh)
    #    J = csr_fenics2scipy(\
    #              dl.assemble((X.sub(1)*dl.inner(dl.grad(utrial), dl.grad(p)) + \
    #                            mtrial*dl.inner(dl.grad(X.sub(0)), dl.grad(p))+ dl.Constant(me.beta)*utrial*p)*\
    #                           dl.dx(me.Vh.mesh())))
    #    J = J[:me.n1,:]
    #    return J
    ## constraint Jacobian
    #def Dxxcpform(me, x, p):
    #    X = dl.Function(me.Vh)
    #    P = dl.Function(me.Vh)
    #    P.vector().vec()[:me.n1] = p[:]
    #    X.vector().set_local(x)
    #    utest, mtest   = dl.TestFunctions(me.Vh)
    #    utrial, mtrial = dl.TrialFunctions(me.Vh)
    #    return (mtest * dl.inner(dl.grad(utrial), dl.grad(P.sub(0))) + \
    #                                         mtrial * dl.inner(dl.grad(utest), dl.grad(P.sub(0))))*\
    #                                        dl.dx(me.Vh.mesh())
    #def Dxxcp(me, x, p):
    #    return dl.assemble(me.Dxxcpform(x, p))
    #def phi(me, x, mu):
    #    return me.f(x) - mu * sum(me.Mm.dot(np.log(x[me.n1:] - me.rhol)))
    #def Dxphi(me, x, mu):
    #    y = np.zeros(me.n)
    #    y += me.Dxf(x)
    #    one = np.ones(me.Mm.shape[0])
    #    y[me.n1:] -= mu * me.Mlumpedvec / (x[me.n1:] - me.rhol)
    #    #y[me.n1:] += -me.Mm.dot(mu / (x[me.n1:] - me.rhol))
    #    return y 
    #def L(me, X):
    #    x, lam, z = X[:]
    #    return (me.f(x) + np.inner(lam, me.c(x)) - np.inner(z, me.Mlumped.dot(x[me.n1:] - me.rhol)))
    #    #return (me.f(x) + np.inner(lam, me.c(x)) - np.inner(z, me.Mm.dot(x[me.n1:]-me.rhol)))
    #def DxL(me, X):
    #    x, lam, z = X[:]
    #    y = np.zeros(me.n)
    #    y = me.Dxf(x) + me.Dxc(x).transpose().dot(lam)#np.dot(lam.T, me.Dxc(x))
    #    y[me.n1:]  -= me.Mlumpedvec * z
    #    #y[me.n1:] -= me.Mm.dot(z[:])
    #    return y
    #def DxxL(me, X):
    #    x, lam, z = X[:]
    #    y = me.Dxxf(x) + csr_fenics2scipy(dl.assemble(me.Dxxcpform(x, lam)))    
    #    return y
    #def E(me, X, mu, smax):
    #    x, lam, z = X[:]
    #    rho = x[me.n1:]
    #    DxL = me.DxL(X)
    #    cx  = me.c(x)
    #    E1 = np.sqrt(np.inner(DxL, spla.spsolve(me.Mx, DxL)))
    #    E2 = np.sqrt(np.inner(cx, spla.spsolve(me.Mu, cx)))
    #    #E3 = np.linalg.norm((rho-me.rhol)*z - mu, np.inf)
    #    E3    = sum(me.Mm.dot(np.abs(rho - me.rhol)*z - mu))
    #    lamL2 = np.sqrt(np.inner(lam, me.Mu.dot(lam)))
    #    zL2   = np.sqrt(np.inner(z, me.Mlumped.dot(z)))
    #    sc    = max(smax, zL2) / smax
    #    sd    = max(smax, lamL2 / 2. + zL2 / 2.) / smax
    #    return max(E1 / sd, E2, E3 / sc), E1, E2, E3
    #def restore_feasibility(me, x):
    #    u = x[:me.n1]
    #    rho = x[me.n1:]
    #    utest  = dl.TestFunction(me.Vh2)
    #    utrial = dl.TrialFunction(me.Vh2)
    #    rhofunc = dl.Function(me.Vh1)
    #    rhofunc.vector()[:] = rho[:]
    #    aform = (dl.inner(dl.grad(utest), dl.grad(utrial)) + me.beta * utest * utrial) * dl.dx(me.Vh2.mesh())
    #    Lform = me.g * utest * dl.dx(me.Vh2.mesh())
    #    usol  = dl.Function(me.Vh2)
    #    A, b  = dl.assemble_system(aform, Lform, [])
    #    dl.solve(A, usol.vector(), b)
    #    return usol.vector()[:]




