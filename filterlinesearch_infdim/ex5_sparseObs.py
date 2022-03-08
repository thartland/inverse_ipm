#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import dolfin as dl
# False - natural ordering; True - interlace
dl.parameters['reorder_dofs_serial'] = False


from filterLineSearch import interior_pt
from hippylib import nb

from problems import inverseDiffusion

from helperfunctions import *

from hippylib import PointwiseStateObservation


# In[2]:


plt.style.use('classic')
plt.rcParams.update({'font.size': 16})


# In[27]:


nx = 60 


# coarse and fine meshes
mesh = dl.UnitSquareMesh(nx, nx)

# In[28]:


P1_deg = 1 # degree of finite-element polynomials for parameter (rho)
P2_deg = 1 # degree of finite-element polynomials for state

DG_rho = False
if DG_rho:
    P1 = dl.FiniteElement("DG", mesh.ufl_cell(), P1_deg)
else:
    P1 = dl.FiniteElement("CG", mesh.ufl_cell(), P1_deg)
P2 = dl.FiniteElement("CG", mesh.ufl_cell(), P2_deg)
Th = dl.MixedElement([P2, P1])
Vh = dl.FunctionSpace(mesh, Th)
Vh1 = dl.FunctionSpace(mesh, P1)
Vh2 = dl.FunctionSpace(mesh, P2)

print("dim(state) = {0:d}, dim(parameter) = {1:d}".format(Vh.sub(0).dim(), Vh.sub(1).dim()))


# In[29]:


beta   = 1.
sparse_obs = True
if P1_deg == 0:
    if sparse_obs:
        gamma1 = 1.
    else:
        gamma1 = 1.e-5
    gamma2 = 0.
else:
    gamma1 = 1.e-1
    gamma2 = 1.e-2
Crhol  = 0.75


rhol = dl.interpolate(dl.Expression('C', element=Vh1.ufl_element(), C=Crhol), Vh1).vector()[:]
         
ud   = dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',                                          pi=np.pi, element=Vh2.ufl_element()), Vh2)
g    = dl.interpolate(dl.Expression('(2.*pi*pi*(0.5+x[0]) + beta)*std::cos(x[0]*pi)*std::cos(x[1]*pi)'+                                          '+pi*std::sin(pi*x[0])*std::cos(pi*x[1])',                                           pi=np.pi, beta=beta, element=Vh2.ufl_element()), Vh2)



if sparse_obs:
    ntargets = 50
    rel_noise = 0.01

    #Targets only on the bottom
    np.random.seed(1)
    targets_x = np.random.uniform(0.05,0.95, [ntargets] )
    targets_y = np.random.uniform(0.05,0.5, [ntargets] )
    targets = np.zeros([ntargets, 2])
    targets[:,0] = targets_x
    targets[:,1] = targets_y

    dl.plot(mesh)
    plt.plot(targets_x, targets_y, '*')
    plt.show()
    
    test = dl.TestFunction(Vh1)
    trial = dl.TrialFunction(Vh1)
    K = dl.assemble(dl.inner(dl.grad(test), dl.grad(trial))*dl.dx(mesh))
    print(np.linalg.norm(K.array()))

#targets everywhere
#targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    print( "Number of observation points: {0}".format(ntargets) )
    misfit = PointwiseStateObservation(Vh2, targets)
    B      = sps.csr_matrix(csr_fenics2scipy(misfit.B), shape=(ntargets, Vh2.dim()))
    problem = inverseDiffusion(Vh, Vh1, Vh2, beta, gamma1, gamma2, ud, g, rhol, B=B)
else:
    print("Observations everywhere!")
    problem = inverseDiffusion(Vh, Vh1, Vh2, beta, gamma1, gamma2, ud, g, rhol)



rhotrue = dl.interpolate(dl.Expression('0.5+x[0]',                                      element=Vh1.ufl_element()), Vh1)   


# In[30]:


# ---- gradient check

  
F     = lambda x : problem.phi(x, 0.1)
gradF = lambda x : problem.Dxphi(x, 0.1)

# initial point
x0   = dl.interpolate(dl.Expression(('std::cos(x[0]*pi)*std::cos(x[1]*pi)', '1.0+x[0]*x[1]'),                                          pi=np.pi, element=Vh.ufl_element()), Vh).vector()[:]

xhat   = dl.interpolate(dl.Expression(('std::sin(x[0]*pi)*std::cos(x[1]*pi)', '1.0+x[0]*x[1]'),                                          pi=np.pi, element=Vh.ufl_element()), Vh).vector()[:]

grad_check(F, gradF, x0, xhat, save=False)


# In[33]:


solver = interior_pt(problem, linsolve_strategy="presmoothing")
x0      = np.ones(problem.n)
x0[:problem.n1] = problem.restore_feasibility(x0)
X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
solver.initialize(X0)
mu0 = 1.e0
tol = 1.e-8
max_it = 50
Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
xf, lamf, zf = Xf[:]
GMRESresiduals = [len(solver.residuals[i]) for i in range(len(solver.residuals))]
print("{0:d} total Krylov iterations".format(sum(GMRESresiduals)))
print("{0:1.1f} average Krylov iterations per IP-Newton system solve".format(sum(GMRESresiduals)/len(GMRESresiduals)))



