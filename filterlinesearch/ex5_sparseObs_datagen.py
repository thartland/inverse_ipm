#!/usr/bin/env python
# coding: utf-8

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


# In[5]:


plt.style.use('classic')
plt.rcParams.update({'font.size': 16})


# In[6]:


nx_coarse = 5
nx = 2*nx_coarse

# coarse and fine meshes
mesh_coarse = dl.UnitSquareMesh(nx_coarse, nx_coarse)

mesh_fine   = dl.UnitSquareMesh(nx, nx)

meshes = [mesh_coarse, mesh_fine]
mesh   = meshes[-1]
lvl    = len(meshes) # depth of mesh hierarchy
dl.plot(mesh)
plt.title('Mesh of PDE domain')
plt.show()


# In[7]:


P1_deg = 1 # degree of finite-element polynomials for parameter (rho)
P2_deg = 1 # degree of finite-element polynomials for state

P1s  = [dl.FiniteElement("CG", meshes[i].ufl_cell(), P1_deg) for i in range(lvl)]
P2s  = [dl.FiniteElement("CG", meshes[i].ufl_cell(), P2_deg) for i in range(lvl)] 
Ths  = [dl.MixedElement([P2s[i], P1s[i]]) for i in range(lvl)]
Vhs  = [dl.FunctionSpace(meshes[i], Ths[i]) for i in range(lvl)]
Vh1s = [dl.FunctionSpace(meshes[i], P1s[i]) for i in range(lvl)]
Vh2s = [dl.FunctionSpace(meshes[i], P2s[i]) for i in range(lvl)]

P1  = P1s[-1]
P2  = P2s[-1]
Th  = Ths[-1]
Vh  = Vhs[-1]
Vh1 = Vh1s[-1]
Vh2 = Vh2s[-1]

print("dim(state) = {0:d}, dim(parameter) = {1:d}".format(Vh.sub(0).dim(), Vh.sub(1).dim()))


# In[8]:


beta   = 1.
gamma1 = 1.e-2
gamma2 = 1.e-1
Crhol  = 0.75


rhols = [dl.interpolate(dl.Expression('C', element=Vh1s[i].ufl_element(), C=Crhol), Vh1s[i]).vector().get_local()          for i in range(lvl)]
uds   = [dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',                                          pi=np.pi, element=Vh2s[i].ufl_element()), Vh2s[i])          for i in range(lvl)]
gs   =  [dl.interpolate(dl.Expression('(2.*pi*pi*(0.5+x[0]) + beta)*std::cos(x[0]*pi)*std::cos(x[1]*pi)'+                                          '+pi*std::sin(pi*x[0])*std::cos(pi*x[1])',                                           pi=np.pi, beta=beta, element=Vh2s[i].ufl_element()), Vh2s[i])          for i in range(lvl)]


ntargets = 25
rel_noise = 0.01

#Targets only on the bottom
targets_x = np.random.uniform(0.1,0.9, [ntargets] )
targets_y = np.random.uniform(0.1,0.9, [ntargets] )
targets = np.zeros([ntargets, 2])
targets[:,0] = targets_x
targets[:,1] = targets_y


dl.plot(mesh)
plt.plot(targets_x, targets_y, '*')
plt.show()


misfits = [PointwiseStateObservation(Vh2s[i], targets) for i in range(lvl)]
Bs      = [sps.csr_matrix(csr_fenics2scipy(misfits[i].B), shape=(ntargets, Vh2s[i].dim())) for i in range(lvl)]


problems = [inverseDiffusion(Vhs[i], Vh1s[i], Vh2s[i], beta, gamma1, gamma2, uds[i], gs[i], rhols[i], B = Bs[i])            for i in range(lvl)]

 

rhol = rhols[-1]
ud   = uds[-1]
g    = gs[-1]
problem = problems[-1]

rhotrue = dl.interpolate(dl.Expression('0.5+x[0]',                                      element=Vh1.ufl_element()), Vh1)


# In[9]:


# ---- gradient check   
    
F     = lambda x : problem.c(x)
gradF = lambda x : problem.Dxc(x)

# initial point
x0 = np.array([1.+np.random.randn() if i < Vh2.dim()       else (abs(np.random.randn())+1. + rhol[i-Vh2.dim()]) for i in range(Vh.dim())])
xhat = np.random.randn(Vh.dim()) 


grad_check(F, gradF, x0, xhat)


# In[10]:


linsolve_strategy="reduced"
precond_strategy="diagHmm"
solver = interior_pt(problems, linsolve_strategy, precond_strategy)
x0      = np.ones(problem.n)
x0[:problem.n1] = problem.restore_feasibility(x0)
X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
solver.initialize(X0)
mu0    = 1.e0
opttol = 1.e-8
max_it = 25
Xf, mu, E, Mus = solver.solve(opttol, max_it, mu0)
xf, lamf, zf = Xf[:]


# In[103]:


GMRESiterations = [len(solver.residuals[i]) for i in range(len(solver.residuals)) ]
print("total GMRES iterations = {0:d}".format(sum(GMRESiterations)))

