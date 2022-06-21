#!/usr/bin/env python
# coding: utf-8

# Here we seek to solve
# \begin{align}
# \min_{(u,\rho)\in\mathcal{V}\times\mathcal{M}}J(u,\rho)&:=\frac{1}{2}\int_{\Omega}(u-u_{d})^{2}\mathrm{d}V+\frac{1}{2}R(\rho)\\
# R(\rho)&:=\int_{\Omega}(\gamma_{1}\,\rho^{2}+\gamma_{2}\nabla \rho\cdot\nabla \rho)\mathrm{d}V
# \end{align}
# 
# subject to the partial differential equality constraint
# 
# \begin{align*}
# -\nabla\cdot\left(\rho\,\nabla u\right)+\beta\, u&=f,\,\,\,\text{ in }\Omega \\
# \frac{\partial u}{\partial n}&=0,\,\,\,\text{ on }\partial\Omega
# \end{align*}
# 
# and bound constraint
# 
# \begin{align*}
# \rho(x)\geq \rho_{\ell}(x)>0,\,\,\,\text{ on }\overline{\Omega}
# \end{align*}
# 
# here $\beta\in\mathbb{R}$, $f:\Omega\rightarrow\mathbb{R}$, $u_{d}:\Omega\rightarrow\mathbb{R}$, $\rho_{\ell}:\overline{\Omega}\rightarrow\mathbb{R}_{>0}$, $\lbrace \gamma_{j}\rbrace_{j=1}^{2}\subset\mathbb{R}_{\geq 0}$ are given.
# 
# 

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


# In[3]:


nx = 25 


# coarse and fine meshes
mesh = dl.UnitSquareMesh(nx, nx)

dl.plot(mesh)
plt.title('Mesh of PDE domain')
plt.show()


# In[4]:


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


# In[5]:


beta   = 1.
sparse_obs = True
if P1_deg == 0:
    if sparse_obs:
        gamma1 = 1.
    else:
        gamma1 = 1.e-5
    gamma2 = 0.
else:
    if sparse_obs:
        gamma1 = 1.e-1#*1.e-5
        gamma2 = 1.e-2#*1.e-2
    else:
        gamma1 = 1.e-6
        gamma2 = 1.e-4
Crhol  = 0.75


rhol = dl.interpolate(dl.Expression('C', element=Vh1.ufl_element(), C=Crhol), Vh1).vector()[:]
         
udF   = dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',                                          pi=np.pi, element=Vh2.ufl_element()), Vh2)
ud = udF.vector()[:]
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

    print( "Number of observation points: {0}".format(ntargets) )
    misfit = PointwiseStateObservation(Vh2, targets)
    B      = sps.csr_matrix(csr_fenics2scipy(misfit.B), shape=(ntargets, Vh2.dim()))
    d      = B.dot(ud)
    eta    = rel_noise * np.random.randn(len(d)) * np.linalg.norm(d)
    d[:]   += eta[:]
    problem = inverseDiffusion(Vh, Vh1, Vh2, beta, gamma1, gamma2, d, g, rhol, B=B)
else:
    print("Observations everywhere!")
    problem = inverseDiffusion(Vh, Vh1, Vh2, beta, gamma1, gamma2, ud, g, rhol)



rhotrue = dl.interpolate(dl.Expression('0.5+x[0]',                                      element=Vh1.ufl_element()), Vh1)   


# In[7]:


solver  = interior_pt(problem, linsolve_strategy="GS")
x0      = np.ones(problem.n)
x0[:problem.n1] = problem.restore_feasibility(x0)
X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
solver.initialize(X0)
mu0 = 1.e0
tol = 1.e-8
max_it = 30
Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
xf, lamf, zf = Xf[:]
GMRESresiduals = [len(solver.residuals[i]) for i in range(len(solver.residuals))]
print("{0:d} total Krylov iterations".format(sum(GMRESresiduals)))
print("{0:1.1f} average Krylov iterations per IP-Newton system solve".format(sum(GMRESresiduals)/len(GMRESresiduals)))
