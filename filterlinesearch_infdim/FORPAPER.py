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


nx = 10 


# coarse and fine meshes
mesh = dl.UnitSquareMesh(nx, nx)

dl.plot(mesh)
plt.title('Mesh of PDE domain')
plt.show()


# In[4]:


P1_deg = 1 # degree of finite-element polynomials for parameter (rho)
P2_deg = 1 # degree of finite-element polynomials for state

P1 = dl.FiniteElement("CG", mesh.ufl_cell(), P1_deg)
P2 = dl.FiniteElement("CG", mesh.ufl_cell(), P2_deg)
Th = dl.MixedElement([P2, P1])
Vh = dl.FunctionSpace(mesh, Th)
Vh1 = dl.FunctionSpace(mesh, P1)
Vh2 = dl.FunctionSpace(mesh, P2)

print("dim(state) = {0:d}, dim(parameter) = {1:d}".format(Vh.sub(0).dim(), Vh.sub(1).dim()))


# In[5]:


beta   = 1.
gamma1 = 1.e-1#*1.e-5
gamma2 = 1.e-1#*1.e-2
Crhol  = 1.0


rhol = dl.interpolate(dl.Expression('C', element=Vh1.ufl_element(), C=Crhol), Vh1).vector()[:]
         
udF   = dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',                                          pi=np.pi, element=Vh2.ufl_element()), Vh2)
ud = udF.vector()[:]
g    = dl.interpolate(dl.Expression('(2.*pi*pi*(0.5+x[0]+x[1]) + beta)*std::cos(x[0]*pi)*std::cos(x[1]*pi)'+                                          '+pi*std::sin(pi*x[0])*std::cos(pi*x[1])+pi*std::cos(pi*x[0])*std::sin(pi*x[1])',                                           pi=np.pi, beta=beta, element=Vh2.ufl_element()), Vh2)


dl.plot(g)
plt.show()
#ntargets = 50
rel_noise = 0.01

#Targets only on the bottom
ntargetsroot = 5
ntargets = ntargetsroot**2
Dx = 1. / (ntargetsroot-1.)
targets = np.zeros([ntargets, 2])
for i in range(ntargetsroot):
    for j in range(ntargetsroot):
        targets[i+j*ntargetsroot, 0] = i*Dx
        targets[i+j*ntargetsroot, 1] = j*Dx
dl.plot(mesh)
plt.plot(targets[:,0], targets[:,1], '*')
plt.show()
    
print( "Number of observation points: {0}".format(ntargets) )
misfit = PointwiseStateObservation(Vh2, targets)
B      = sps.csr_matrix(csr_fenics2scipy(misfit.B), shape=(ntargets, Vh2.dim()))
d      = B.dot(ud)
eta    = rel_noise * np.random.randn(len(d)) * np.linalg.norm(d)
d[:]   += eta[:]
problem = inverseDiffusion(Vh, Vh1, Vh2, beta, gamma1, gamma2, d, g, rhol, B=B)

rhotrue = dl.interpolate(dl.Expression('0.5+x[0]+x[1]',                                      element=Vh1.ufl_element()), Vh1)   


# In[6]:


# ---- gradient check

F     = lambda x : problem.phi(x, 0.1)
gradF = lambda x : problem.Dxphi(x, 0.1)

# initial point
x0   = dl.interpolate(dl.Expression(('std::cos(x[0]*pi)*std::cos(x[1]*pi)', '2.0+x[0]*x[1]'),                                          pi=np.pi, element=Vh.ufl_element()), Vh).vector()[:]

xhat   = dl.interpolate(dl.Expression(('std::sin(x[0]*pi)*std::cos(x[1]*pi)', '2.0+x[0]*x[1]'),                                          pi=np.pi, element=Vh.ufl_element()), Vh).vector()[:]

grad_check(F, gradF, x0, xhat, save=False)


# In[7]:


solver = interior_pt(problem, linsolve_strategy="reduced")
x0      = 10.*np.ones(problem.n)
x0[:problem.n1] = problem.restore_feasibility(x0)
X0      = [x0, 10.*np.ones(problem.m), 10.*np.ones(problem.n2)]
solver.initialize(X0)
mu0 = 1.e0
tol = 1.e-6
max_it = 50
Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
xf, lamf, zf = Xf[:]
GMRESresiduals = [len(solver.residuals[i]) for i in range(len(solver.residuals))]
print("{0:d} total Krylov iterations".format(sum(GMRESresiduals)))
print("{0:1.1f} average Krylov iterations per IP-Newton system solve".format(sum(GMRESresiduals)/len(GMRESresiduals)))


# In[8]:


uReconstruction   = dl.Function(Vh2)
lamReconstruction = dl.Function(Vh2)

rhoReconstruction = dl.Function(Vh1)
zReconstruction   = dl.Function(Vh1)


uReconstruction.vector()[:]   = xf[:problem.n1]
rhoReconstruction.vector()[:] = xf[problem.n1:]

lamReconstruction.vector()[:] = lamf[:]
zReconstruction.vector()[:]   = zf[:]

nb.multi1_plot([uReconstruction, udF], ["state reconstruction", "observations"])
plt.show()


# In[9]:


nb.multi1_plot([rhoReconstruction, rhotrue], ["parameter reconstruction", "true parameter"])
plt.show()


# In[10]:


nb.multi1_plot([lamReconstruction, zReconstruction],             ["equality constraint multiplier (adjoint)", "bound constraint multiplier"],            same_colorbar=False)
plt.show()

print("{0:1.2e} <= z <= {1:1.2e}".format(min(zf), max(zf)))
print("portion of bound constraint dofs below tol = {0:1.2f}".format(sum(zf < tol)/len(zf)))


# In[11]:


Es = [[E[i][j] for i in range(len(E))] for j in range(len(E[0]))]
labels = ["optimality error", "stationarity", "feasibility", "complementarity"]

for i in range(1,4):
    plt.plot(Es[i], label=labels[i], linewidth=2.0)
    plt.yscale('log')
plt.legend(loc='lower center')
plt.grid()
plt.title('convergence history')
plt.show()


# In[12]:


Ak, Jk, drk = solver.formH(Xf, mu)
JkT = Jk.transpose()
Wk  = Ak[:problem.n1+problem.n2, :problem.n1+problem.n2]
GS = GaussSeidel(Wk, JkT, Jk,problem.n1)


# In[13]:


ei = np.zeros(Ak.shape[0])
A  = np.zeros(Ak.shape)
PA = np.zeros(Ak.shape)
for i in range(Ak.shape[0]):
    ei[i] = 1.0
    A[:,i] = Ak.dot(ei)
    PA[:,i] = GS.dot(A[:,i])
    ei[i] = 0.0


# In[14]:


condA = np.linalg.cond(A)
condPA = np.linalg.cond(PA)


# In[15]:


print("cond(A) = {0:1.3e}".format(condA))
print("cond(PA) = {0:1.3e}".format(condPA))


# In[ ]:




