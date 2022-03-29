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


# In[21]:


nx_coarse = 5

# coarse and fine meshes
mesh_coarse = dl.UnitSquareMesh(nx_coarse, nx_coarse)
meshes = [mesh_coarse]
lvl = 4
for i in range(lvl-1):
    meshes.append(dl.refine(meshes[-1]))
#mesh_fine   = dl.UnitSquareMesh(nx, nx)
#meshes = [mesh_coarse, mesh_fine]
mesh   = meshes[-1]
lvl    = len(meshes) # depth of mesh hierarchy
dl.plot(mesh)
plt.title('Mesh of PDE domain')
plt.show()


# In[22]:


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


# In[23]:


beta   = 1.
gamma1 = 1.e-1
gamma2 = 1.e-2
Crhol  = 0.75


rhols = [dl.interpolate(dl.Expression('C', element=Vh1s[i].ufl_element(), C=Crhol), Vh1s[i]).vector().get_local()          for i in range(lvl)]
uds   = [dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',                                          pi=np.pi, element=Vh2s[i].ufl_element()), Vh2s[i])          for i in range(lvl)]
gs   =  [dl.interpolate(dl.Expression('(2.*pi*pi*(0.5+x[0]) + beta)*std::cos(x[0]*pi)*std::cos(x[1]*pi)'+                                          '+pi*std::sin(pi*x[0])*std::cos(pi*x[1])',                                           pi=np.pi, beta=beta, element=Vh2s[i].ufl_element()), Vh2s[i])          for i in range(lvl)]


ntargets = 25
rel_noise = 0.01

#Targets only on the bottom
targets_x = np.random.uniform(0.05,0.95, [ntargets] )
targets_y = np.random.uniform(0.1,0.5, [ntargets] )
targets = np.zeros([ntargets, 2])
targets[:,0] = targets_x
targets[:,1] = targets_y


dl.plot(mesh)
plt.plot(targets_x, targets_y, '*')
plt.show()

for i in range(lvl):
    uds[i].vector()[:] += rel_noise * np.random.randn(Vh2s[i].dim()) / np.linalg.norm(uds[i].vector()[:], np.inf)


misfits = [PointwiseStateObservation(Vh2s[i], targets) for i in range(lvl)]
Bs      = [sps.csr_matrix(csr_fenics2scipy(misfits[i].B), shape=(ntargets, Vh2s[i].dim())) for i in range(lvl)]


problems = [inverseDiffusion(Vhs[i], Vh1s[i], Vh2s[i], beta, gamma1, gamma2, uds[i], gs[i], rhols[i], B = Bs[i]) for i in range(lvl)]

 

rhol = rhols[-1]
ud   = uds[-1]
g    = gs[-1]
problem = problems[-1]

rhotrue = dl.interpolate(dl.Expression('0.5+x[0]', element=Vh1.ufl_element()), Vh1)


# In[24]:


# ---- gradient check   
    
F     = lambda x : problem.c(x)
gradF = lambda x : problem.Dxc(x)

# initial point
x0 = np.array([1.+np.random.randn() if i < Vh2.dim()  else (abs(np.random.randn())+1. + rhol[i-Vh2.dim()]) for i in range(Vh.dim())])
xhat = np.random.randn(Vh.dim()) 


linsolve_strategy ="fullmultigrid"
precond_strategy = "diagHmm"
solver = interior_pt(problems, linsolve_strategy, precond_strategy)
x0      = np.ones(problem.n)
x0[:problem.n1] = problem.restore_feasibility(x0)
X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
solver.initialize(X0)
mu0    = 1.e0
opttol = 1.e-8
max_it = 40
Xf, mu, E, Mus = solver.solve(opttol, max_it, mu0)
xf, lamf, zf = Xf[:]


# In[26]:


GMRESiterations = [len(solver.residuals[i]) for i in range(len(solver.residuals)) ]
print("total GMRES iterations = {0:d}".format(sum(GMRESiterations)))
print("parameter/state dim = {0:d}".format(Vh2.dim()))


# In[18]:


uReconstruction   = dl.Function(Vh2)
lamReconstruction = dl.Function(Vh2)

rhoReconstruction = dl.Function(Vh1)
zReconstruction   = dl.Function(Vh1)

uReconstruction.vector()[:]   = xf[:problem.n1]
rhoReconstruction.vector()[:] = xf[problem.n1:]

lamReconstruction.vector()[:] = lamf[:]
zReconstruction.vector()[:]   = zf[:]

nb.multi1_plot([uReconstruction, ud], ["state reconstruction", "observations"])
plt.show()


# In[19]:


nb.multi1_plot([rhoReconstruction, rhotrue], ["parameter reconstruction", "true parameter"])
plt.show()


# In[20]:


nb.multi1_plot([lamReconstruction, zReconstruction],             ["equality constraint multiplier (adjoint)", "bound constraint multiplier"],            same_colorbar=False)
plt.show()

print("{0:1.2e} <= z <= {1:1.2e}".format(min(zf), max(zf)))
print("portion of bound constraint dofs below tol = {0:1.2f}".format(sum(zf < opttol)/len(zf)))


# In[21]:


Es = [[E[i][j] for i in range(len(E))] for j in range(len(E[0]))]
labels = ["optimality error", "stationarity", "feasibility", "complementarity"]

for i in range(1,4):
    plt.plot(Es[i], label=labels[i], linewidth=2.0)
    plt.yscale('log')
plt.legend(loc = 'lower center')
plt.grid()
plt.title('convergence history')
plt.show()


# In[22]:


Prhos = [csr_fenics2scipy(         dl.PETScDMCollection.create_transfer_matrix(         Vh1s[i], Vh1s[i+1])) for i in range(lvl-1)]
Rrhos = [Prhos[i].transpose() for i in range(lvl-1)]
Pstates = [csr_fenics2scipy(          dl.PETScDMCollection.create_transfer_matrix(           Vh2s[i], Vh2s[i+1])) for i in range(lvl-1)]
Rstates = [Pstates[i].transpose() for i in range(lvl-1)]

Ps = [sps.bmat([[Pstates[i], None, None],
                [None, Prhos[i], None],
                [None, None, Pstates[i]]], format="csr") for i in range(lvl-1)]
Rs = [sps.bmat([[Rstates[i], None, None],
                [None, Rrhos[i], None],
                [None, None, Rstates[i]]], format="csr") for i in range(lvl-1)]


# In[23]:


Ak = solver.formH(Xf, mu)[0]
Wk = Ak[:Vh.dim(), :Vh.dim()]
Jk = Ak[Vh.dim():, :Vh.dim()]
JkT = Jk.transpose()



# In[24]:


As = [None for i in range(lvl)]
Ss = [None for i in range(lvl)]
bs   = [None for i in range(lvl)]

As[-1] = Ak
bs[-1] = np.ones(Ak.shape[0])
for i in range(lvl-1)[::-1]:
    As[i] = Rs[i].dot(As[i+1]).dot(Ps[i])
    bs[i] = Rs[i].dot(bs[i+1])
for i in range(lvl):
    Wki   = As[i][:Vhs[i].dim(), :Vhs[i].dim()]
    Jki   = As[i][Vhs[i].dim():, :Vhs[i].dim()]
    JkiT  = Jki.transpose()
    Ss[i] = SchurComplementSmoother(Wki, JkiT, Jki, Vh2s[i].dim())


# In[25]:


from pyamg.krylov import gmres as gmres

smoothing_steps = 1 
lintol  = 1.e-12
maxiter = 50

residuals = [None for i in range(lvl-1)]
for sys in range(2, lvl+1):
    print("levels = {0:d}".format(sys))
    print("-"*50)
    multi_grid_P = multi_grid_action(As[:sys], Ss[:sys], Ps[:sys-1], Rs[:sys-1], smoothing_steps)
    krylov_convergence = Krylov_convergence(As[sys-1], bs[sys-1], residual_callback=False)
    res = list()
    u, info = gmres(As[sys-1], bs[sys-1], tol=lintol,                    M = multi_grid_P, maxiter=maxiter, residuals=res, callback=krylov_convergence.callback)
    print(info)
    print("final residual = {0:1.2e} (multi-grid) \n".format(np.linalg.norm(As[sys-1].dot(u)-bs[sys-1])))
    residuals[sys-2] = res[:] / np.linalg.norm(multi_grid_P._matvec(bs[sys-1]))
    print("number of iterations = {0:d}".format(len(residuals[sys-2])))
    krylov_convergence.reset()
krylov_convergence.reset()
res = list()
u, info = gmres(As[sys-1], bs[sys-1], tol=lintol,                    maxiter=maxiter, residuals=res, callback=krylov_convergence.callback)
residuals_noP = res[:] / np.linalg.norm(bs[sys-1])


# In[26]:


for i in range(lvl-1):
    plt.plot(residuals[i], '-^', label='{0:d} MG levels'.format(i+2), linewidth=2.0)
#plt.plot(residuals_noP, label='no P')    
plt.yscale('log')
plt.ylabel(r'$||M r_{k}||/||M r_{0}||$, relative preconditioned residual norm')
plt.xlabel(r'Krylov subspace iteration')
plt.title('Linear system convergence history')
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.grid()
plt.savefig('IPNewton_SchurComplementSmoothing_fullMultigrid.png')
plt.show()


# In[ ]:




