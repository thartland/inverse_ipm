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

import sys

argumentList = sys.argv[1:]

nx_coarse = int(argumentList[0])
nx = 2*nx_coarse

# coarse and fine meshes
mesh_coarse = dl.UnitSquareMesh(nx_coarse, nx_coarse)

mesh_fine   = dl.UnitSquareMesh(nx, nx)

meshes = [mesh_coarse, mesh_fine]
mesh   = meshes[-1]
lvl    = len(meshes) # depth of mesh hierarchy




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




beta   = 1.
gamma1 = 1.e-4
gamma2 = 1.e-4
Crhol  = 0.75


rhols = [dl.interpolate(dl.Expression('C', element=Vh1s[i].ufl_element(), C=Crhol), Vh1s[i]).vector().get_local() \
         for i in range(lvl)]
uds   = [dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',\
                                          pi=np.pi, element=Vh2s[i].ufl_element()), Vh2s[i]) \
         for i in range(lvl)]
gs   =  [dl.interpolate(dl.Expression('(2.*pi*pi*(0.5+x[0]) + beta)*std::cos(x[0]*pi)*std::cos(x[1]*pi)'+\
                                          '+pi*std::sin(pi*x[0])*std::cos(pi*x[1])',\
                                           pi=np.pi, beta=beta, element=Vh2s[i].ufl_element()), Vh2s[i]) \
         for i in range(lvl)]

problems = [inverseDiffusion(Vhs[i], Vh1s[i], Vh2s[i], beta, gamma1, gamma2, uds[i], gs[i], rhols[i]) \
           for i in range(lvl)]

rhol = rhols[-1]
ud   = uds[-1]
g    = gs[-1]
problem = problems[-1]

rhotrue = dl.interpolate(dl.Expression('0.5+x[0]',\
                                      element=Vh1.ufl_element()), Vh1)


linsolve_strategy=argumentList[1]
solver = interior_pt(problems, linsolve_strategy)
x0      = np.ones(problem.n)
x0[:problem.n1] = problem.restore_feasibility(x0)
X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
solver.initialize(X0)
mu0    = 1.e0
opttol = 1.e-8
max_it = 30
Xf, mu, E, Mus = solver.solve(opttol, max_it, mu0)
xf, lamf, zf = Xf[:]


GMRESiterations = [(i+1, len(solver.residuals[i])) for i in range(len(solver.residuals))]

np.savetxt("data/GMRES"+linsolve_strategy+"nx"+str(nx)+".dat", GMRESiterations)


