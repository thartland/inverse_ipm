import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
# False - natural ordering; True - interlace
dl.parameters['reorder_dofs_serial'] = False
dl.parameters['dof_ordering_library'] = 'Boost'
from problems import inverseDiffusion
from hippylib import PointwiseStateObservation

import mpi4py.MPI as MPI
import os


sparse_obs = True

# ---- discretized PDE domain
nx = 100
mesh = dl.UnitSquareMesh(nx, nx)



# ---- MPI process info
comm = mesh.mpi_comm()
rank = comm.rank
nProcs = comm.size
iAmRoot = rank == 0

# ---- directory to save data to
rootDir = os.path.abspath(os.curdir)
relSaveDir = './Figures/'
saveDir = os.path.join(rootDir, relSaveDir)
if iAmRoot:
    os.makedirs(saveDir, exist_ok=True)


# ---- Finite element spaces over discretized domain
PrhoDeg = 1
PuDeg   = 1
Prho    = dl.FiniteElement("CG", mesh.ufl_cell(), PrhoDeg)
Pu      = dl.FiniteElement("CG", mesh.ufl_cell(), PuDeg)
Th      = dl.MixedElement([Pu, Prho])
Vh      = dl.FunctionSpace(mesh, Th)
Vhu     = dl.FunctionSpace(mesh, Pu)
Vhrho   = dl.FunctionSpace(mesh, Prho)

beta   = 1.
gamma1 = 1.e-1
gamma2 = 1.e-2
Crhol  = 0.75


rhol = dl.interpolate(dl.Expression('C', element=Vhrho.ufl_element(), C=Crhol), Vhrho).vector()
ud   = dl.interpolate(dl.Expression('std::cos(x[0]*pi)*std::cos(x[1]*pi)',\
                                          pi=np.pi, element=Vhu.ufl_element()), Vhu).vector()
g    = dl.interpolate(dl.Expression('(2.*pi*pi*(0.5+x[0]) + beta)*std::cos(x[0]*pi)*std::cos(x[1]*pi)'+\
                                          '+pi*std::sin(pi*x[0])*std::cos(pi*x[1])',\
                                           pi=np.pi, beta=beta, element=Vhu.ufl_element()), Vhu)
X    = dl.interpolate(dl.Expression(('x[0]', 'x[1]+1.'), element=Vh.ufl_element()), Vh).vector()

ntargets = 200
rel_noise = 0.01

#Targets only on the bottom
np.random.seed(1)
xmin = 0.05
xmax = 0.95
ymin = 0.05
ymax = 0.95#0.5
targets_x = np.random.uniform(xmin, xmax, [ntargets] )
targets_y = np.random.uniform(ymin, ymax, [ntargets] )
targets = np.zeros([ntargets, 2])
targets[:,0] = targets_x
targets[:,1] = targets_y


misfit = PointwiseStateObservation(Vhu, targets)
if sparse_obs:
    problem = inverseDiffusion(Vh, Vhrho, Vhu, beta, gamma1, gamma2, ud, g, rhol, B=misfit.B)
else:
    problem = inverseDiffusion(Vh, Vhrho, Vhu, beta, gamma1, gamma2, ud, g, rhol)
rhotrue = dl.interpolate(dl.Expression('0.5+x[0]',\
                                      element=Vhrho.ufl_element()), Vhrho)

X.apply("insert")

dl.File(saveDir+"g.pvd") << g

fX = problem.f(X)


if iAmRoot:
    if sparse_obs:
        print(2.* fX / ntargets)
    else:
        print(2. * fX)

g = problem.Dxf(X)


X0 = X.copy()
X1 = X.copy()
Y  = X.copy() # direction for differentiation!
Y.zero()
Y.set_local(np.random.randn(Y.local_size()))
gY = g.inner(Y)

epss = np.logspace(1, 40, base=0.5, num=50)
errs = np.zeros(len(epss))
for i in range(len(epss)):
    eps = epss[i]
    X1.zero()
    X1.axpy(1.0, X)
    X1.axpy(eps, Y) # X1 = X + eps Y
    X1.apply("insert")
    fX1 = problem.f(X1)
    errs[i] = abs((fX1 - fX)/eps - gY)


if iAmRoot:
    plt.plot(epss, errs)
    plt.plot(epss, epss*errs[0]/epss[0]*0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()









