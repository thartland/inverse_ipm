import numpy as np
import matplotlib.pyplot as plt
from filterLineSearch import interior_pt 


"""
Here we seek to solve

min_(x in R^n) f(x) := 0.5 x^T x
s.t.
0<(xl)j <= xj, j=1,...,n
c1(x) = x1 - C, C > (xl)1
"""



class example4:
    def __init__(self, nx):
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



if __name__ == "__main__":
    nx = 4
    problem = example4(nx)
    

    #---- gradient checks

    epss  = np.logspace(1, 40, base=0.5)
    x0    = 1. + np.abs(np.random.randn(problem.n))
    F     = lambda x : problem.f(x)
    gradF = lambda x : problem.Dxf(x)
    F0   = F(x0)
    gradF0   = gradF(x0)
    xhat = np.random.randn(problem.n)
    grad_err = np.zeros(len(epss))
    for j, eps in enumerate(epss):
        Fplus = F(x0 + eps*xhat)
        if len(gradF0.shape) > 1:
            grad_err[j] = np.linalg.norm((Fplus - F0)/ eps - np.dot(gradF0, xhat), 2)
        else:
            grad_err[j] = np.abs((Fplus - F0) / eps - np.inner(gradF0, xhat))
    plt.plot(epss, grad_err, 'k')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.title('finite difference check')
    plt.show()


    solver  = interior_pt(problem)
    #solver.kSig = 1.2 
    x0      = np.array([i+1.0 for i in range(problem.n)])#1.0 * np.ones(problem.n)
    X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
    solver.initialize(X0)


    mu0 = 1.e0
    tol = 1.e-8
    max_it = 15
    Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
    xf = Xf[0][:]

    print("computed optimal solution = ", Xf[0])
    print("computed equality-constraint multipliers = ", Xf[1])
    print("computed bound-constraint multipliers = ", Xf[2])
    print("filter = ", solver.F)
    Es = [np.array([E[i][j] for i in range(len(E))]) for j in range(len(E[0]))]
    titles = ["optimality error", "stationarity", "feasibility", "complementarity"]
    for j in range(4):
        plt.plot(Es[j], "*")
        plt.title(titles[j] + " history")
        plt.yscale('log')
        plt.show()
