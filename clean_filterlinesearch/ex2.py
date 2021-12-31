import numpy as np
import matplotlib.pyplot as plt
from filterLineSearch import interior_pt 

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
        y[:self.nx, self.nx:] = A
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


if __name__ == "__main__":
    nx = 10
    A  = np.zeros((nx, nx))
    for i in range(nx):
        if i < int(nx/2):
            A[i,i] = -0.25
        else:
            A[i,i] = 1.
    problem = example2(nx, A)
    

    #---- gradient checks

    epss  = np.logspace(1, 40, base=0.5)
    x0    = 2. + np.abs(np.random.randn(problem.n))
    F     = lambda x : problem.phi(x, 0.1)
    gradF = lambda x : problem.Dxphi(x, 0.1)
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
    
    x0      = 1.0 * np.ones(problem.n)
    X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
    solver.initialize(X0)
    complementarity0 = np.linalg.norm(X0[2][:]*(x0[problem.n1:] - problem.rhol[:]), np.inf)
    feasibility0     = problem.theta(x0)
    optimality0      = np.linalg.norm(problem.DxL(X0), 2)


    mu0 = 1.e0
    tol = 1.e-10
    max_it = 35
    Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
    xf = Xf[0][:]
    complementarityf = np.linalg.norm(Xf[2][:]*(xf[problem.n1:] - problem.rhol[:]), np.inf)
    feasibilityf     = problem.theta(xf)
    optimalityf      = np.linalg.norm(problem.DxL(Xf), 2)
    print("complementarity error reduction = {0:1.3e}".format(complementarityf / complementarity0))
    print("feasibility error reduction = {0:1.3e}".format(feasibilityf / feasibility0))
    print("optimality error reduction = {0:1.3e}".format( optimalityf / optimality0))
    print("||lam||_1 = ", np.linalg.norm(Xf[1], 1))
    print("||z||_1   = ", np.linalg.norm(Xf[2], 1))

    xexpected = np.ones(nx)
    xexpected[1]  = 1. / (A[1,1] / A[0,0] + nx - 1.)
    xexpected[2:] = xexpected[1]
    xexpected[0]  = (A[1,1] / A[0,0]) * xexpected[1]
    lamexpected   = -A[1,1] * xexpected[1]
    print("computed optimal solution = ", Xf[0][:nx])
    print("expected optimal solution = ", xexpected)
    print("computed equality constraint multiplier = ", Xf[1][-1])
    print("expected equality constraint multiplier = ", lamexpected)
    print("computed bound-constraint multipliers   = ", Xf[2])
    Es = [np.array([E[i][j] for i in range(len(E))]) for j in range(len(E[0]))]
    titles = ["optimality", "feasibility", "complementarity"]
    for j in range(1,4):
        plt.plot(Es[j], "*")
        plt.title(titles[j-1] + " history")
        plt.yscale('log')
        plt.show()
