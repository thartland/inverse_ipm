import numpy as np
import matplotlib.pyplot as plt
from filterLineSearch import interior_pt 


"""
In this example problem we seek to solve

min_(x1, x2, x3, x4) f(x1, x2, x3, x4) := x1 * x4 * (x1 + x2 + x3) + x_2

s.t.
1 <= xj <=5, j = 1,2,3,4
      c1(x1, x2, x3, x4) = 0
25 <= C2(x1, x2, x3, x4)

c1(x1, x2, x3, x4) := sum_j xj^2 - 40
C2(x1, x2, x3, x4) := x1 * x2 * x3 * x4

slack variables are then introduced
x5 = 5 - x1
x6 = 5 - x2
x7 = 5 - x3
x8 = 5 - x4
x9 = C2(x1, x2, x3, x4) - 25

so that we then seek the solution of

min_(x1, x2, x3, x4, x5, x6, x7, x8, x9) f(x1, x2, x3, x4)

s.t.
1 <= xj, j = 1,2,3,4
0 <= xj, j = 5,6,7,8,9
cj(x) = 0, j=1,2,..,6
c1(x) := sum_(1<=j<=4) xj^2 - 40
c2(x) := x9 + C2(x1, x2, x3, x4) - 25
c3(x) := x1 + x5 - 5
c4(x) := x2 + x6 - 5
c5(x) := x3 + x7 - 5
c6(x) := x4 + x8 - 5
"""

class hs071:
    def __init__(self):
        self.n1   = 0
        self.n2   = 9
        self.n    = self.n1 + self.n2
        self.m    = 6
        self.rhol = np.zeros(self.n2)
        self.rhol[:4] = 1.
    def f(self, x):
        return x[0]*x[3]*sum(x[:3]) + x[2]
    def Dxf(self, x):
        g = np.zeros(self.n)
        g[0] = x[3]*sum(x[:3]) + x[0] * x[3]
        g[1] = x[0]*x[3]
        g[2] = 1. + x[0]*x[3]
        g[3] = x[0]*sum(x[:3])
        return g
    def Dxxf(self, x):
        H    = np.zeros((self.n, self.n))
        H[0, 0] = 2.*x[3]
        H[0, 1:3] = x[3]
        H[0, 3]   = 2.*x[0] + x[1] + x[2]
        H[1:3, 0] = x[3]
        H[3, 0]   = 2.*x[0] + x[1] + x[2]
        H[3, 1:3] = x[0]
        H[1:3, 3] = x[0]
        return H
    def c(self, x):
        y = np.zeros(self.m)
        y[0] = np.inner(x[:4], x[:4]) - 40.
        y[1] = x[8] - np.prod(x[:4]) + 25.
        y[2:] = x[:4] + x[4:8] - 5.
        return y
    def theta(self, x):
        return np.linalg.norm(self.c(x), 2)
    def Dxc(self, x):
        J = np.zeros((self.m, self.n))
        J[0,:4] =  2. * x[:4]
        J[1, 0] = -1. * np.prod(x[1:4])
        J[1, 1] = -1. * x[0]*np.prod(x[2:4])
        J[1, 2] = -1. * np.prod(x[:2])*x[3]
        J[1, 3] = -1. * np.prod(x[:3])
        J[1, 8] =  1.
        for j in range(4):
            J[j + 2, j]     = 1.
            J[j + 2, j + 4] = 1.
        return J
    def Dxxc(self, x):
        y = np.zeros((self.m, self.n, self.n))
        y[0][:4,:4] = 2. * np.identity(4)
        y[1][:4, :4]  = -1.*np.array([[0.       , x[2]*x[3], x[1]*x[3], x[1]*x[2]],\
                                [x[2]*x[3], 0.       , x[0]*x[3], x[0]*x[2]],\
                                [x[1]*x[3], x[0]*x[3], 0.       , x[0]*x[1]],\
                                [x[1]*x[2], x[0]*x[2], x[0]*x[1], 0.]]) 
        return y
    def phi(self, x, mu):
        return self.f(x) - mu * sum(np.log(x[self.n1:] - self.rhol))
    def Dxphi(self, x, mu):
        return self.Dxf(x) - mu / (x[self.n1:] - self.rhol)
    def L(self, X):
        x, lam, z = X[:]
        return (self.f(x) + np.inner(lam, self.c(x)) - np.inner(z, x[self.n1:]-self.rhol))
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
        sd    = max(smax, (laml1 + zl1) / (self.m + self.n)) / smax
        sc    = max(smax, zl1 / self.n) / smax
        if self.m > 0:
            E2 = np.linalg.norm(self.c(x), np.inf)
            return max(E1 / sd, E2, E3 / sc), E1, E2, E3
        else:
            return max(E1 / sd, E3 / sc)
    def restore_feasibility(self, X):
        x, lam, z = X[:]
        A = np.zeros((self.n+self.m, self.n+self.m))
        A[:self.n, :self.n] = np.identity(self.n)
        J = self.Dxc(x)
        A[self.n:self.n+self.m, :self.n] = J[:,:]
        A[:self.n, self.n:self.n+self.m] = (J.T)[:,:]
        r = np.zeros(self.n+self.m)
        r[self.n:self.n+self.m] = self.c(x)[:]
        sol = np.linalg.solve(A, -r)
        xhat = sol[:self.n]
        lamhat = sol[self.n:self.n+self.m]
        return x+xhat, lam+lamhat, z


if __name__ == "__main__":
    problem = hs071()
    

    #---- gradient checks

    epss = np.logspace(1, 40, base=0.5)
    x0   = 2. + np.abs(np.random.randn(problem.n))
    f0   = problem.phi(x0, 1.)
    g0   = problem.Dxphi(x0, 1.)
    xhat = np.random.randn(problem.n)
    grad_err = np.zeros(len(epss))
    for j, eps in enumerate(epss):
        fplus = problem.phi(x0 + eps*xhat, 1.)
        grad_err[j] = abs((fplus - f0)/ eps - np.inner(g0, xhat))#, 2)
    plt.plot(epss, grad_err)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



    solver  = interior_pt(problem)
    
    x0      = np.ones(problem.n)
    x0[0]   = 1.1
    x0[1]   = 4.9
    x0[2]   = 4.9
    x0[3]   = 1.1
    X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
    solver.initialize(X0)
    #solver.kSig = 1.e12
    complementarity0 = np.linalg.norm(X0[2][:]*(x0[problem.n1:] - problem.rhol[:]), np.inf)
    feasibility0     = problem.theta(x0)
    optimality0      = np.linalg.norm(problem.DxL(X0), 2)


    mu0 = 1.
    tol = 1.e-8
    max_it = 100
    Xf, mu, E, Mus = solver.solve(tol, max_it, mu0)
    xf = Xf[0][:]
    complementarityf = np.linalg.norm(Xf[2][:]*(xf[problem.n1:] - problem.rhol[:]), np.inf)
    feasibilityf     = problem.theta(xf)
    optimalityf      = np.linalg.norm(problem.DxL(Xf), 2)
    print("complementarity error reduction = ", complementarityf / complementarity0)
    print("feasibility error reduction = ", feasibilityf / feasibility0)
    print("optimality error reduction = ", optimalityf / optimality0)
    print(Xf[2][:]*(xf[problem.n1] - problem.rhol[:]))
    #print(Mus)
    print("||lam||_1 = ", np.linalg.norm(Xf[1], 1))
    print("||z||_1   = ", np.linalg.norm(Xf[2], 1))
    print("c(x)      = ", problem.c(xf))
    print("DxL(X)    = ", problem.DxL(Xf))


    print("optimal solution = ", Xf[0][:4])
    Es = [np.array([E[i][j] for i in range(len(E))]) for j in range(len(E[0]))]
    titles = ["optimality error", "stationarity", "feasibility", "complementarity"]
    for j in range(4):
        plt.plot(Es[j], "*")
        plt.title(titles[j] + " history")
        plt.yscale('log')
        plt.show()

