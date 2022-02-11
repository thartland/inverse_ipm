import numpy as np
import matplotlib.pyplot as plt
from filterLineSearch import interior_pt 

from problems import hs071
from helperfunctions import grad_check

if __name__ == "__main__":
    problem = hs071()
    

    # ---- gradient check   
    F     = lambda x : problem.phi(x, 1.)
    gradF = lambda x : problem.Dxphi(x, 1.)

    # initial point
    x0   = 2. + np.abs(np.random.randn(problem.n))
    xhat = np.random.randn(problem.n)
    grad_check(F, gradF, x0, xhat)


    solver  = interior_pt(problem, linsolve_strategy="direct")
    
    x0      = np.ones(problem.n)
    x0[0]   = 1.1
    x0[1]   = 4.9
    x0[2]   = 4.9
    x0[3]   = 1.1
    X0      = [x0, np.ones(problem.m), np.ones(problem.n2)]
    solver.initialize(X0)
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

