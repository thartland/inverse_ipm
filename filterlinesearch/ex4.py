import numpy as np
import matplotlib.pyplot as plt
from filterLineSearch import interior_pt 

from problems import example4
from helperfunctions import grad_check





if __name__ == "__main__":
    nx = 4
    problem = example4(nx)
    

    #---- gradient check
    x0    = 1. + np.abs(np.random.randn(problem.n))
    F     = lambda x : problem.f(x)
    gradF = lambda x : problem.Dxf(x)
    xhat = np.random.randn(problem.n)
    grad_check(F, gradF, x0, xhat)
    
    solver  = interior_pt(problem, "direct")
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
    Es = [np.array([E[i][j] for i in range(len(E))]) for j in range(len(E[0]))]
    titles = ["optimality error", "stationarity", "feasibility", "complementarity"]
    for j in range(4):
        plt.plot(Es[j], "*")
        plt.title(titles[j] + " history")
        plt.yscale('log')
        plt.show()
