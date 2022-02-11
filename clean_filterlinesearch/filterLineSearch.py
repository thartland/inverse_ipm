import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from helperfunctions import *
import sys, inspect
import problems as problemDefs


class interior_pt:
    def __init__(self, problems, linsolve_strategy):
        self.problems = problems

        problemDefMembers = inspect.getmembers(problemDefs, inspect.isclass)
        problemDefTypes   = [problemDefMembers[i][1] for i in range(len(problemDefMembers))]
        
        if type(self.problems) in problemDefTypes:
            # a single problem has been supplied
            self.problem = self.problems
            problem = self.problem
        elif type(self.problems) is list:
            self.problem  = self.problems[-1]
            problem = self.problem
            if type(problem) not in problemDefTypes:
                raise TypeError("problems argument must be a problem class or list of problem classes from problems.py")

        self.sparse_struct   = self.problem.sparse_struct
        self.linsolve_strategy = linsolve_strategy

        if linsolve_strategy == "multigrid":
            if type(self.problems) is not list:
                raise RuntimeError("a list of problems must be supplied in order to utilize a multigrid strategy")
            self.multigridHierarchy = multigridHierarchy(self.problems)
            self.residuals          = []

        # -------- lower-bound constraint
        self.rhol = problem.rhol
        
        # -------- dimensions of variables
        self.n1    = problem.n1        # dimension of variable that does not have a lower-bound constraint
        self.n2    = problem.n2        # dimension of variable which does have have a lower-bound constraint
        self.n     = self.n1 + self.n2 # dimension of the primal variable
        self.m     = problem.m         # dimension of the dual equality constraint Lagrange multiplier

        self.x    = np.zeros(self.n)     # 0, primal
        self.lam  = np.zeros(self.m)     # 1, dual (c(x) = 0)
        self.z    = np.zeros(self.n2)    # 6, (rho >= rhol)
        
        # -------- construct index set 
        self.idx0 = 0
        self.idx1 = self.idx0 + self.n
        self.idx2 = self.idx1 + self.m
        self.idx3 = self.idx2 + self.n2
        
        self.it   = 0
        self.X    = [np.zeros(self.n), np.zeros(self.m), np.zeros(self.n2)]
        self.Xhat = [np.zeros(self.n), np.zeros(self.m), np.zeros(self.n2)]
        

        # -------- parameters for interior point -------

        # ensure that large Lagrange multipliers
        # do not corrupt E
        self.smax = 1.e2
        
        # control deviation from primal dual Hessian
        self.kSig = 1.0e10

        # control the minimum size of the fraction to boundary parameter
        self.tau_min = 0.8

        # backtracking constant
        self.eta = 1.e-4

        # control the allowable violation of the equality constraints
        self.theta_min = 1.e-4

        # constants in line-step A-5.4
        self.delta  = 1.
        self.stheta = 1.1
        self.sphi   = 2.3

        # control the rate at which the penalty parameter is decreased
        self.kmu     = 0.5
        self.thetamu = 1.5

        # the Filter
        self.F = []

        # maximum constraint violation
        self.theta_max = 1.e6

        # data for second order correction
        self.ck_soc = 0.
        self.k_soc  = 0.99

        # equation (18)
        self.gtheta = 1.e-5
        self.gphi   = 1.e-5

        #
        self.keps = 10.

        # track the inertia-corrections
        self.deltalast = 0.

        # curvature-constraint
        self.alphad = 1.e-11

        # maximum inertia correction attempts
        self.max_ic = 20

        # inertia-correction algorithm parameters (taken from Wachter-Biegler, Algorithm IC)
        self.kappaminus   = 1. / 3.
        self.delta0       = 1.e-4
        self.kappahatplus = 100.
        self.kappaplus    = 8.
        self.deltamin     = 1.e-20 # not as in Wachter-Biegler

        
    def initialize(self, X):
        self.X = X
    
    def split(self, X):
        return [X[i].copy() for i in range(len(X))]


    """
    Setup the IP-Newton system as needed in equations (11) and (26) of Wachter Biegler
    """
    def formH(self, X, mu):
        Hk = self.problem.DxxL(X)
        x, lam, z = self.split(X)
        u   = x[:self.n1]
        rho = x[self.n1:]

        Jk  = self.problem.Dxc(x)
        JkT = Jk.transpose()
       
        if self.sparse_struct:
            dHrhorho = sps.diags(z / (rho - self.rhol))
            Ak = sps.bmat([[Hk[:self.n1, :self.n1], Hk[:self.n1, self.n1:], JkT[:self.n1,:]],\
                           [Hk[self.n1:, :self.n1], Hk[self.n1:, self.n1:] + dHrhorho, JkT[self.n1:, :]],\
                           [Jk[:, :self.n1], Jk[:, self.n1:], None]], format="csr")
        else:
            Ak = np.zeros((self.idx2, self.idx2))

            dHrhorho = np.diag(z / (rho - self.rhol))

            Ak[        0:self.idx1,         0:self.idx1] = Hk[:, :]     # 1,1 block
            Ak[   self.n1:self.idx1,   self.n1:self.idx1]+= dHrhorho
            Ak[self.idx1:self.idx2,         0:self.idx1] = Jk[:, :]     # 2,1 block
            Ak[        0:self.idx1, self.idx1:self.idx2] = (Jk.T)[:, :] # 1,2 block

        """
        By eliminating the bound-constraint multiplier, the rhs of the IP-Newton system is altered by dr
        """
        drk    = -mu / (rho - self.rhol)
        return Ak, Jk, drk

    def linsolve(self, A, b):
        if self.sparse_struct:
            W  = A[:self.n, :self.n]
            JT = A[:self.n, self.n:]
            J  = A[self.n:, :self.n]
            if self.linsolve_strategy == "multigrid":
                self.multigridHierarchy.constructPreconditioner(W, JT, J, self.problem.n1)
                M = two_grid_action(self.multigridHierarchy.Lfine, self.multigridHierarchy.Lcoarse,\
                                       self.multigridHierarchy.Spre, self.multigridHierarchy.P,\
                                       self.multigridHierarchy.R, 2, Spost=self.multigridHierarchy.Spost)
            elif self.linsolve_strategy == "presmoothing":
                M = ConstrainedPreSmoother(W, JT, J, self.problem.n1)
            if self.linsolve_strategy == "direct":
                sol = spla.spsolve(A, b)
            else:
                lintol= 1.e-8
                maxiter = 100
                krylov_convergence = Krylov_convergence(A, b)
                sol, info = spla.gmres(A, b, tol=lintol, atol=lintol, \
                                       M = M, maxiter=maxiter, callback=krylov_convergence.callback)
                self.residuals.append(krylov_convergence.residuals)
        else:
            sol = np.linalg.solve(A, b)
        return sol 
    """ 
    determine the solution
    of the perturbed KKT conditions
    no inertia-correction!
    """
    def pKKT_solve(self, X, mu, soc=False):
        if not soc:
            print("-"*50+" determining search direction ")
        x, lam, z = self.split(X)
        xhat      = np.zeros(self.n)
        lamhat    = np.zeros(self.m)
        zhat      = np.zeros(self.n2)

        Ak, Jk, drk = self.formH(X, mu)
        JkT = Jk.transpose()
        rk = np.zeros(self.n+self.m)
        rk[:self.n] = self.problem.Dxf(x) + JkT.dot(lam)

        if not soc:
            rk[self.n:] = self.problem.c(x)[:]
        else:
            rk[self.n:] = self.ck_soc[:]
        rk[self.n1:self.n] += drk[:]
        

        # use same regularization for soc as for the step computation
        if soc:
            if self.sparse_struct:
                Ak[:self.n,:self.n] += self.deltalast * sps.identity(self.n)
            else:
                Ak[:self.n,:self.n] += self.deltalast * np.identity(self.n)
        
        sol = self.linsolve(Ak, -rk)
                
        """
        Here is where the inertia-correcting scheme will begin
        """
        xhat[:]   = sol[:self.n]
        lamhat[:] = sol[self.n:]
        lamplus   = (lamhat + lam)
        rhohat    = xhat[self.n1:].copy()
        rho       = x[self.n1:].copy()

        if soc:
            zhat[:] = -1.*(z[:] + (z[:]*rhohat[:] - mu) / (rho[:] - self.rhol[:]))
            return xhat, lamhat, zhat


        Wk      = Ak[:self.idx1, :self.idx1]

        if np.inner(xhat, Wk.dot(xhat)) + max(0, -1.*np.inner(lamplus, rk[self.idx1:])) >= self.alphad * np.inner(xhat, xhat):
            zhat[:] = -1.*(z[:] + (z[:]*rhohat[:] - mu) / (rho[:] - self.rhol[:]))  
            self.deltalast = 0.
            print("NO INERTIA CORRECTION WAS REQUIRED")
            return xhat.copy(), lamhat.copy(), zhat.copy()
        else:
            print("INERTIA CORRECTION REQUIRED")
            if self.deltalast == 0.:
                delta = self.delta0
            else:
                delta = max(self.deltamin, self.kappaminus*self.deltalast)
            j_ic = 0
            while j_ic < self.max_ic:
                if self.sparse_struct:
                    Akdelta = sps.bmat([[Ak[:self.idx1, :self.idx1] + delta * sps.identity(self.idx1), Ak[self.idx1:, :self.idx1]],\
                                    [Ak[:self.idx1, self.idx1:], None]], format="csr")
                else:
                    Akdelta = np.zeros((self.idx2, self.idx2))
                    Akdelta[:,:] = Ak[:,:]
                    Akdelta[:self.idx1, :self.idx1] += delta * np.identity(self.idx1)
                Wk        = Akdelta[:self.idx1, :self.idx1]
                sol       = self.linsolve(Akdelta, -rk)
                xhat[:]   = sol[:self.idx1]
                lamhat[:] = sol[self.idx1:]
                lamplus   = (lamhat + lam)
                if np.inner(xhat, Wk.dot(xhat)) + max(0, -np.inner(lamplus, rk[self.idx1:])) >= self.alphad * np.inner(xhat, xhat):
                    zhat[:] = -1.*(z[:] + (z[:]*rhohat[:] - mu) / (rho[:] - self.rhol[:]))  
                    self.deltalast = delta
                    print("inertia correction = {0:1.3e}".format(delta))
                    return xhat, lamhat, zhat
                elif self.deltalast == 0.:
                    delta = self.kappahatplus * delta
                else:
                    delta = self.kappaplus * delta
                j_ic += 1
            if j_ic == self.max_ic:
                print("WAS NOT ABLE TO SATISFY THE CURVATURE CONDITIONS OF THE INERTIA-FREE REGULARIZATION SCHEME!!!")
    def soc_solve(self, X, mu):
        return self.pKKT_solve(X, mu, soc=True)

    def filter_check(self, theta, phi):
        in_filter_region = False
        for i in range(len(self.F)):
            if theta >= (self.F)[i][0] and phi >= (self.F)[i][1]:
                in_filter_region = True
        return in_filter_region

    def line_search(self, X, Xhat, mu, tau):
        # indicator of Filter Line Search success
        linesearch_success = True
        
        x, lam, z = self.split(X)
        xhat, lamhat, zhat = self.split(Xhat)
        
        rho    = x[self.n1:].copy()
        rhohat = xhat[self.n1:].copy()
        
        # BEGIN A-5.1.
        """
          determine maximum step-length to ensure
          primal-dual update is interior
        """
        alpha_max = 1.
        alphaz   = 1.
        for i in range(self.n2):
            rhoi    = rho[i]
            rhohati = rhohat[i]
            rholi   = self.rhol[i]
            zi      = z[i]
            zhati   = zhat[i]
            if rhohati < 0.:
                alpha_tmp = -1. * tau * (rhoi - rholi) / rhohati
                alpha_max = min(alpha_max, alpha_tmp)
            if zhati < 0.:
                alphaz_tmp = -1.*tau*zi / zhati
                alphaz = min(alphaz, alphaz_tmp)
        alpha = alpha_max
        print("alpha = {0:1.3e}, alphaz = {1:1.3e}".format(alpha, alphaz))
        # END A-5.1.

        
        max_backtrack     = 40
        it_backtrack      = 0
        
        xtrial = np.zeros(self.n)
        x_soc  = np.zeros(self.n)
        # A-5.2--> A-5.10 --> A-5.2 --> ... loop
        while it_backtrack < max_backtrack:
            # ------ A-5.2.
            xtrial[:] = x[:] + alpha * xhat[:]
            # ------
            
            # ------ A.5.3
            # if not in filter region go to A.5.4, potential exit and then A.5.5
            # else go to A.5.5
            Dxphi_xhat = np.inner(self.problem.Dxphi(x, mu), xhat)
            
            descent_direction = (Dxphi_xhat < 0.)
            in_filter_region  = self.filter_check(self.problem.theta(xtrial), self.problem.phi(xtrial, mu))

            print("in filter region? ", in_filter_region)
            if not in_filter_region:
                # ------ A.5.4: Check sufficient decrease
                # CASE I
                print("A.5.4")
                print("theta(x) = {0:1.2e}".format(self.problem.theta(x)))
                angle = np.arccos(Dxphi_xhat / (np.linalg.norm(xhat, 2) * np.linalg.norm( \
                        self.problem.Dxphi(x, mu), 2))) * 180 / np.pi
                print("angle between xhat and Dxphi = {0:.1f} (degrees)".format(angle))
                print("descent direction? ", descent_direction)
                print("theta(x) < theta_min? ", self.problem.theta(x) <= self.theta_min)
                
                if not descent_direction:
                    switch_condition = False
                else:
                    switch_condition  = (alpha*(-1.*Dxphi_xhat)**self.sphi > self.delta * (self.problem.theta(x))**self.stheta)
                    
                if self.problem.theta(x) <= self.theta_min and switch_condition:
                    sufficient_decrease = (self.problem.phi(xtrial, mu) <= \
                                              self.problem.phi(x, mu) + self.eta * alpha * Dxphi_xhat)
                    if sufficient_decrease:
                        # ACCEPT THE TRIAL STEP
                        print("step accepted A-5.4 CASE I")
                        linesearch_success = True
                        return xtrial.copy(), xhat.copy(), lamhat.copy(), alpha, alphaz, linesearch_success
                # CASE II
                else:
                    if self.problem.theta(xtrial) <= (1.-self.gtheta)*self.problem.theta(x) or \
                        self.problem.phi(xtrial, mu) <= self.problem.phi(x, mu) - \
                        self.gphi*self.problem.theta(x):
                            print("step accepted A-5.4 CASE II")
                            # ACCEPT THE TRIAL STEP
                            linesearch_success = True
                            return xtrial.copy(), xhat.copy(), lamhat.copy(), alpha, alphaz, True
            # A.5.5: Initialize the second order correction
            print("A.5.5")
            print("theta(xtrial) = {0:1.2e}, theta(x) = {1:1.2e}".format(self.problem.theta(xtrial), self.problem.theta(x)))

            if not (it_backtrack > 0 or self.problem.theta(xtrial) <= self.problem.theta(x)):
                p = 1
                maxp = 4
                
                # equation (27)
                self.ck_soc  = alpha*self.problem.c(x) + self.problem.c(xtrial)
                theta_old_sc = self.problem.theta(x)
                
                # A-5.6--> A-5.9 --> A-5.6 --> ... loop
                while p < maxp:
                    print("A.5.6")
                    # A-5.6: Compute the second order correction
                    xhat_soc, lamhat_soc, zhat_soc = self.soc_solve(X, mu)
                    rhohat_soc = xhat_soc[self.n1: ].copy()
                    alpha_soc   = 1.
                    """
                    Since x is a stacked vector of the state and parameter,
                    we access the components corresponding to the parameter m
                    via the [n1, n1+1,...n) indices of x
                    """
                    for i in range(self.n2):
                        rhohat_soci = rhohat_soc[i]
                        rhoi        = rho[i]
                        rholi       = self.rhol[i]
                        if rhohat_soci < 0.:
                            alpha_tmp = -1.*tau* (rhoi - rholi) / rhohat_soci
                            if 0. < alpha_tmp and alpha_tmp < 1.:
                                alpha_soc = min(alpha_soc, alpha_tmp)
                    print("alpha_soc = {0:1.3e}".format(alpha_soc))
                    x_soc = x + alpha_soc * xhat_soc

                    # end A-5.6 
                    # A-5.7: Check acceptability to the filter
                    # If in filter region, exit A.5.6 --> A.5.9 loop to return to A.5.10
                    if self.filter_check(self.problem.theta(x_soc), self.problem.phi(x_soc, mu)):
                        print("soc trial point in filter region")
                        break
                    else:
                        print("not acceptable to the filter")
                    # otherwise continue in loop to step A.5.8
                    # A-5.8: Check sufficient decrease with respect to the current iterate (in SOC)
                    # CASE I
                    Dxphi_xhat_soc = np.inner(self.problem.Dxphi(x, mu), xhat_soc)
                    if self.problem.theta(x) < self.theta_min and \
                            Dxphi_xhat_soc < 0. and \
                            alpha_max*(-1.*Dxphi_xhat_soc)**self.sphi > \
                            self.delta*(self.problem.theta(x))**self.stheta:
                        if self.problem.phi(x_soc, mu) < self.problem.phi(x, mu) + \
                                self.eta*alpha_soc*Dxphi_xhat_soc:
                            # ACCEPT THE SOC TRIAL STEP
                            print("backtracking successful")
                            linesearch_success = True
                            return x_soc, xhat_soc, lamhat_soc, alpha_soc, alphaz, linesearch_success
                    # CASE II
                    else:
                        if self.problem.theta(x_soc) <= (1.-self.gtheta)*self.problem.theta(x) or \
                                self.problem.phi(x_soc, mu) <= \
                                     self.problem.phi(x, mu) - self.gphi*self.problem.theta(x):
                            # ACCEPT THE SOC TRIAL STEP
                            return x_soc, xhat_soc, lamhat_soc, alpha_soc, alphaz, True
                    # A-5.9: Next second order correction
                    if p == maxp or self.problem.theta(x_soc) > self.k_soc*theta_old_sc:
                        break
                    else:
                        p += 1
                        self.ck_soc = alpha_soc*self.ck_soc + self.problem.c(x_soc)
                        theta_old_sc = self.problem.theta(x_soc)
            # A-5.10: Choose a new trial step size
            print("A-5.10")
            alpha *= 0.5
            it_backtrack += 1                        
        if it_backtrack == max_backtrack:
            linesearch_success = False
            print("FILTER LINE SEARCH FAILURE -- should continue to feasibility restoration A-9")
        return xtrial, xhat, lamhat, alpha, alphaz, linesearch_success

    def project_z(self, X, mu):
        x, lam, z = X[0].copy(), X[1].copy(), X[2].copy()
        """
        Make copies of numpy 'np' copies of z, m, and ml
        Return a copy of the nodal DOF values of the projected multiplier z
        according to (16) of Wachter, Biegler
        """
        zproj = np.zeros(self.n2)
        rho   = x[self.n1:]
        for i in range(self.n2):
            zproj[i] = max(min(z[i], self.kSig*mu / (rho[i] - self.rhol[i])), mu / (self.kSig*(rho[i]-self.rhol[i])))
        return zproj.copy()

    """
    Determine a minimizer of the objective, utilizing the IP method with filter-line search
    """
    def solve(self, tol, max_iterations, mu0):
        mu  = mu0
        self.it = 0

        X    = self.split(self.X)
        Xhat = self.split(self.Xhat)
        theta0 = self.problem.theta(X[0])
        self.theta_min = 1.e-4 * max(1., theta0)
        self.theta_max  = 1.e4  * max(1., theta0)
        Es = []
        Mus = []
        tau = max(self.tau_min, 1. - mu)
        # initialize the filter for the given value of mu
        self.F = [[self.theta_max, -1.*np.inf]]
        while self.it < max_iterations:
            print("-"*50+" it = {0:d}".format(self.it))
            Es.append(self.problem.E(X, 0., self.smax)[:])
            Mus.append(mu)
            
            # -------- Check for convergence to global problem
            if self.problem.E(X, 0., self.smax)[0] < tol:
                print("solved interior point problem")
                break
            # -------- Check for convergence of the barrier subproblem
            # A-3
            while self.problem.E(X, mu, self.smax)[0] < self.keps*mu:
                print("solved barrier problem (mu = {0:1.3e})".format(mu))
                # decrease barrier parameter 
                mu  = max(tol / 10., min(self.kmu*mu, mu**self.thetamu))
                tau = max(self.tau_min, 1. - mu)
                # Re-initialize the filter for the new barrier subproblem
                self.F = [[self.theta_max, -1.*np.inf]]
            
            # -------- Obtain search direction from perturbed KKT system
            # A-4
            Xhat[0], Xhat[1], \
                Xhat[2] = self.pKKT_solve(self.split(X), mu)
            
            # -------- Determine appropriate step length
            # A-5 Backtracking line-search
            x, xhat, lamhat, alpha, alphaz, linesearch_success = self.line_search(self.split(X), self.split(Xhat), mu, tau)
            print("linesearch success? ", linesearch_success)
            
            # check if line search was successful if not we restore feasibility
            if linesearch_success:
                # if either equations (19) or (20) Wachter-Biegler do not hold, then the filter is augmented
                Dx_phi_xhat       = np.dot(self.problem.Dxphi(X[0], mu), xhat[:])
                descent_direction = (Dx_phi_xhat < 0.)
                if not descent_direction:
                    switch_condition = False
                else:
                    switch_condition = (alpha*(-Dx_phi_xhat)**self.sphi > self.delta*(self.problem.theta(X[0]))**self.stheta)
                sufficient_decrease = (self.problem.phi(x, mu) <= self.problem.phi(X[0], mu) + self.eta * alpha * Dx_phi_xhat)
                if switch_condition == False or sufficient_decrease == False:
                    self.F.append([(1.-self.gtheta)*self.problem.theta(X[0]),\
                        self.problem.phi(X[0], mu) - self.gphi*self.problem.theta(X[0])])
                # A-6: Accept trial point
                print("accepted trial point for the subproblem")
                X[0][:] = x[:]
                X[1][:] += alpha * lamhat[:]
                X[2][:] += alphaz * Xhat[2][:]
                X[2][:] = self.project_z(X, mu)
            else:
                print("feasibility restoration")
                if self.problem.theta(X[0]) < 1.e-13:
                    print("ATTEMPTING TO RESTORE FEASIBILITY WHEN ITERATE IS FEASIBLE")
                # A-9 
                # Augment the filter
                self.F.append([(1.-self.gtheta)*self.problem.theta(X[0]),\
                    self.problem.phi(X[0], mu) - self.gphi*self.problem.theta(X[0])])
                X = self.restore_feasibility(X, tau)
            self.it += 1
        return X, mu, Es, Mus

    def restore_feasibility(self, X, tau):
        print("haven't implemented feasibility restoration! :( .... exiting :(")
        exit()
        x, lam, z = self.split(X) 
        J         = self.problem.Dxc(x)
        
        A         = np.zeros((self.idx2, self.idx2))
        r         = np.zeros(self.idx2)
        A[:self.idx1, :self.idx1] = np.identity(self.idx1)
        A[:self.idx1, self.idx1:self.idx2] = (J.T)[:,:]
        A[self.idx1:self.idx2, :self.idx1] = J[:,:]
        r[self.idx1:self.idx2] = self.problem.c(x)[:]

        sol = np.linalg.solve(A, -r)
        xhat   = sol[         : self.idx1]
        lamhat = sol[self.idx1: self.idx2]

        # determine appropriate step length
        rho = x[self.n1:]
        rhohat = xhat[self.n1:]
        alpha_max = 1.
        for i in range(self.n2):
            rhoi    = rho[i]
            rhohati = rhohat[i]
            rholi = self.rhol[i]
            if rhohati < 0.:
                alpha_max = min(alpha_max, -1. * tau * (rhoi - rholi) / rhohati)
        return x + alpha_max * xhat, lam, z

