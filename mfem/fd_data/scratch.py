import numpy as np
import matplotlib.pyplot as plt

eps  = np.loadtxt("eps.dat")
err  = np.loadtxt("graderr.dat")
cerr = np.loadtxt("cgraderr.dat")
phierr = np.loadtxt("phigraderr.dat")

plt.plot(eps, err)
plt.plot(eps, eps*err[0]/eps[0] * 0.5)
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.plot(eps, cerr)
plt.plot(eps, eps*cerr[0]/eps[0] * 0.5)
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.plot(eps, phierr)
plt.plot(eps, eps*phierr[0]/eps[0] * 0.5)
plt.yscale('log')
plt.xscale('log')
plt.show()

np.savetxt("graderrpairs.dat", [(eps[i], err[i]) for i in range(len(err))])
np.savetxt("refcurvepairs.dat", [(eps[i], eps[i] *err[0]/eps[0] * 0.5) for i in range(len(err))])

np.savetxt("cgraderrpairs.dat", [(eps[i], cerr[i]) for i in range(len(cerr))])
np.savetxt("crefcurvepairs.dat", [(eps[i], eps[i] *cerr[0]/eps[0] * 0.5) for i in range(len(cerr))])

np.savetxt("phigraderrpairs.dat", [(eps[i], phierr[i]) for i in range(len(cerr))])
np.savetxt("phirefcurvepairs.dat", [(eps[i], eps[i] * phierr[0]/eps[0] * 0.5) for i in range(len(cerr))])
