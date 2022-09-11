#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef IPSOLVER 
#define IPSOLVER

class interiorPtSolver
{
protected:
	double sMax, kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
	double thetaMax, kSoc, gTheta, gPhi, kEps;
	
    // filter
	Array<double> F1, F2;
	
	// quantities computed in lineSearch
	double alpha, alphaz;
	double thx0, thxtrial;
    double phx0, phxtrial;
    bool descentDirection, switchCondition, sufficientDecrease, lineSearchSuccess, inFilterRegion;
    double Dxphi0_xhat;

	int dimU, dimM;
	Array<int> block_offsetsx, block_offsetsu, block_offsetsumlz, block_offsetsuml;
	Vector Mmlumped, ml;

	Vector ckSoc;
	BlockOperator    *Hk1;
	BlockOperator    *Hk2;
	SumOperatorBlock *Hk;
	
	BlockOperator  *Wk;
	HypreParMatrix *Wkmm;
	HypreParMatrix *Hk2mm;
	HypreParMatrix *Wmm;
	SparseMatrix   *diagSparse;
	BlockOperator  *Jk;
	TransposeOperatorBlock *JkT;
	HypreParMatrix *Hlogbar;
	optimizationProblem* problem;
	int jOpt;
	bool converged;
	bool outputVTK;
	
	int MyRank;
    MPI_Comm MyComm;
    bool iAmRoot;
    std::ofstream IPNewtonSolveTimes;
    std::ofstream IPNewtonSysFormTimes;
    StopWatch IPNewtonSolveStopWatch;
    StopWatch IPNewtonSysFormStopWatch;

    double sc, sd, E1, E2, E3;
public:
	interiorPtSolver(optimizationProblem*);
	double computeAlphaMax(Vector& , Vector& , Vector& , double);
	double computeAlphaMax(Vector& , Vector& , double);
	void solve(BlockVector& , BlockVector& , double , int , double);
	void formH(BlockVector& , Vector& , Vector& , double, BlockOperator &);
	void pKKTSolve(BlockVector& , Vector& , Vector& , Vector&, BlockVector& , double, bool);
	void lineSearch(BlockVector& , BlockVector& , double);
	void projectZ(Vector & , Vector &, double);
	void filterCheck(double, double);
	double E(BlockVector &, Vector &, Vector &, double);
	bool GetConverged() const;
	void SetOutputVTK(bool);

	virtual ~interiorPtSolver();
};

#endif
