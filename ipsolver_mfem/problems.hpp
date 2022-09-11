#include "mfem.hpp"
#include "helper.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS





// abstract optimizationProblem class
class optimizationProblem
{
protected:


    // mass bilinearforms, sparse matrices, solvers and lumped objects
    // for both the state (u) and parameter (m) 
    ParBilinearForm* Muform;
    HypreParMatrix   Mu;
    HyprePCG*        Musolver;
    
    ParBilinearForm* Mform;
    HypreParMatrix   Mm;
    Vector           Mmlumped;
    DiagonalOperator *MmlumpedOp;

    Array<int> empty_tdof_list; // needed for calls to FormSystemMatrix
    // mass operators for the primal variable x = (u, m)
    // Mx = [[Mu   0]
    //        0    Mm]]
    BlockOperator * Mx;
    CGSolver      * Mxsolver;

    Vector ml;

    /*
     * dimU    -- local dimension of true dof for state u
     * dimM    -- local dimension of true dof for parameter m
     * dimUdof -- local dimension of dof for state u
     * dimMdof -- local dimension of dof for parameter m
     */
    int dimU, dimM;
    int dimUdof, dimMdof;
    Array<int> block_offsetsx;
    Array<int> block_offsetsxl;
    Array<int> block_offsetsu;
    MemoryType mt;
    Vector z1, z2, z3, z4, w1, w2, z1dof, z2dof, w1dof, w2dof;

    MPI_Comm MyComm;
    int MyRank, NRanks;
    bool iAmRoot;
public:
    // function spaces for both the state (u) and parameter (m)
    ParFiniteElementSpace *Vhu;
    ParFiniteElementSpace *Vhm;
    const Operator *Pu;
    const Operator *Ru;
    const Operator *Pm;
    const Operator *Rm;
	optimizationProblem(ParFiniteElementSpace*, ParFiniteElementSpace*, Vector &, MemoryType);
	virtual double f(BlockVector &) = 0;
	virtual void Dxf(BlockVector &, BlockVector *) = 0;
	virtual void Dxxf(BlockVector &, BlockOperator* ) = 0;
	virtual void c(BlockVector &, Vector *) = 0;
	virtual void Dxc(BlockVector &, BlockOperator* ) = 0;
	virtual void DxcTp(BlockVector &, Vector &, BlockVector *) = 0;
	virtual void Dxxcp(BlockVector &, Vector &, BlockOperator *) = 0;
    virtual void feasibilityRestoration(BlockVector &, double) = 0;
	double theta(BlockVector &);
    double thetaM(BlockVector &);
	double phi(BlockVector &, double);
	void Dxphi(BlockVector &, double, BlockVector *);
    double  L(BlockVector &, Vector &, Vector &);
	void DxL(BlockVector &, Vector &, Vector &, BlockVector *);
	void DxxL(BlockVector &, Vector &, BlockOperatorSum *);
	void DxxL(BlockVector &, Vector &, BlockOperator *, BlockOperator *);
    void MxSolveMult(const Vector &, Vector &) const;
    void MuSolveMult(const Vector &, Vector &) const;
    void MuMult(const Vector &, Vector &) const;
	double  E(BlockVector &, Vector &, Vector &, double, double);
	int getdimU();
	int getdimM();
	Array<int> getblock_offsetsx();
	Array<int> getblock_offsetsxl();
	Array<int> getblock_offsetsu();
	Vector getMmlumped();
	Vector getml();
    MPI_Comm GetComm() const;
    int GetNRanks() const;
    int GetMyRank() const;
	virtual ~optimizationProblem() {};	
};



class inverseDiffusion : public optimizationProblem
{
protected:
    ParFiniteElementSpace *Vhgrad;
    ParFiniteElementSpace *VhL2;
    
    // regularization
    ParBilinearForm *Rform;
    HypreParMatrix  R; 
    // weighted mass-matrix which allows
    // us to mimick the behavior of a pointwise
    // observation operator 
    ParBilinearForm *Mwform;
    HypreParMatrix  Mw;
    HypreParVector *ud;
    
    //ParGridFunction *ud; // data


    // functionality necessary to compute f, c, ...
    DiscreteLinearOperator *Grad;
    ParMixedBilinearForm *ML2mform;
    SparseMatrix ML2m;

    ParBilinearForm *Juform;
    HypreParMatrix Ju;
    HypreParMatrix Jm;
    HypreParMatrix *JmT;

    HypreParMatrix Jum, Jmu;
    //HypreParMatrix, *Jmu;
	
    // additional data which defines the 
    // partial differential equality constraint
    ParGridFunction *g;
    ConstantCoefficient * beta;
public:
	inverseDiffusion(
	ParFiniteElementSpace *,\
        ParFiniteElementSpace *,\
        ParFiniteElementSpace *,\
        ParFiniteElementSpace *,\
	ParGridFunction &,\
	ParGridFunction &,\
        Vector &,\
	ParGridFunction &,\
	Array<int>,\
        double,\
	double,\
	double,\
	MemoryType);
	double f(BlockVector &);
	void Dxf(BlockVector &, BlockVector *);
	void Dxxf(BlockVector &, BlockOperator* );
	void c(BlockVector &, Vector *);
	void Dxc(BlockVector &, BlockOperator* );
	void DxcTp(BlockVector &, Vector &, BlockVector *);
	void Dxxcp(BlockVector &, Vector &, BlockOperator *);
    void feasibilityRestoration(BlockVector &, double);
	//void GetJuHypreParMatrix(HypreParMatrix &);
    virtual ~inverseDiffusion();
};

class inverseDiffusion2 : public optimizationProblem
{
protected:
    ParFiniteElementSpace *Vhgrad;
    ParFiniteElementSpace *VhL2;
    
    // regularization
    ParBilinearForm *Rform;
    HypreParMatrix  R; 
    // weighted mass-matrix which allows
    // us to mimick the behavior of a pointwise
    // observation operator 
    ParBilinearForm *Mwform;
    HypreParMatrix  Mw;
    HypreParVector *ud;
    
    //ParGridFunction *ud; // data


    // functionality necessary to compute f, c, ...
    DiscreteLinearOperator *Grad;
    ParMixedBilinearForm *ML2mform;
    SparseMatrix ML2m;

    ParBilinearForm *Juform;
    HypreParMatrix Ju;
    HypreParMatrix Jm;
    HypreParMatrix *JmT;

    HypreParMatrix Jum, Jmu;
    //HypreParMatrix, *Jmu;
    
    // additional data which defines the 
    // partial differential equality constraint
    ParGridFunction *g;
    ConstantCoefficient * beta;
public:
    inverseDiffusion2(
    ParFiniteElementSpace *,\
    ParFiniteElementSpace *,\
    ParFiniteElementSpace *,\
    ParFiniteElementSpace *,\
    ParGridFunction &,\
    ParGridFunction &,\
    Vector &,\
    Array<int>,\
    double,\
    double,\
    double,\
    MemoryType);
    double f(BlockVector &);
    void Dxf(BlockVector &, BlockVector *);
    void Dxxf(BlockVector &, BlockOperator* );
    void c(BlockVector &, Vector *);
    void Dxc(BlockVector &, BlockOperator* );
    void DxcTp(BlockVector &, Vector &, BlockVector *);
    void Dxxcp(BlockVector &, Vector &, BlockOperator *);
    void feasibilityRestoration(BlockVector &, double);
    //void GetJuHypreParMatrix(HypreParMatrix &);
    virtual ~inverseDiffusion2();
};



#endif
