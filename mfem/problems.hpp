
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class inverseDiffusion
{
protected:
	ParFiniteElementSpace *Vhu;
        ParFiniteElementSpace *Vhm;
	ParFiniteElementSpace *Vhgrad;
	ParFiniteElementSpace *VhL2;
	int Vhu_size;
        int Vhm_size;
	mutable Vector z1, z2, w1, w2;
	ParBilinearForm *Mform;
	SparseMatrix  *Mm;
	ParBilinearForm *Rform;
	SparseMatrix  *Rm;
        ParBilinearForm *Muform;
        SparseMatrix *Mu;
	ParBilinearForm *Mwform;
	SparseMatrix *Mw;
        DiscreteLinearOperator *Grad;
	ParMixedBilinearForm *ML2mform;
	SparseMatrix *ML2m;
	
	/*MixedBilinearForm *Cform;
        SparseMatrix  *C;
        SparseMatrix *Ct;
     
        LinearForm    *b;
	LinearForm    *bp;
	mutable Vector B;
	mutable Vector Bp;
	mutable Vector U;  // state
        mutable Vector P;  // adjoint
        	
	mutable Vector z1; // helper vector
	mutable Vector z2;
        mutable Vector w1;
        mutable Vector w2;
        */
	ParGridFunction *ud;
        ParGridFunction *g;
	ParGridFunction *ml;
	mutable Vector Mmlumped;
	/*GridFunction *f;
        GridFunction *ml;
        GridFunction *mhat;
        GridFunction *lamhat;*/
	ConstantCoefficient * beta;
public:
	inverseDiffusion(ParFiniteElementSpace *,\
                        ParFiniteElementSpace *,\
                        ParFiniteElementSpace *,\
                        ParFiniteElementSpace *,\
			ParGridFunction,\
			ParGridFunction,\
                        ParGridFunction,\
			ParGridFunction,\
			Array<int>,\
                        double,\
			double,\
			double);
	double f(BlockVector &);
	void Dxf(BlockVector &, BlockVector *);
	void Dxxf(BlockVector &, BlockOperator* );
	void c(BlockVector &, Vector *);
	void Dxc(BlockVector &, BlockOperator* );
	void DxcTp(BlockVector& , Vector &, BlockVector *);
	void Dxxcp(BlockVector &, Vector &, BlockOperator *);
	double phi(BlockVector &, double);
	void Dxphi(BlockVector &, double, BlockVector *);
	double L(BlockVector &, Vector &, Vector &);
        virtual ~inverseDiffusion();
};


class relaxedDirac
{
protected:
	double sig;    // width of Gaussian Dirac approximations
	Vector obsPts; // observation points
	double fval;   // to not need to redeclare double
	int dim;       // dimension of PDE domain \Omega
	int npts;      // number of points 
	 // npts * dim = dim(obsPts)
	double x0, x1, x2;
public:
	relaxedDirac(int , Vector &, double);
        double relaxedDiracFunEval(const Vector &);
	virtual ~relaxedDirac();
};
