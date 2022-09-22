#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

class relaxedDirac
{
protected:
	double sig;    // width of Gaussian Dirac approximations
	Vector obsPts; // observation points
	double fval;   // to not need to redeclare double
	int dim;       // dimension of PDE domain \Omega
	int npts;      // number of points 
	               // npts * dim = dim(obsPts)
	double tol;
	double x0, x1, x2;
public:
	relaxedDirac(int , Vector &, double);
    double relaxedDiracFunEval(const Vector &);
	virtual ~relaxedDirac();
};

class BlockOperatorSum : public BlockOperator
{
protected:
	Array<int> blockoffsets_x;
	Array<int> blockoffsets_y;
public:
	double c1;
	double c2;
	BlockOperator *bOp1;
	BlockOperator *bOp2;
	BlockOperatorSum(Array<int>, Array<int>);
	BlockOperatorSum(BlockOperator *, BlockOperator *, Array<int>, Array<int>);
	void Mult(const Vector &, Vector &) const;
	virtual ~BlockOperatorSum();
};

class SumOperator : public Operator
{
private:
  const Operator& A_;
  const Operator& B_;
public:
  SumOperator(const Operator *, const Operator *);
  void Mult(const Vector &, Vector &) const;
  void MultTranspose(const Vector &, Vector &) const;
  virtual ~SumOperator();	
};


class SumOperatorBlock : public BlockOperator
{
private:
  Array<SumOperator *> Cij;
public:
  SumOperatorBlock(BlockOperator *, BlockOperator *);
  virtual ~SumOperatorBlock();	
};

class TransposeOperatorBlock : public BlockOperator
{
private:
  Array<Operator *> Cji;
public:
  TransposeOperatorBlock(BlockOperator *);
  virtual ~TransposeOperatorBlock();
};


// design of the Block-GS preconditioner
class GSPreconditioner : public Solver
{
private:
  int nRowBlocks, nColBlocks;
  Array<int> col_offsets, row_offsets;

  HypreParMatrix * A20;
  HyprePCG       * A20solver;
  HypreBoomerAMG * A20prec;
  
  //HypreParMatrix * A02;
  //HyprePCG       * A02solver;
  //HypreBoomerAMG * A02prec;

  HypreParMatrix * A11;
  HyprePCG       * A11solver;
  HypreBoomerAMG * A11prec;

  mutable BlockVector xblock;
  mutable BlockVector yblock;
  mutable Vector r0;
  mutable Vector r1;
  mutable Vector r2;
  mutable Vector tmp0;
  mutable Vector tmp1;
  mutable Vector tmp2;

  mutable Vector A20x;


  mutable int numits;
  mutable Array<int> A20applies;
  mutable Array<int> A11applies;
  mutable Array<int> A02applies;
  BlockOperator * Abo;
  mutable std::ofstream A02SolveTimeDataStream;
  mutable std::ofstream A11SolveTimeDataStream;
  mutable StopWatch blockSolveStopWatch;
public:
  GSPreconditioner(BlockOperator *, double, string, string);
  void Mult(const Vector &, Vector &) const;
  void SetOperator(const Operator &);
  void saveA20applies(string);
  void saveA11applies(string);
  void saveA02applies(string);
  virtual ~GSPreconditioner();
};






class DiagonalOperator : public Operator
{
protected:
	Vector d;
public:
	DiagonalOperator(Vector *);
	virtual void Mult(const Vector &, Vector &) const;
	virtual ~DiagonalOperator();
};

void constructBlockOpFromBlockOp(BlockOperator* , BlockOperator* , BlockOperator* , BlockOperator& );



class SaveDataSolverMonitor : public IterativeSolverMonitor
{
private:
  std::ofstream resDataStream;
  string saveFile;
public:
  SaveDataSolverMonitor(string);
  void MonitorResidual(int, double, const Vector &, bool);
  virtual ~SaveDataSolverMonitor();
};
#endif