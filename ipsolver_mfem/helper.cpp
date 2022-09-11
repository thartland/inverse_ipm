#include "mfem.hpp"
#include "helper.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

relaxedDirac::relaxedDirac(int Dim, Vector & ObsPts, double Sig)
{
  dim = Dim;
  npts = ObsPts.Size() / dim;
  obsPts.SetSize(npts * dim);
  obsPts.Set(1.0, ObsPts);
  sig = Sig;
}


double relaxedDirac::relaxedDiracFunEval(const Vector & p)
{
  double f = 0.0;
  double arg;
  double ftemp;
  double xij;
  for(int i = 0; i < npts; i++) // sum each relaxed Dirac function contribution
  {
    arg = 0.0;
    ftemp = 0.0;
    for(int j = 0; j < dim; j++) // all components of the point (center) of the current relaxed Dirac
    {
      xij = obsPts(i + j * npts);
      arg += pow((p(j) - xij) / sig, 2.);
    }
    ftemp = exp(-0.5*arg);
    for(int j = 0; j < dim; j++)
    {
      ftemp /= (sig * pow(2.*M_PI, 0.5));
    }
    for(int j = 0; j < dim; j++)
    {
      xij = obsPts(i + j * npts);
      // if the given observation point is near a boundary
      // of the spatial domain (assumed to be a unit square)
      // then scale the relaxed Dirac distributions
      // so that they still have unit mass
      if( xij < 0. + 1.e-13 || xij > 1. - 1.e-13)
      {
        ftemp *= 2.;
      }
    }
    f += ftemp;   
  }
  return f;
}

relaxedDirac::~relaxedDirac() {}



BlockOperatorSum::BlockOperatorSum(Array<int> Blockoffsets_x, \
                                 Array<int> Blockoffsets_y) : \
                  BlockOperator(Blockoffsets_x, Blockoffsets_y)
{
  c1 = 1.0;
  c2 = 1.0;
  blockoffsets_x = Blockoffsets_x;
  blockoffsets_y = Blockoffsets_y;
  bOp1 = new BlockOperator(blockoffsets_x, blockoffsets_y);
  bOp2 = new BlockOperator(blockoffsets_x, blockoffsets_y);
}

void BlockOperatorSum::Mult(const Vector &x, Vector &y) const
{
  Vector z;
  z.SetSize(y.Size()); z = 0.0;
  bOp1->Mult(x, z);
  z *= c1;
  bOp2->Mult(x, y);
  y *= c2;
  y.Add(1.0, z);
}


BlockOperatorSum::~BlockOperatorSum()
{
  delete bOp1;
  delete bOp2;
}

SumOperator::SumOperator(const Operator *A, const Operator *B) : \
                  Operator(A->Height(), A->Width()), A_(*A), B_(*B)
{
  MFEM_VERIFY(A_.Height() == B_.Height() && A_.Width() == B_.Width(), "Operator additional only defined for operators of the same size");
}

void SumOperator::Mult(const Vector &x, Vector &y) const
{
  Vector yhelp(A_.Height()); yhelp = 0.0;
  A_.Mult(x, yhelp);
  B_.Mult(x, y);
  y.Add(1.0, yhelp);
}

void SumOperator::MultTranspose(const Vector &y, Vector &x) const
{
  Vector xhelp(A_.Width()); xhelp = 0.0;
  A_.MultTranspose(y, xhelp);
  B_.MultTranspose(y, x);
  x.Add(1.0, xhelp); 
}

SumOperator::~SumOperator()
{ }

SumOperatorBlock::SumOperatorBlock(BlockOperator *A, BlockOperator *B) : \
                  BlockOperator(A->RowOffsets(), A->ColOffsets())
{
  for(int i = 0; i < A->NumRowBlocks(); i++)
  {
    for(int j = 0; j < A->NumColBlocks(); j++)
    {
      if(!A->IsZeroBlock(i,j) && !B->IsZeroBlock(i,j))
      {
        Cij.Append(new SumOperator(&(A->GetBlock(i,j)), &(B->GetBlock(i,j))));
        SetBlock(i, j, Cij[Cij.Size()-1]);
      }
      if(!A->IsZeroBlock(i,j) && B->IsZeroBlock(i,j))
      {
        SetBlock(i, j, &(A->GetBlock(i,j)));
      }
      if(A->IsZeroBlock(i,j) && !B->IsZeroBlock(i,j))
      {
        SetBlock(i, j, &(B->GetBlock(i,j)));
      }
    }
  }
}

SumOperatorBlock::~SumOperatorBlock()
{
  for(int i = 0; i < Cij.Size(); i++)
  {
    delete Cij[i];
  }
  Cij.LoseData();
}



TransposeOperatorBlock::TransposeOperatorBlock(BlockOperator *A) : BlockOperator(A->ColOffsets(), A->RowOffsets())
{
  for(int i = 0; i < A->NumRowBlocks(); i++)
  {
    for(int j = 0; j < A->NumColBlocks(); j++)
    {
      if(!A->IsZeroBlock(i,j))
      {
        Cji.Append(new TransposeOperator(&(A->GetBlock(i,j))));
        SetBlock(j, i, Cji[Cji.Size()-1]);
      }
    }
  }
}

TransposeOperatorBlock::~TransposeOperatorBlock()
{
  for(int i = 0; i < Cji.Size(); i++)
  {
    delete Cji[i];
  }
  Cji.LoseData();
}


DiagonalOperator::DiagonalOperator(Vector *D) : Operator(D->Size(), D->Size())
{
  d.SetSize(D->Size()); d = 0.0;
  for(int i = 0; i < D->Size(); i++)
  {
    d(i) = D->Elem(i);
  }
}

void DiagonalOperator::Mult(const Vector &x, Vector &y) const
{
  for(int i = 0; i < d.Size(); i++)
  {
    y(i) = (x(i) * d(i));
  }
}

DiagonalOperator::~DiagonalOperator()
{
}



GSPreconditioner::GSPreconditioner(BlockOperator *A, double tol) : Solver(A->Height()) 
{
  Abo          = A;
  col_offsets  = A->ColOffsets();
  row_offsets  = A->RowOffsets();
  nRowBlocks   = A->NumRowBlocks();
  nColBlocks   = A->NumColBlocks();
  r0.SetSize(row_offsets[1] - row_offsets[0]);
  r1.SetSize(row_offsets[2] - row_offsets[1]);
  r2.SetSize(row_offsets[3] - row_offsets[2]);
  tmp0.SetSize(row_offsets[1] - row_offsets[0]);
  tmp1.SetSize(row_offsets[2] - row_offsets[1]);
  tmp2.SetSize(row_offsets[3] - row_offsets[2]);

  int precPrintLevel   = 0;
  int solverPrintLevel = 0;
  int MaxIter          = 200;

  // Ju
  A20       = dynamic_cast<HypreParMatrix *>(&(A->GetBlock(2, 0)));
  A20solver = new HyprePCG(MPI_COMM_WORLD);

  A20prec   = new HypreBoomerAMG;
  A20prec->SetPrintLevel(precPrintLevel);
  A20solver->SetTol(tol);
  A20solver->SetMaxIter(MaxIter);
  A20solver->SetPrintLevel(solverPrintLevel);
  A20solver->SetPreconditioner(*A20prec);
  A20solver->SetOperator(*A20);

  // Ju^T
  // WARNING -- currently assuming A02 = A20!!!
  /*A02       = dynamic_cast<HypreParMatrix *>(&(A->GetBlock(0, 2)));
  A02solver = new HyprePCG(MPI_COMM_WORLD);
  A02prec   = new HypreBoomerAMG;
  A02prec->SetPrintLevel(precPrintLevel);
  A02solver->SetTol(tol);
  A02solver->SetMaxIter(MaxIter);
  A02solver->SetPrintLevel(solverPrintLevel);
  A02solver->SetPreconditioner(*A02prec);
  A02solver->SetOperator(*A02);*/

  // Wmm
  A11       = dynamic_cast<HypreParMatrix *>(&(A->GetBlock(1, 1)));
  A11solver = new HyprePCG(MPI_COMM_WORLD);
  A11prec   = new HypreBoomerAMG;
  A11prec->SetPrintLevel(precPrintLevel);
  A11solver->SetTol(tol);
  A11solver->SetMaxIter(MaxIter);
  A11solver->SetPrintLevel(solverPrintLevel);
  A11solver->SetPreconditioner(*A11prec);
  A11solver->SetOperator(*A11);
}

void GSPreconditioner::Mult(const Vector & x, Vector & y) const
{
  MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
  MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

  x.Read();
  double zero = 0.0;
  y.Write(); y = zero;
  tmp0 = zero;
  tmp1 = zero;
  tmp2 = zero;


  
  xblock.Update(const_cast<Vector&>(x), col_offsets);
  yblock.Update(y, row_offsets);

  r2.Set(1.0, xblock.GetBlock(2));
  A20solver->Mult(r2, tmp0);
  A20solver->GetNumIterations(numits);
  A20applies.Append(numits);
  yblock.GetBlock(0).Set(1.0, tmp0);

  r0.Set(1.0, xblock.GetBlock(0));
  (Abo->GetBlock(0,0)).Mult(yblock.GetBlock(0), tmp0);
  r0.Add(-1.0, tmp0);
  A20solver->Mult(r0, tmp2);
  A20solver->GetNumIterations(numits);
  A02applies.Append(numits);
  //A02solver->Mult(r0, tmp2);
  yblock.GetBlock(2).Set(1.0, tmp2);

  r1.Set(1.0, xblock.GetBlock(1));
  (Abo->GetBlock(1, 0)).Mult(yblock.GetBlock(0), tmp1);
  r1.Add(-1.0, tmp1);
  (Abo->GetBlock(1, 2)).Mult(yblock.GetBlock(2), tmp1);
  r1.Add(-1.0, tmp1);
  A11solver->Mult(r1, tmp1);
  A11solver->GetNumIterations(numits);
  A11applies.Append(numits);
  yblock.GetBlock(1).Set(1.0, tmp1);
  
  for (int iRow=0; iRow < nRowBlocks; ++iRow)
  {
     yblock.GetBlock(iRow).SyncAliasMemory(y);
  }
  double ynorm = InnerProduct(MPI_COMM_WORLD, y, y);
}


void GSPreconditioner::SetOperator(const Operator & op)
{}

void GSPreconditioner::saveA02applies(string saveFile)
{
  if(Mpi::Root())
  {
    std::ofstream A02DataStream;
    A02DataStream.open(saveFile, ios::out | ios::trunc);
    for(int ii = 0; ii < A02applies.Size(); ii++)
    {
      A02DataStream << setprecision(30) << A02applies[ii] << endl;
    }
    A02DataStream.close();
  }
}

void GSPreconditioner::saveA11applies(string saveFile)
{
  if(Mpi::Root())
  {
    std::ofstream A11DataStream;
    A11DataStream.open(saveFile, ios::out | ios::trunc);
    for(int ii = 0; ii < A11applies.Size(); ii++)
    {
      A11DataStream << setprecision(30) << A11applies[ii] << endl;
    }
    A11DataStream.close();
  }
}

void GSPreconditioner::saveA20applies(string saveFile)
{
  if(Mpi::Root())
  {
    std::ofstream A20DataStream;
    A20DataStream.open(saveFile, ios::out | ios::trunc);
    for(int ii = 0; ii < A20applies.Size(); ii++)
    {
      A20DataStream << setprecision(30) << A20applies[ii] << endl;
    }
    A20DataStream.close();
  }
}










GSPreconditioner::~GSPreconditioner()
{
/* WARNING! do not delete A02, A20, A11 as they point to the physical memory location as
 * as that of the blocks of A
 */
 delete A20prec;
 delete A20solver;
 /*delete A02prec;
 delete A02solver;*/
 delete A11prec;
 delete A11solver;
}







void constructBlockOpFromBlockOp(BlockOperator* A11, BlockOperator*  A12, BlockOperator* A21, BlockOperator & A)
{
  for(int i = 0; i < A11->NumRowBlocks(); i++)
  {
    for(int j = 0; j < A11->NumColBlocks(); j++)
    {
      if(!A11->IsZeroBlock(i,j))
      {
        A.SetBlock(i, j, &(A11->GetBlock(i,j)));
      }
    }
  }
  for(int i = 0; i < A12->NumRowBlocks(); i++)
  {
    for(int j = 0; j < A12->NumColBlocks(); j++)
    {
      if(!A12->IsZeroBlock(i,j))
      {
        A.SetBlock(i, j+A11->NumColBlocks(), &(A12->GetBlock(i,j)));
      }
    }
  }
  for(int i = 0; i < A21->NumRowBlocks(); i++)
  {
    for(int j = 0; j < A21->NumColBlocks(); j++)
    {
      if(!A21->IsZeroBlock(i,j))
      {
        A.SetBlock(i+A11->NumRowBlocks(), j, &(A21->GetBlock(i,j)));
      }
    }
  }  	
}



SaveDataSolverMonitor::SaveDataSolverMonitor(string savefile) : saveFile(savefile) {};

void SaveDataSolverMonitor::MonitorResidual(int i, double norm, const Vector &r, bool final)
{
  if(Mpi::Root())
  {
    if(i==0)
    {
      resDataStream.open(saveFile, ios::out | ios::trunc);
    }

    resDataStream << setprecision(30) << norm << "\n";

    if(final)
    {
      resDataStream.close();
    }
  }
}


SaveDataSolverMonitor::~SaveDataSolverMonitor() {};
