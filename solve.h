#ifndef SOLVE_H
#define SOLVE_H

template<class MV, class OP>
MV  solve ( MV& X, const MV& B, const OP& A)
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcpFromRef;
  typedef typename MV::scalar_type scalar_type;

  RCP<ParameterList> solverParams = parameterList();
solverParams->set ("Num Blocks", 40);
solverParams->set ("Maximum Iterations", 5000);
solverParams->set ("Convergence Tolerance", 1.0e-2);
Belos::SolverFactory<scalar_type, MV, OP> factory;
  RCP<Belos::SolverManager<scalar_type, MV, OP> > solver =
    factory.create ("GMRES", solverParams);

  // Create a LinearProblem struct with the problem to solve.
  // A, X, B, and M are passed by (smart) pointer, not copied.
  typedef Belos::LinearProblem<scalar_type, MV, OP> problem_type;
  RCP<problem_type> problem =
    rcp (new problem_type (rcpFromRef (A), rcpFromRef (X), rcpFromRef (B)));
 problem->setProblem ();

  // Tell the solver what problem you want to solve.
  solver->setProblem (problem);
  Belos::ReturnType result = solver->solve();
 int numIters = solver->getNumIters();
if (result == Belos::Converged) {
//   std::cout << "The Belos solve took " << numIters << " iteration(s) to reach "
//      "a relative residual tolerance of " << 1.0e-4 << "." << std::endl;
  } else {
  // std::cout << "The Belos solve took " << numIters << " iteration(s), but did not reach "
    //  "a relative residual tolerance of " << 1.0e-4 << "." << std::endl;
  }


return X;
}

#endif
