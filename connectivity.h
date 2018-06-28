#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Version.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
//#include <Ifpack2_Factory.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include<cmath>
#include<ctime>
#include<fstream>
#include <Kokkos_Array.hpp>
#include"time.h"
#define pi 3.14

class Connectivity
{
public:

  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Map<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Map<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::CrsMatrix<> matrix_type;
  typedef Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type> op_type;
  typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type> vec_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type> map_type;
  typedef typename Kokkos::Details::ArithTraits<scalar_type>::val_type impl_scalar_type;
  typedef  Tpetra::CrsMatrix<>::node_type Node;
  typedef Tpetra::CrsMatrix<>::device_type device_type;
  typedef Tpetra::CrsMatrix<>:: execution_space execution_space;
  typedef Kokkos::CrsMatrix<impl_scalar_type, local_ordinal_type, execution_space> local_matrix_type;
  typedef local_matrix_type::values_type k_values;

    void setSize(int x, int y)
    {

       numHor=x;
       numVer=y;


    }
    void setConnectionMatrix(Kokkos::View<double*,Kokkos::LayoutRight> & arr1,Kokkos::View<double**,Kokkos::LayoutRight> & arr2 )
 {

  Kokkos::parallel_for((Connectivity::numHor)*(Connectivity::numVer) ,KOKKOS_LAMBDA(Tpetra::global_size_t i){

  arr1(i)=i;



});

int team_size = std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::DefaultHostExecutionSpace>::value?
                  1:256;



typedef Kokkos::TeamPolicy<>::member_type team_member;
Kokkos::parallel_for(Kokkos::TeamPolicy<>(Connectivity::numHor,team_size),KOKKOS_LAMBDA(const team_member &thread){
const Tpetra::global_size_t j=thread.league_rank();
Kokkos::parallel_for(Kokkos::TeamThreadRange(thread,Connectivity::numVer),[&](const Tpetra::global_size_t &i){
arr2(i,j)=i*numHor+j;


});


});



 }



private:
   Tpetra::global_size_t numHor;
   Tpetra::global_size_t numVer;




};

#endif
