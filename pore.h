#ifndef PORE_H
#define PORE_H
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
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_config.h"
#include "box_muller.h"
class Pore
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
  



  void setViscosity(double a, double b) { mu1=a;mu2=b;}
  void setSurfaceTension(double a) { sigma=a;}
  double getDefendingViscosity(){return mu1;}
  double getInvadingViscosity(){return mu2;}
  double getSurfaceTension(){return sigma;}


  template <class T1, class T2, class T3, class T4>
void setupLengthMatrix(int flag, T1 &L, T2 & arr2D, T3 & myGlobalElements_k, T4 & map, int m, int n,int *N)
  {

Teuchos::RCP<Teuchos::ParameterList> parlist =Teuchos::rcp(new Teuchos::ParameterList("Parameters"));

std::string xmlInFileName ="pore_input.xml";
//std::string name="numHor";
if(xmlInFileName.length())
 {
  Teuchos::updateParametersFromXmlFile(xmlInFileName,parlist.ptr());
}

  int upper=parlist->get<int>("upper")+1;
  int lower=parlist->get<int>("lower");
  

        for(size_t i=0;i<map->getNodeNumElements();i++)
{
  if(myGlobalElements_k[i]<m*n)
{
 N[0]=-1;
 N[1]=-1;
 N[2]=-1;
 N[3]=-1;

int q=myGlobalElements_k[i]/m;
int rem =myGlobalElements_k[i]%m;
  if(q>0)
 {
  N[0]=arr2D(q-1,rem);
 }
if(rem>0)
{
 N[1]=arr2D(q,rem-1);
}

if(q<m-1)
  {
   N[2]=arr2D(q+1,rem);
 }

if(rem<m-1)
  {
   N[3]=arr2D(q,rem+1);

 }

size_t NumEntries_k;


     size_t countNZ=0;
    for(int j=0;j<4;j++)
      {
        if(N[j]>=0)
       {
        countNZ=countNZ+1;
       }
      }
NumEntries_k=countNZ;

Teuchos::Array<scalar_type> values_k(NumEntries_k);
Teuchos::Array<global_ordinal_type> indices_k(NumEntries_k);

     global_ordinal_type countN=0;
      for(global_ordinal_type j=0;j<4;j++)
      {
        if(N[j]>=0)
       {

         indices_k[countN]=N[j];
         if(myGlobalElements_k[i]<m &&(( myGlobalElements_k[i]-N[j]==1)||( myGlobalElements_k[i]-N[j]==-1)) )
          {
            values_k[countN]=0;
          }
        else
   {
         double l=rand()%upper+lower;
         values_k[countN]= static_cast<scalar_type> (10);

  }

 countN=countN+1;
       }
      }


 L->insertGlobalValues (myGlobalElements_k[i], indices_k, values_k);
}
}

L->fillComplete();
  }//dont tocuh this shit

template <class T1, class T2, class T3, class T4>
void setupRadiusMatrix(int flag, T1 &R, T2 & arr2D, T3 & myGlobalElements_k, T4 & map, int m, int n,int *N)

{
        for(size_t i=0;i<map->getNodeNumElements();i++)
{
  if(myGlobalElements_k[i]<m*n)
{
 N[0]=-1;
 N[1]=-1;
 N[2]=-1;
 N[3]=-1;

int q=myGlobalElements_k[i]/m;
int rem =myGlobalElements_k[i]%m;
  if(q>0)
 {
  N[0]=arr2D(q-1,rem);
 }
if(rem>0)
{
 N[1]=arr2D(q,rem-1);
}

if(q<m-1)
  {
   N[2]=arr2D(q+1,rem);
 }

if(rem<m-1)
  {
   N[3]=arr2D(q,rem+1);

 }

size_t NumEntries_k;


     size_t countNZ=0;
    for(int j=0;j<4;j++)
      {
        if(N[j]>=0)
       {
        countNZ=countNZ+1;
       }
      }
NumEntries_k=countNZ;
Teuchos::Array<scalar_type> values_k(NumEntries_k);
Teuchos::Array<global_ordinal_type> indices_k(NumEntries_k);

     global_ordinal_type countN=0;
      for(global_ordinal_type j=0;j<4;j++)
      {
        if(N[j]>=0)
       {

         indices_k[countN]=N[j];
         if(myGlobalElements_k[i]<m &&(( myGlobalElements_k[i]-N[j]==1)||( myGlobalElements_k[i]-N[j]==-1)) )
          {
            values_k[countN]=0;
          }
        else
   {
 //        double l=rand()%10+15;
         double l=rand_normal(15,3);
         values_k[countN]= static_cast<scalar_type> (l);

  }

 countN=countN+1;
       }
      }


 R->insertGlobalValues (myGlobalElements_k[i], indices_k, values_k);
}
}

R->fillComplete();


}


 private:
   double mu1;
   double mu2;
   double sigma;

};

#endif

