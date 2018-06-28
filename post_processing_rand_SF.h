#ifndef POST_H
#define POST_H

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
//#include "box_muller.h"
class postProcessing
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
  







      template<class T1, class T2, class T3,class T4, class T5,class T6>
   void setupCurrentMatrix( T6 & L,T1 & Current,  Kokkos::Experimental::View<double*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<0u> >& x_1d, T2 & map, T3 & myGlobalElements, T4& K, T5 Vol_frac, double time, Kokkos::View<double **,Kokkos::LayoutRight> & arr2D,int *N, int m, int n, double dt,double surf)
{
for(size_t i=0;i<map->getNodeNumElements();i++)
{
  if(myGlobalElements[i]<m*n)
{
 N[0]=-1;
 N[1]=-1;
 N[2]=-1;
 N[3]=-1;

int q=myGlobalElements[i]/m;
int rem =myGlobalElements[i]%m;
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

int counter=0;
 for(int j=0;j<4;j++)
  {
      if (N[j]>=0)
   {
        counter++;
 }


 }
Teuchos::Array<global_ordinal_type> mat_ind(counter);// This size will always remian five
Teuchos::Array<scalar_type>mat_val(counter);
size_t num;

   if(time==0)
 {

    if (myGlobalElements[i]<m)
  {
//Teuchos::Array<global_ordinal_type> mat_ind(counter);// This size will always remian five
//Teuchos::Array<scalar_type>mat_val(counter);

K->getGlobalRowCopy(myGlobalElements[i],mat_ind,mat_val,num);
size_t NumEntries;


     size_t countNZ=0;
    for(int j=0;j<counter;j++)
      {
        if(mat_val[j]>0)
       {
        countNZ=countNZ+1;
       
       }
      }
NumEntries=countNZ;
       
 


 
   
Teuchos::Array<scalar_type> values_current(counter);
Teuchos::Array<scalar_type> values_X(counter);
Teuchos::Array<global_ordinal_type> indices_current(counter);
double diagonal=0;
global_ordinal_type countN=0;
    for(int j=0;j<counter;j++)
      {
        


        values_current[countN]=abs(x_1d(myGlobalElements[i])-x_1d(mat_ind[j]))*mat_val[j];
        values_X[countN]=values_current[countN]*dt;


        indices_current[countN]=mat_ind[j];

       countN=countN+1;
      



     }


Current->insertGlobalValues (myGlobalElements[i], indices_current, values_current);

Vol_frac->insertGlobalValues(myGlobalElements[i], indices_current, values_X);



}//if its on the first row
else // not ont he first row
{
//Teuchos::Array<global_ordinal_type> mat_ind(counter);// This size will always remian five
//Teuchos::Array<scalar_type>mat_val(counter);
//size_t num;
Teuchos::Array<global_ordinal_type> duplicate_ind(4);
Teuchos::Array<scalar_type> duplicate_val(4);
K->getGlobalRowCopy(myGlobalElements[i],mat_ind,mat_val,num);
//Vol_frac->getGlobalRowCopy(myGlobalElements[i],duplicate_ind,duplicate_val,num);
//Vol_frac->hasTransposeApply();
size_t tr;
for (int jj=0;jj<4;jj++)
 {
    if(N[jj]>=0 && N[jj]+1<myGlobalElements[i])
   {
       tr=N[jj];

   }
 
 }
     
Vol_frac->getGlobalRowCopy(myGlobalElements[tr],duplicate_ind,duplicate_val,num);

     double val;
     for (int jj=0;jj<4;jj++)
   {
         if (duplicate_ind[jj]==myGlobalElements[i])
              {
                   val=duplicate_val[jj];
               }

    }


size_t NumEntries;


     size_t countNZ=0;
    for(int j=0;j<counter;j++)
      {
        if(mat_val[j]>0)
       {
        countNZ=countNZ+1;

       }
      }
NumEntries=countNZ;
       





Teuchos::Array<scalar_type> values_current(counter);
Teuchos::Array<scalar_type> values_X(counter);
Teuchos::Array<global_ordinal_type> indices_current(counter);
double diagonal=0;
global_ordinal_type countN=0;
    for(int j=0;j<counter;j++)
      {
        

            
        values_current[countN]=(1e-3)*abs(x_1d(myGlobalElements[i])-x_1d(mat_ind[j]))*mat_val[j];
        values_X[countN]=(mat_ind[j]==tr)?val:0;
      // values_X[countN]=0;

        indices_current[countN]=mat_ind[j];

       countN=countN+1;
      

      }


Current->insertGlobalValues (myGlobalElements[i], indices_current, values_current);
Vol_frac->insertGlobalValues(myGlobalElements[i], indices_current, values_X);

}//not on the first row

}//if its at t=0

if(time !=0)
 {
  
  if (myGlobalElements[i]<m)
 {
//Teuchos::Array<global_ordinal_type> mat_ind(counter);
//Teuchos::Array<scalar_type> mat_val(counter);
K->getGlobalRowCopy(myGlobalElements[i],mat_ind,mat_val,num);
size_t NumEntries;


     size_t countNZ=0;
    for(int j=0;j<counter;j++)
      {
        if(mat_val[j]>0)
       {
        countNZ=countNZ+1;

       }
      } 
NumEntries=countNZ;
Teuchos::Array<scalar_type> values_current(counter);
Teuchos::Array<scalar_type> values_X(counter);
Teuchos::Array<global_ordinal_type> indices_current(counter);
global_ordinal_type countN=0;
 
    for(int j=0;j<NumEntries;j++)
      {
       
        values_current[countN]=(1e-3)*abs(x_1d(myGlobalElements[i])-x_1d(mat_ind[j]))*mat_val[j];
        values_X[countN]=values_current[countN]*dt;
        indices_current[countN]=mat_ind[j];
        countN=countN+1;

       

       }
Current->resumeFill();
Current->replaceGlobalValues (myGlobalElements[i], indices_current, values_current);
Vol_frac->resumeFill();
Vol_frac->sumIntoGlobalValues(myGlobalElements[i], indices_current, values_X);
//std::cout<<"wprks till <m";
}

else
{
 Teuchos::Array<global_ordinal_type> mat_ind(counter);
Teuchos::Array<scalar_type> mat_val(counter);
 Teuchos::Array<global_ordinal_type> len_ind(4);
Teuchos::Array<scalar_type> len_val(4);
 Teuchos::Array<global_ordinal_type> duplicate_ind(4);
Teuchos::Array<scalar_type> duplicate_val(4);
K->getGlobalRowCopy(myGlobalElements[i],mat_ind,mat_val,num);
L->getGlobalRowCopy(myGlobalElements[i],len_ind,len_val,num);
size_t tr1;
size_t tr2=-1;
for (int jj=0;jj<4;jj++)
 {
    if(N[jj]>=0 && N[jj]+1<myGlobalElements[i])
   {
       tr1=N[jj];

   }

 }
 for (int jj=0;jj<4;jj++)
 {
    if(N[jj]>=0 && N[jj]+1==myGlobalElements[i])
   {
       tr2=N[jj];

   }

 }

Vol_frac->getGlobalRowCopy(myGlobalElements[tr1],duplicate_ind,duplicate_val,num);
L->getGlobalRowCopy(myGlobalElements[tr1],len_ind,len_val,num);
     double val1;
     double l_val1;
     double val2;
     double l_val2;
     for (int jj=0;jj<4;jj++)
   {
         if (duplicate_ind[jj]==myGlobalElements[i])
              {
                   val1=duplicate_val[jj];
                   l_val1=len_val[jj];
               }

    }
if(tr2>=0)
{
Vol_frac->getGlobalRowCopy(myGlobalElements[tr2],duplicate_ind,duplicate_val,num);
L->getGlobalRowCopy(myGlobalElements[tr2],len_ind,len_val,num);   
     double val2;
     double l_val2;
     for (int jj=0;jj<4;jj++)
   {
         if (duplicate_ind[jj]==myGlobalElements[i])
              {
                   val2=duplicate_val[jj];
                   l_val2=len_val[jj];
               }

    }
}


size_t NumEntries;


     size_t countNZ=0;
    for(int j=0;j<counter;j++)
      {
        if(mat_val[j]>0)
       {
        countNZ=countNZ+1;

       }
      }
NumEntries=countNZ;

Teuchos::Array<scalar_type> values_current(counter);
Teuchos::Array<scalar_type> values_X(counter);
Teuchos::Array<global_ordinal_type> indices_current(counter);
double diagonal=0;
global_ordinal_type countN=0;
    for(int j=0;j<counter;j++)
      {
        


        values_current[countN]=(1e-3)*abs(x_1d(myGlobalElements[i])-x_1d(mat_ind[j]))*mat_val[j];
             if(N[j]==tr1)
          {
              values_X[countN]=val1;

         }
             else if(N[j]==tr2)
        {
              values_X[countN]=val2;

        }
             else if(val1>=l_val1||val2>=l_val2)
          {
        values_X[countN]=values_current[countN]*dt;
         }
             else
        {
            values_X[countN]=0;
     }
        indices_current[countN]=mat_ind[j];
          
       countN=countN+1;
      



     }

Current->resumeFill();
Current->replaceGlobalValues (myGlobalElements[i], indices_current, values_current);
Vol_frac->resumeFill();
Vol_frac->sumIntoGlobalValues(myGlobalElements[i], indices_current, values_X);
}



 }//If time not zero
//size_t num;
Teuchos::Array<global_ordinal_type> excess_indices(counter);
Teuchos::Array<scalar_type> excess_val(counter);
Teuchos::Array<global_ordinal_type> length_ind(counter);
Teuchos::Array<scalar_type> length_val(counter);
Vol_frac->getGlobalRowCopy(myGlobalElements[i],excess_indices,excess_val,num);
L->getGlobalRowCopy(myGlobalElements[i],length_ind,length_val,num);
   for(int j=0;j<counter;j++)
 {
          if(excess_val[j]>length_val[j])
            excess_val[j]=length_val[j];
 }
 
 Vol_frac->replaceGlobalValues(myGlobalElements[i],excess_indices,excess_val);


} //Don't touch this. Upper most ifNumNodes

}// Don't touch this . Upper most for




Vol_frac->fillComplete();




}//dont touch this


};
#endif
