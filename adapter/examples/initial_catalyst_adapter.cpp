#include "catalyst_adapter.hpp"

// VTK/ParaView header files
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkMultiProcessController.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

// miniFE header files
#include "Box.hpp"
#include "Parameters.hpp"

#include <mpi.h>

namespace {
  // We store the main Catalyst class object so that we can have a purely
  // functional interface for using Catalyst. This means we can be
  // lazy and skip passing these objects around through the other
  // functions.
  vtkCPProcessor* Processor = NULL;
}

namespace Catalyst
{
  void getlocalpointarray(const Box& global_box, const Box& local_box,
                          std::vector<double>& minifepointdata,
                          std::vector<double>& vtkpointdata);

  void initialize(miniFE::Parameters& params)
  {
    cout << "catalyst_adapter.cpp: initializing Catalyst\n";
  }

  void coprocess(const double spacing[3], const Box& global_box, const Box& local_box,
                 std::vector<double>& minifepointdata, int time_step,
                 double time, bool force_output)
  {
    cout << "catalyst_adapter.cpp: checking for co-processing in Catalyst\n";

    // vtkpointdata is the point data array that stores the information in the
    // same order as we expect for our VTK ordering of the grid. We compute
    // it in getlocalpointarray();
    std::vector<double> vtkpointdata;
    getlocalpointarray(global_box, local_box, minifepointdata, vtkpointdata);

 }

  void finalize()
  {
    cout << "catalyst_adapter.cpp: finalizing Catalyst\n";
  }

  void getlocalpointarray(const Box& global_box, const Box& local_box,
                          std::vector<double>& minifepointdata,
                          std::vector<double>& vtkpointdata)
  {
    // I can't figure out the ordering of the "external" dofs that miniFE
    // uses so I wimp out and just construct the global array and then
    // extract the parts I need locally. Simulation code developers
    // will be knowledgeable enough about their own data structures
    // to do this properly but probably don't know enough about miniFE
    // to do it here so we do it for them. The other thing to note is
    // that miniFE does a node partitioning of the grid so minifepointdata
    // only stores the results at the nodes that each process "owns".
    // Because of this we would have to get the "external" dofs anyways
    // but would really use exchange_externals() to do it more efficiently.
    int num_local_indices = 1;
    int num_global_indices = 1;
    for(int i=0;i<3;i++)
      {
      int local_nodes = local_box[i][1] - local_box[i][0]+1;
      if(local_box[i][1] != global_box[i][1])
        {
        local_nodes--;
        }
      num_local_indices *= local_nodes;
      num_global_indices *= global_box[i][1] - global_box[i][0]+1;
      }

    std::vector<double> tmparray(num_global_indices, VTK_DOUBLE_MIN);
    std::vector<double> tmparray2(num_global_indices);

    int counter = 0;
    for(int iz=local_box[2][0]; iz<=local_box[2][1]; ++iz)
      {
      if(iz < local_box[2][1]  || local_box[2][1] == global_box[2][1])
        {
        for(int iy=local_box[1][0]; iy<=local_box[1][1]; ++iy)
          {
          if(iy < local_box[1][1]  || local_box[1][1] == global_box[1][1])
            {
            for(int ix=local_box[0][0]; ix<=local_box[0][1]; ++ix)
              {
              if(ix < local_box[0][1]  || local_box[0][1] == global_box[0][1])
                {
                int id = ix+(global_box[0][1]+1)*iy+(global_box[0][1]+1)*(global_box[1][1]+1)*iz;
                tmparray[id] = minifepointdata[counter];
                counter++;
                }
              }
            }
          }
        }
      }

    MPI_Allreduce(&(tmparray[0]), &(tmparray2[0]), num_global_indices,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // vtkpointdata is the point data array that stores the information in the
    // same order as we expect for our VTK ordering of the grid.
    for(int iz=local_box[2][0]; iz<=local_box[2][1]; ++iz)
      {
      for(int iy=local_box[1][0]; iy<=local_box[1][1]; ++iy)
        {
        for(int ix=local_box[0][0]; ix<=local_box[0][1]; ++ix)
          {
          int id = ix+(global_box[0][1]+1)*iy+(global_box[0][1]+1)*(global_box[1][1]+1)*iz;
          vtkpointdata.push_back(tmparray2[id]);
          }
        }
      }
    // We're done creating the local point data array. Now we move on to creating
    // VTK objects.
  }
}
