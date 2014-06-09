#include "catalyst_adapter.hpp"

#include <vtkCPDataDescription.h>
#include <stdio.h>

#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include "Box.hpp"
#include "box_utils.hpp"
#include "Parameters.hpp"

#include "mpi.h"

namespace {
  // We store the main Catalyst class object so that we can have a purely
  // functional interface for using Catalyst. This means we can be
  // lazy and skip passing these objects around through the other
  // functions.
  vtkCPProcessor* Processor = NULL;
}

namespace Catalyst
{
  void coprocess_internal(const double spacing[3], const Box& global_box, const Box& local_box,
                          std::vector<double> &vtkpointdata,
                          vtkCPDataDescription* dataDescription);

  void coprocess(const double spacing[3], const Box& global_box, const Box& local_box,
                 std::vector<double>& minifepointdata, int time_step,
                 double time, bool lastTimeStep)
  {
    int myproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

    vtkNew<vtkCPDataDescription> dataDescription;
    dataDescription->AddInput("input");
    dataDescription->SetTimeData(time, time_step);

    if(lastTimeStep == true)
      {
      // assume that we want to all the pipelines to execute if it
      // is the last time step.
      dataDescription->ForceOutputOn();
      }

    if(Processor->RequestDataDescription(dataDescription.GetPointer()) == 0)
      {
      return; // no co-processing to be done this time step.
      }

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

    // I can't figure out the ordering of the "external" dofs that miniFE
    // uses so I wimp out and just construct the global array and then
    // extract the parts I need locally.
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

    std::vector<double> vtkpointdata;
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

    Catalyst::coprocess_internal(spacing, global_box, local_box, vtkpointdata,
                                 dataDescription.GetPointer());
  }

  void initialize(miniFE::Parameters& params)
  {
    cout << "FEAdapter::Initialize called\n";
    if(Processor == NULL)
      {
      Processor = vtkCPProcessor::New();
      Processor->Initialize();
      }
    else
      {
      cout << "  Processor not Null, unexpected, but remove pipelines\n";
      Processor->RemoveAllPipelines();
      }
    for(std::vector<std::string>::const_iterator it=params.script_names.begin();
        it!=params.script_names.end();it++)
      {
      vtkNew<vtkCPPythonScriptPipeline> pipeline;
      pipeline->Initialize(it->c_str());
      Processor->AddPipeline(pipeline.GetPointer());
      }

    // hack so that I don't have to remember the script input
    if(params.script_names.empty())
      {
      vtkNew<vtkCPPythonScriptPipeline> pipeline;
      pipeline->Initialize("gridwriter.py");
      Processor->AddPipeline(pipeline.GetPointer());
      }
  }

  void finalize()
  {
    if(Processor)
      {
      Processor->Finalize();
      Processor->Delete();
      Processor = NULL;
      }
  }

  void coprocess_internal(const double spacing[3], const Box& global_box, const Box& local_box,
                          std::vector<double> &inCalcData, vtkCPDataDescription* dataDescription)
  {
    vtkNew<vtkImageData> grid;

    int extent[6] = {local_box[0][0], local_box[0][1], local_box[1][0],
                     local_box[1][1], local_box[2][0], local_box[2][1]};
    grid->SetExtent(extent);
    grid->SetSpacing(spacing[0], spacing[1], spacing[2]);
    grid->SetOrigin(0, 0, 0);


    dataDescription->GetInputDescriptionByName("input")->SetGrid(grid.GetPointer());

    vtkSmartPointer<vtkDoubleArray> myDataArray =
      vtkSmartPointer<vtkDoubleArray>::New();
    myDataArray->SetNumberOfComponents(1);
    myDataArray->SetName("myData");
    myDataArray->SetArray(&(inCalcData[0]), inCalcData.size(), 1);

    if(inCalcData.size() != grid->GetNumberOfPoints())
      {
      int myproc;
      MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
      cerr << myproc << " WRONG -- in data is too small " << inCalcData.size()
           << " but should be " << grid->GetNumberOfPoints() << endl;;
      }

    //add data value to grid
    grid->GetPointData()->AddArray(myDataArray);

    int wholeExtent[6] = {global_box[0][0],
                          global_box[0][1],
                          global_box[1][0],
                          global_box[1][1],
                          global_box[2][0],
                          global_box[2][1]};

    dataDescription->GetInputDescriptionByName("input")->SetWholeExtent(wholeExtent);
    Processor->CoProcess(dataDescription);
  }


// below I'm trying to use miniFE's external dof mapping information
// to figure out the field mapping between VTK's and miniFE's
// ordering. I wasn't able to figure it out though but may come
// back to it later.
  // void dump_mesh_state(const Box& global_box, const Box& local_box, std::vector<int>& external_index,
  //                      std::vector<int>& external_local_index,
  //                      std::vector<double>& xx, int time_step, double time, bool lastTimeStep)
  // {
  //   int myproc;
  //   MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  //   vtkNew<vtkCPDataDescription> dataDescription;
  //   dataDescription->AddInput("input");
  //   dataDescription->SetTimeData(time, time_step);

  //   if(lastTimeStep == true)
  //     {
  //     // assume that we want to all the pipelines to execute if it
  //     // is the last time step.
  //     dataDescription->ForceOutputOn();
  //     }

  //   if(Processor->RequestDataDescription(dataDescription.GetPointer()) == 0)
  //     {
  //     return; // no co-processing to be done this time step.
  //     }

  //   Box box;
  //   miniFE::copy_box(local_box, box);

  //   std::vector<double> calc_solns;
  //   std::vector<double> external_calc_solns;

  //   std::vector<size_t> local_indices;
  //   std::vector<size_t> external_indices;

  //   size_t num_local_indices = 1;
  //   for(int i=0;i<3;i++)
  //     {
  //     int local_nodes = local_box[i][1] - local_box[i][0]+1;
  //     if(local_box[i][1] != global_box[i][1])
  //       {
  //       local_nodes--;
  //       }
  //     num_local_indices *= local_nodes;
  //     }

  //   if(!myproc)
  //     {
  //     for(size_t i=9;i<18;i++)
  //       {
  //       cerr << "x[" << i << "]=" << xx[i] << " " << external_local_index[i-9] << endl;
  //       }
  //     }

  //   std::vector<double> vtkfield;
  //   size_t external_indices_counter = 0;
  //   size_t local_indices_counter = 0;
  //   for(int iz=local_box[2][0]; iz<=local_box[2][1]; ++iz)
  //     {
  //     bool isExternalZ = (iz==local_box[2][1] && local_box[2][1] != global_box[2][1]);
  //     for(int iy=local_box[1][0]; iy<=local_box[1][1]; ++iy)
  //       {
  //       bool isExternalY = (iy==local_box[1][1] && local_box[1][1] != global_box[1][1]);
  //       for(int ix=local_box[0][0]; ix<=local_box[0][1]; ++ix)
  //         {
  //         bool isExternalX = (ix==local_box[0][1] && local_box[0][1] != global_box[0][1]);
  //         if(isExternalX || isExternalY || isExternalZ)
  //           {
  //           if(xx.size() <= num_local_indices+external_indices_counter)
  //             {
  //             vtkfield.push_back(999);
  //             cerr << myproc << " wwwwwwwwwwwwwwwwwwwweird\n";
  //             }
  //           else
  //             {
  //             if(external_index[external_indices_counter] >= xx.size())
  //               {
  //               vtkfield.push_back(444);
  //               cerr << myproc << " wwwwwwwwwwwwwwwwwwwweird222222222222 "<< num_local_indices << " "
  //                    << external_index[external_indices_counter] << endl;
  //               }
  //             vtkfield.push_back(xx[external_index[external_indices_counter]]);
  //             cerr << "val is " << xx[external_index[external_indices_counter]] << " from index "
  //                  << external_index[external_indices_counter] << endl;
  //             }
  //           external_indices_counter++;
  //           }
  //         else
  //           {
  //           if(xx.size() <= local_indices_counter)
  //             {
  //             vtkfield.push_back(999);
  //             cerr << myproc << " wwwwwwwwwwwwwwwwwwwweird22222\n";
  //             }
  //           else
  //             {
  //             vtkfield.push_back(xx[local_indices_counter]);
  //             }
  //           local_indices_counter++;
  //           }
  //         }
  //       }
  //     }

  //   if(num_local_indices != local_indices_counter)
  //     {
  //     cerr << myproc << " something wrong here.....catalystadaptor "<< num_local_indices << " " <<  local_indices_counter << endl;
  //     }
  //   Catalyst::dump_mesh_state2(global_box, local_box, vtkfield, dataDescription.GetPointer());
  // }

}
