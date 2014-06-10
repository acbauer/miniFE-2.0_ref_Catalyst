#ifndef FEADAPTOR_HEADER
#define FEADAPTOR_HEADER

#include <vector>

namespace miniFE
{
class Parameters;
}
struct Box;

namespace Catalyst
{
void initialize(miniFE::Parameters& params);
void coprocess(const double spacing[3], const Box& global_box, const Box& local_box,
               std::vector<double>& minifepointdata, int time_step,
               double time, bool force_output);
void finalize();
}

#endif
