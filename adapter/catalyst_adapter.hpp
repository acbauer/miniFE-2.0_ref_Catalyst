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
void coprocess(const double spacing[3], const Box& global_box, const Box& local_box,
               std::vector<double>& minifepointdata, int time_step,
               double time, bool lastTimeStep);

void initialize(miniFE::Parameters& params);
void finalize();
}

#endif
