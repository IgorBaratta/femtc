#include <benchmark/benchmark.h>
#include <mpi.h>

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter
{
public:
    NullReporter() {}
    virtual bool ReportContext(const Context &) { return true; }
    virtual void ReportRuns(const std::vector<Run> &) {}
    virtual void Finalize() {}
};

/// Reduce the time from all processes to the root process
/// @param[in] time the time to reduce
/// @return the reduced time
double reduce_time(double time)
{
    double global_time;
    MPI_Reduce(&time, &global_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    global_time /= comm_size;
    return global_time;
}