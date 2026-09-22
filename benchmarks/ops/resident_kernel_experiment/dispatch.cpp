// SPDX-License-Identifier: Apache-2.0
#include "launch.h"

void resident_experiment_baseline_union(void*, const ResidentLaunch&);
void resident_experiment_baseline_finalize(void*, const ResidentLaunch&);
void resident_experiment_baseline_update(void*, const ResidentLaunch&);
void resident_experiment_optimized_union(void*, const ResidentLaunch&);
void resident_experiment_optimized_finalize(void*, const ResidentLaunch&);
void resident_experiment_optimized_update(void*, const ResidentLaunch&);

void resident_experiment_run(void* stream, const ResidentLaunch& args, int stage)
{
    if (stage == 0 || stage == 1) resident_experiment_baseline_union(stream, args);
    if (stage == 0 || stage == 2) resident_experiment_baseline_finalize(stream, args);
    if (stage == 0 || stage == 3) resident_experiment_baseline_update(stream, args);
}

void resident_experiment_run_optimized(void* stream, const ResidentLaunch& args, int stage)
{
    if (stage == 0 || stage == 1) resident_experiment_optimized_union(stream, args);
    if (stage == 0 || stage == 2) resident_experiment_optimized_finalize(stream, args);
    if (stage == 0 || stage == 3) resident_experiment_optimized_update(stream, args);
}
