// SPDX-License-Identifier: Apache-2.0
#include "launch.h"

void resident_experiment_baseline_union(void*, const ResidentLaunch&);
void resident_experiment_baseline_finalize(void*, const ResidentLaunch&);
void resident_experiment_baseline_update(void*, const ResidentLaunch&);
void resident_experiment_optimized_union(void*, const ResidentLaunch&);
void resident_experiment_optimized_finalize(void*, const ResidentLaunch&);
void resident_experiment_optimized_update(void*, const ResidentLaunch&);
void resident_experiment_compact_update(void*, const ResidentLaunch&);
void resident_experiment_sharded_finalize(void*, const ResidentLaunch&);
void resident_experiment_vector_union(void*, const ResidentLaunch&);
void resident_experiment_baseline_union_sort(void*, const ResidentLaunch&);
void resident_experiment_baseline_union_dedup(void*, const ResidentLaunch&);
void resident_experiment_vector_union_sort(void*, const ResidentLaunch&);
void resident_experiment_vector_union_dedup(void*, const ResidentLaunch&);

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

void resident_experiment_run_variant(void* stream, const ResidentLaunch& args, int variant, int stage)
{
    if (stage == 4) {
        if (variant == 5) resident_experiment_vector_union_sort(stream, args);
        else resident_experiment_baseline_union_sort(stream, args);
        return;
    }
    if (stage == 5) {
        if (variant == 5) resident_experiment_vector_union_dedup(stream, args);
        else resident_experiment_baseline_union_dedup(stream, args);
        return;
    }
    if (variant == 5) {
        if (stage == 0 || stage == 1) resident_experiment_vector_union(stream, args);
        if (stage == 0 || stage == 2) resident_experiment_baseline_finalize(stream, args);
        if (stage == 0 || stage == 3) resident_experiment_baseline_update(stream, args);
        return;
    }
    if (variant == 0) { resident_experiment_run(stream, args, stage); return; }
    if (variant == 1) { resident_experiment_run_optimized(stream, args, stage); return; }
    // 2: compact remap only; 3: sharded finalize only; 4: both. Neither new
    // variant enables the earlier unchanged-state fast path.
    if (stage == 0 || stage == 1) resident_experiment_baseline_union(stream, args);
    if (stage == 0 || stage == 2) {
        if (variant >= 3) resident_experiment_sharded_finalize(stream, args);
        else resident_experiment_baseline_finalize(stream, args);
    }
    if (stage == 0 || stage == 3) {
        if (variant == 2 || variant == 4) resident_experiment_compact_update(stream, args);
        else resident_experiment_baseline_update(stream, args);
    }
}
