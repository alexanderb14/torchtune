import os

EXP_OUT_DIR = "exp_outs"
OUT_FILE = "torchtune_stats.csv"

def get_batch_shapes(log):
    return [line for line in log.split("\n") if line.startswith("{'tokens': ")]

def check_experiment_conditions(log_default, log_reduce_overhead):
    # (1) Check that the batch shapes are the same between the two experiments
    shapes_default = get_batch_shapes(log_default)
    shapes_reduce_overhead = get_batch_shapes(log_reduce_overhead)
    assert (
        shapes_default == shapes_reduce_overhead
    ), "Batch are different between the two experiments"

    # (2) Check that no CUDAGraph is recorded after warmup in the reduce-overhead experiment
    after_warmup = log_reduce_overhead.split("Warmup complete")[1]
    assert (
        "CUDAGraphTreeManager::record_function" not in after_warmup
    ), "CUDAGraph is recorded after warmup"


def main():
    # Read the logs
    with open(os.path.join(EXP_OUT_DIR, "inductor-default.log")) as f:
        log_default = f.read()
    with open(os.path.join(EXP_OUT_DIR, "inductor-reduce-overhead.log")) as f:
        log_reduce_overhead = f.read()

    # Check that the experiment conditions are valid
    check_experiment_conditions(log_default, log_reduce_overhead)

    # Create csv header
    with open(OUT_FILE, "w") as f:
        f.write("batch_idx,shape,mode,forward_time,backward_time\n")

    # Get times
    def get_times(log, time_type):
        return [line for line in log.split("\n") if "%s time of batch " % time_type in line]

    fwd_times_default = get_times(log_default, "Forward")
    fwd_times_reduce_overhead = get_times(log_reduce_overhead, "Forward")

    bck_times_default = get_times(log_default, "Backward")
    bck_times_reduce_overhead = get_times(log_reduce_overhead, "Backward")

    assert len(fwd_times_default) == len(
        fwd_times_reduce_overhead
    ), "Number of loss times are different between the two experiments"

    shapes = get_batch_shapes(log_default)
    shapes = [shape.split(", ")[1].split("]")[0] for shape in shapes]

    # Write to csv
    for batch_idx, (fwd_time_default, fwd_time_reduce_overhead, bck_times_default, bck_times_reduce_overhead, shape) in enumerate(
        zip(fwd_times_default, fwd_times_reduce_overhead, bck_times_default, bck_times_reduce_overhead, shapes)
    ):
        # Parse times
        fwd_time_default = float(
            fwd_time_default.split("Forward time of batch %d: " % batch_idx)[1]
        )
        fwd_time_reduce_overhead = float(
            fwd_time_reduce_overhead.split("Forward time of batch %d: " % batch_idx)[1]
        )

        bck_time_default = float(
            bck_times_default.split("Backward time of batch %d: " % batch_idx)[1]
        )
        bck_time_reduce_overhead = float(
            bck_times_reduce_overhead.split("Backward time of batch %d: " % batch_idx)[1]
        )

        # Write to csv
        with open(OUT_FILE, "a") as f:
            f.write(f"{batch_idx},{shape},default,{fwd_time_default},{bck_time_default}\n")
            f.write(f"{batch_idx},{shape},reduce-overhead,{fwd_time_reduce_overhead},{bck_time_reduce_overhead}\n")


if __name__ == "__main__":
    main()
