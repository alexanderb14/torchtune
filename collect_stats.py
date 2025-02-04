import os

EXP_OUT_DIR = "exp_outs"
OUT_FILE = "torchtune_stats.csv"


def check_experiment_conditions(log_default, log_reduce_overhead):
    def get_batch_shapes(log):
        return [line for line in log.split("\n") if line.startswith("{'tokens': ")]

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
        f.write("mode,loss_time\n")

    # Get loss times
    def get_loss_times(log):
        return [line for line in log.split("\n") if "Loss time of batch " in line]

    loss_times_default = get_loss_times(log_default)
    loss_times_reduce_overhead = get_loss_times(log_reduce_overhead)
    assert len(loss_times_default) == len(
        loss_times_reduce_overhead
    ), "Number of loss times are different between the two experiments"

    for idx, (loss_time_default, loss_time_reduce_overhead) in enumerate(
        zip(loss_times_default, loss_times_reduce_overhead)
    ):
        # Parse loss times
        loss_time_default = float(
            loss_time_default.split("Loss time of batch %d: " % idx)[1]
        )
        loss_time_reduce_overhead = float(
            loss_time_reduce_overhead.split("Loss time of batch %d: " % idx)[1]
        )

        # Write to csv
        with open(OUT_FILE, "a") as f:
            f.write(f"default,{loss_time_default}\n")
            f.write(f"reduce-overhead,{loss_time_reduce_overhead}\n")


if __name__ == "__main__":
    main()
