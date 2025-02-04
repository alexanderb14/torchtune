import os

exp_out_dir = "exp_outs_10"
out_file = "torchtune_training_stats.csv"

num_batches = 10

# Create csv header
with open(out_file, "w") as f:
    f.write("mode,padding_multiple,warmup_time,loss_time,opt_time\n")

for filename in os.listdir(exp_out_dir):
    # mode
    if "cudagraphs" in filename:
        mode = "reduce-overhead"
    elif "default" in filename and "disabled_padding" not in filename:
        mode = "default"
    elif "disabled_padding" in filename:
        mode = "default-disabled_padding"
    else:
        raise ValueError("Unknown mode")

    # warmup_time, loss_time, opt_time
    with open(os.path.join(exp_out_dir, filename)) as f:
        content = f.read()
        warmup_time = float(content.split("Overall Warmup time:  ")[1].split("\n")[0])
        loss_time = float(content.split("Overall Loss time:  ")[1].split("\n")[0])
        opt_time = float(content.split("Overall Opt time:  ")[1].split("\n")[0])
    warmup_time /= num_batches
    loss_time /= num_batches
    opt_time /= num_batches

    # # padding_multiple
    if "pad" in filename and not "disabled_padding" in filename:
        padding_multiple = int(filename.split("_pad")[1].split(".")[0])
    else:
        padding_multiple = 0


    with open(out_file, "a") as f:
        f.write("{},{},{},{},{}\n".format(mode, padding_multiple, warmup_time, loss_time, opt_time))
