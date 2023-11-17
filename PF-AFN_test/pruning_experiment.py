from pruning import dressUpInference
from options.test_options import TestOptions
import os
import json
import matplotlib.pyplot as plt

opt = TestOptions().parse()
opt.module = "AFWM_cond_FPN"
opt.sparsity = 0.9

track_params = []
track_disk_size = []
track_inference = []
track_MSE = []
track_SSIM = []

fig, axs = plt.subplots(3, 2, figsize=(20, 30))
for i in range(0, 10-1):
    opt.layer_idx = i

    # pruning and run inference
    obj = dressUpInference(opt)
    obj.model_statistics() # param count & model size on disk
    obj.measure_inference_time(warmup_itr=10) # measure inference time
    obj.get_statistics(img_num=5) # generate one image and compute accuracy (MSE, SSIM); img_num = either 1, 2, 3, 4, 5 (total 5 groundtruth images)

    # statistics
    save_dir = "out"
    data = []
    with open(os.path.join(save_dir, "pruning_results.json"), "r") as f:
        for line in f:
            data.append(json.loads(line))
    warp_params = data[1]["warp_params"]
    gen_params = data[1]["gen_params"]
    warp_disk_size = data[2]["warp_disk_size"]
    gen_disk_size = data[2]["gen_disk_size"]
    warp_inference = data[3]["warp_inference"]
    gen_inference = data[4]["gen_inference"]
    MSE = data[5]["MSE"]
    SSIM = data[5]["SSIM"]

    # plot each metric in a separate subplot
    track_params.append(warp_params)
    axs[0,0].scatter(i, warp_params, color = "blue")
    axs[0,0].set_title(f"{opt.module}, Sparsity: {opt.sparsity}\nWarp Parameter Count vs. Pruned Layer Index")
    axs[0,0].set_xlabel("Pruned Layer Index")
    axs[0,0].set_ylabel("Parameter Count")

    track_disk_size.append(warp_disk_size)
    axs[0,1].scatter(i, warp_disk_size, color = "blue")
    axs[0,1].set_title(f"{opt.module}, Sparsity: {opt.sparsity}\nWarp Model Size on Disk vs. Pruned Layer Index")
    axs[0,1].set_xlabel("Pruned Layer Index")
    axs[0,1].set_ylabel("Model Size on Disk [MB]")

    track_inference.append(warp_inference)
    axs[1,0].scatter(i, warp_inference, color = "blue")
    axs[1,0].set_title(f"{opt.module}, Sparsity: {opt.sparsity}\nWarp Inference Time vs. Pruned Layer Index")
    axs[1,0].set_xlabel("Pruned Layer Index")
    axs[1,0].set_ylabel("Inference Time [s]")

    track_MSE.append(sum(MSE) / len(MSE))
    axs[1,1].scatter(i, sum(MSE) / len(MSE), color = "blue", marker = "*", s=100)
    axs[1,1].set_title(f"{opt.module}, Sparsity: {opt.sparsity}\nMSE vs. Pruned Layer Index")
    axs[1,1].set_xlabel("Pruned Layer Index")
    axs[1,1].set_ylabel("MSE")

    track_SSIM.append(sum(SSIM) / len(SSIM))
    axs[2,0].scatter(i, sum(SSIM) / len(SSIM), color = "blue", marker = "*", s=100)
    axs[2,0].set_title(f"{opt.module}, Sparsity: {opt.sparsity}\nSSIM vs. Pruned Layer Index")
    axs[2,0].set_xlabel("Pruned Layer Index")
    axs[2,0].set_ylabel("SSIM")

# Save the final plot
axs[0,0].plot(track_params, color = "blue")
axs[0,1].plot(track_disk_size, color = "blue")
axs[1,0].plot(track_inference, color = "blue")
axs[1,1].plot(track_MSE, color = "blue")
axs[2,0].plot(track_SSIM, color = "blue")
plt.savefig(os.path.join(save_dir, f"{opt.module}_{opt.sparsity}_pruning_experiment.png"))