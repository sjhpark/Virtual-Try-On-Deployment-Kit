from pruning import dressUpInference
from options.test_options import TestOptions
import os
import json
import numpy as np
import matplotlib.pyplot as plt

opt = TestOptions().parse()
opt.job = "filter_pruning" # hyperparameter

if opt.job == "unstructured_pruning":
    opt.module = "AFWM_cond_FPN"
    opt.sparsity = 0.33

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

elif opt.job == "filter_pruning":
    fig, axs = plt.subplots(2, 2, figsize=(40, 30))
    """
    This experiment is only for the AFWM module inside PF-AFN.
    - Filter pruning for conv2d layers inside AFWM_image_feature_encoder and AFWM_cond_feature_encoder
    - We avoid downsampling layers in each residual block and the last layers in each residual block to avoid dimension mismatch during residual connection.
    - We also avoid any BatchNorm2d layer since it doesn't have any filters to prune."""
    layer_idx2filter = [3,7,13,17,23,27,33,37,43,47] # conv layers in AFWM_cond_feature_encoder
    layer_idx2filter += [53,57,63,67,73,77,83,87,93,97] # conv layers in AFWM_image_feature_encoder

    cmap = plt.get_cmap("tab20")
    for i, idx in enumerate(layer_idx2filter):
        opt.layer_idx = idx
        color = cmap(i)

        track_params = []
        track_disk_size = []
        track_inference = []
        track_MSE = []
        track_SSIM = []

        # for filter_idx in range(0, 64-1):
        for j, filter_idx in enumerate(range(0, 3-1)):
            opt.filter_idx = filter_idx

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
            axs[0,0].scatter(j, warp_params, color = color)
            axs[0,0].set_title(f"AFWM\nWarp Parameter Count vs. Pruned Filter Index")
            axs[0,0].set_xlabel("Pruned Filter Index")
            axs[0,0].set_ylabel("Parameter Count")

            track_inference.append(warp_inference)
            axs[0,1].scatter(j, warp_inference, color = color)
            axs[0,1].set_title(f"AFWM\nWarp Inference Time vs. Pruned Filter Index")
            axs[0,1].set_xlabel("Pruned Filter Index")
            axs[0,1].set_ylabel("Inference Time [s]")

            track_MSE.append(sum(MSE) / len(MSE))
            axs[1,0].scatter(j, sum(MSE) / len(MSE), color = color, marker = "*", s=100)
            axs[1,0].set_title(f"AFWM\nMean MSE vs. Pruned Filter Index")
            axs[1,0].set_xlabel("Pruned Filter Index")
            axs[1,0].set_ylabel("Mean MSE")

            track_SSIM.append(sum(SSIM) / len(SSIM))
            axs[1,1].scatter(j, sum(SSIM) / len(SSIM), color = color, marker = "*", s=100)
            axs[1,1].set_title(f"AFWM\nMean SSIM vs. Pruned Filter Index")
            axs[1,1].set_xlabel("Pruned Filter Index")
            axs[1,1].set_ylabel("Mean SSIM")

        axs[0,0].plot(track_params, color=color, label=f"Conv {idx}")
        axs[0,1].plot(track_inference, color=color, label=f"Conv {idx}")
        axs[1,0].plot(track_MSE, color=color, label=f"Conv {idx}")
        axs[1,1].plot(track_SSIM, color=color, label=f"Conv {idx}")

    axs[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axs[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axs[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axs[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(os.path.join(save_dir, f"AFWM_filter_pruning_experiment.png"))

