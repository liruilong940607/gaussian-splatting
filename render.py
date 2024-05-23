#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import json
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    stats_path = os.path.join(model_path, name, "ours_{}.json".format(iteration))

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ellipse_time = 0
    metrics = {"psnr": [], "ssim": [], "lpips": []}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        tic = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize()
        ellipse_time += time.time() - tic
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        rendering = torch.clamp(rendering, min=0, max=1)
        metrics["psnr"].append(psnr(rendering[None], gt[None]))
        metrics["ssim"].append(ssim(rendering[None], gt[None]))
        metrics["lpips"].append(lpips(rendering[None], gt[None]))
    
    ellipse_time = ellipse_time / len(views)

    psnr = torch.stack(metrics["psnr"]).mean()
    ssim = torch.stack(metrics["ssim"]).mean()
    lpips = torch.stack(metrics["lpips"]).mean()

    stats = {
        "psnr": psnr.item(),
        "ssim": ssim.item(),
        "lpips": lpips.item(),
        "ellipse_time": ellipse_time,
        "num_GS": len(gaussians.get_xyz),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)