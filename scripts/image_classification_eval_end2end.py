"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from tqdm import tqdm
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import sys
sys.path.append('/workspace/Projects/InNOutRobustnessMean0_cifar100/EGC')

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    egc_model_and_diffusion_defaults,
    create_egc_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import get_val_ldm_data
from pgd_attack import pgd_attack_xent_egc, pgd_attack_xent_egc_end2end
from autoencoder import FrozenAutoencoderKL

ddconfig_f8 = dict(
    double_z=True,
    z_channels=4,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0
)

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_egc_model_and_diffusion(
        **args_to_dict(args, egc_model_and_diffusion_defaults().keys())
    )
    
    # Estimate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # encoder = FrozenAutoencoderKL(ddconfig_f8, 4, "./autoencoder_kl.pth", 0.18215).cuda()
    encoder = FrozenAutoencoderKL(ddconfig_f8, 4, "./autoencoder_kl.pth", 0.18215).to(dist_util.dev())
    
    val_data = get_val_ldm_data(
        data_dir=args.val_data_dir,
        img_num=50000,
        batch_size=args.val_batch_size,
        class_cond=True,
        double_z=False,
        sample_z=False,
        scale_factor=1,
    ) # Set double_z=False, sample_z=False, and scale_factor=1 because we are only loading raw image data (sample_z is False by default, scale_factor is 0.18215 by default)

    # test_corrects = []
    # total_samples = 0
    # for image, label in tqdm(val_data):
    #     image = image.to(dist_util.dev())
    #     label = label['y'].to(dist_util.dev())
    #     with th.no_grad():
    #         image = encoder(image * 2. - 1, fn='encode_moments')
    #         image, _ = th.chunk(image, 2, dim=1)
    #         image = image * 0.18215
    #         time = th.zeros([image.shape[0]], dtype=th.long, device=dist_util.dev())
    #         pred = model(image, time, cls_mode=True)
    #         correct = (pred.max(1)[1] == label).float()
    #         test_corrects.append(correct)
    #         
    #         total_samples += correct.shape[0]
    #         running_acc = float(th.cat(test_corrects).mean() * 100)
    #         print(f"Clean running acc: {running_acc:.2f}% ({total_samples} samples)")
    # 
    # test_corrects = th.cat(test_corrects)
    # test_acc = float(test_corrects.mean() * 100)
    # 
    # logger.log(f"Test Acc: {test_acc}")
    
    # Adversarial evaluation
    logger.log("Starting adversarial evaluation...")
    val_data = get_val_ldm_data(
        data_dir=args.val_data_dir,
        img_num=50000,
        batch_size=args.val_batch_size,
        class_cond=True,
        double_z=False,
        sample_z=False,
        scale_factor=1,
    )
    
    adv_corrects = []
    total_adv_samples = 0
    for image, label in tqdm(val_data):
        image = image.to(dist_util.dev())
        label = label['y'].to(dist_util.dev())
        
        # Generate adversarial examples (end-to-end)
        model.eval()
        encoder.eval()
        adv_image = pgd_attack_xent_egc_end2end(
            model=model,
            encoder=encoder,
            x=image,
            true_labels=label,
            norm="L2",
            eps=3.0,  # Standard ImageNet perturbation
            step_size=1.0,
            steps=5,
            random_start=False
        )
        
        with th.no_grad():
            # Encode the adversarial image
            encoded_adv = encoder(adv_image * 2. - 1, fn='encode_moments')
            encoded_adv, _ = th.chunk(encoded_adv, 2, dim=1)
            encoded_adv = encoded_adv * 0.18215
            time = th.zeros([encoded_adv.shape[0]], dtype=th.long, device=dist_util.dev())
            pred = model(encoded_adv, time, cls_mode=True)
            correct = (pred.max(1)[1] == label).float()
            adv_corrects.append(correct)
            
            total_adv_samples += correct.shape[0]
            running_adv_acc = float(th.cat(adv_corrects).mean() * 100)
            print(f"Adversarial running acc: {running_adv_acc:.2f}% ({total_adv_samples} samples)")
    
    adv_corrects = th.cat(adv_corrects)
    adv_acc = float(adv_corrects.mean() * 100)
    
    logger.log(f"Adversarial Acc (L2 eps=3.0): {adv_acc}")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        batch_size=16,
        use_ddim=False,
        model_path="",
        autoencoder_path="",
        local_rank=0,
        autoencoder_type = 'KL',
        autoencoder_stride='8',
        sample_z=False,
        double_z=False,
        scale_factor=1,
        val_data_dir='',
        val_batch_size=16,
    )
    defaults.update(egc_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
