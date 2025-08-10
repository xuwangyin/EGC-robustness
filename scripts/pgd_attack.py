import torch
import torch.nn.functional as F
from torch import nn
from attack_steps import L2Step, LinfStep

def pgd_attack_xent(
    model: nn.Module,
    x: torch.Tensor,
    true_labels: torch.LongTensor,
    norm: str,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Perform untargeted PGD attack using cross-entropy loss.

    Args:
        model: Model to attack
        x: Clean inputs
        true_labels: True labels for the inputs
        norm: Norm to use for the attack ("L2" or "Linf")
        eps: Maximum perturbation size
        step_size: Step size for each iteration
        steps: Number of attack iterations
        random_start: Whether to start with random perturbation

    Returns:
        Adversarial examples that maximize the loss for the true labels
    """
    # Standard P0+r\P0+r\adversarial training uses pgd attack with model in training mode
    # assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    x0 = x.clone().detach()
    step_class = L2Step if norm == "L2" else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x, y=None)

        # Maximize loss for true label (minimize negative loss)
        loss = F.cross_entropy(logits, true_labels)

        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x.clone().detach()


def pgd_attack_xent_egc(
    model: nn.Module,
    x: torch.Tensor,
    true_labels: torch.LongTensor,
    norm: str,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Perform untargeted PGD attack using cross-entropy loss for EGC models.
    
    Args:
        model: EGC model to attack
        x: Clean inputs (encoded features)
        true_labels: True labels for the inputs
        norm: Norm to use for the attack ("L2" or "Linf")
        eps: Maximum perturbation size
        step_size: Step size for each iteration
        steps: Number of attack iterations
        random_start: Whether to start with random perturbation
        
    Returns:
        Adversarial examples that maximize the loss for the true labels
    """
    assert not x.requires_grad
    
    if steps == 0:
        return x.clone()
    
    x0 = x.clone().detach()
    step_class = L2Step if norm == "L2" else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)
    
    if random_start:
        x = step.random_perturb(x)
    
    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        
        # Create time tensor (zeros for classification mode)
        time = torch.zeros([x.shape[0]], dtype=torch.long, device=x.device)
        
        # Forward pass through EGC model in classification mode
        logits = model(x, time, cls_mode=True)
        
        # Maximize loss for true label (minimize negative loss)
        loss = F.cross_entropy(logits, true_labels)
        
        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    
    return x.clone().detach()


def pgd_attack_xent_egc_end2end(
    model: nn.Module,
    encoder: nn.Module,
    x: torch.Tensor,
    true_labels: torch.LongTensor,
    norm: str,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Perform untargeted PGD attack using cross-entropy loss for EGC models (end-to-end).
    
    Args:
        model: EGC model to attack
        encoder: Autoencoder to encode raw images
        x: Clean raw image inputs (0-1 range)
        true_labels: True labels for the inputs
        norm: Norm to use for the attack ("L2" or "Linf")
        eps: Maximum perturbation size (in image space)
        step_size: Step size for each iteration
        steps: Number of attack iterations
        random_start: Whether to start with random perturbation
        
    Returns:
        Adversarial raw images that maximize the loss for the true labels
    """
    assert not x.requires_grad
    
    if steps == 0:
        return x.clone()
    
    x0 = x.clone().detach()
    step_class = L2Step if norm == "L2" else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)
    
    if random_start:
        x = step.random_perturb(x)
    
    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        
        # https://github.com/GuoQiushan/EGC/blob/ec939ae503edece6f41af3379a694323a175d623/scripts/extract_feat.py#L45
        # Encode the perturbed image (scale to [-1, 1] for autoencoder)
        encoded = encoder(x * 2.0 - 1.0, fn='encode_moments')

        # https://github.com/GuoQiushan/EGC/blob/ec939ae503edece6f41af3379a694323a175d623/guided_diffusion/image_datasets.py#L42
        encoded, _ = torch.chunk(encoded, 2, dim=1)
        # https://github.com/GuoQiushan/EGC/blob/ec939ae503edece6f41af3379a694323a175d623/guided_diffusion/image_datasets.py#L52
        encoded = encoded * 0.18215
        
        # https://github.com/GuoQiushan/EGC/blob/ec939ae503edece6f41af3379a694323a175d623/scripts/image_classification_eval.py#L57
        # Create time tensor (zeros for classification mode)
        time = torch.zeros([encoded.shape[0]], dtype=torch.long, device=x.device)
        
        # Forward pass through EGC model in classification mode
        logits = model(encoded, time, cls_mode=True)
        
        # Maximize loss for true label (minimize negative loss)
        loss = F.cross_entropy(logits, true_labels)
        
        # Get gradient w.r.t. input image
        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )
        
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    
    return x.clone().detach()
