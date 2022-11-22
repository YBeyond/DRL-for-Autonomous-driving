from dataclasses import dataclass, field
import torch


@dataclass
class PolicyParam:
    seed: int = 42

    num_workers: int = 10
    num_episode: int = 100000
    batch_size: int = 2048
    minibatch_size: int = 128
    num_epoch: int = 10
    save_num_episode = 10

    gamma: float = 0.9
    lamda: float = 0.97
    loss_coeff_value: float = 0.5
    loss_coeff_entropy: float = 0.01
    clip: float = 0.2
    lr: float = 1e-4

    vf_clip_param: float = 60
    max_grad_norm: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = None

    advantage_norm: bool = False
    use_clipped_value_loss: bool = False
    lossvalue_norm: bool = False
