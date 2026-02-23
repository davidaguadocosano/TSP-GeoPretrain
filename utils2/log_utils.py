import os
import json
import torch
import argparse
from tensorboard_logger import Logger as TbLogger


def log_values(
        loss: torch.Tensor,
        grad_norms: tuple,
        epoch: int,
        batch_id: int,
        step: int,
        tb_logger: TbLogger,
        reward: torch.Tensor = None,
        loss_bl: torch.Tensor = None,
        use_critic: bool = False,
        debug: bool = False) -> None:
    """
    Log values during training to the console and optionally to TensorBoard.

    Args:
        loss (Tensor): Loss values.
        grad_norms (tuple): Tuple containing gradient norms before and after clipping.
        epoch (int): Current epoch number.
        batch_id (int): Batch ID.
        step (int): Current step number.
        tb_logger (TbLogger): TensorBoard logger instance.
        reward (Tensor): Reward values (for DRL).
        loss_bl (Tensor): Baseline loss values.
        use_critic (bool): True if critic baseline is active.
        debug (bool): True if debug mode is active.

    Returns:
        None
    """

    # Get average loss
    loss = loss.mean().item()

    # Get gradients
    grad_norms, clipped = grad_norms

    # Log values to screen
    r = "" if reward is None else f", reward: {reward.mean().item():.4f}"
    print(f"  Epoch: {epoch}, batch_id: {batch_id}, loss: {loss:.4f}, grad_norm: {grad_norms[0]:.4f}, clipped: {clipped[0]:.4f}{r}")

    # Log values to tensorboard
    if not debug:
        tb_logger.log_value('train_loss', loss, step)
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', clipped[0], step)

        # Log critic-related values to tensorboard
        if use_critic:
            tb_logger.log_value('critic_loss', loss_bl.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', clipped[1], step)


def config_logger(opts: argparse.Namespace) -> TbLogger | None:
    """
    Prints and saves arguments into a json file. Optionally sets up TensorBoard logging.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        TbLogger or None: TensorBoard logger instance if enabled, otherwise None.
    """
    tb_logger = None
    
    # If not in debug mode
    if not opts.debug:

        # Save arguments so exact configuration can always be found
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

        # Optionally configure tensorboard
        tb_dir = os.path.join(opts.save_dir, 'log_dir')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        tb_logger = TbLogger(tb_dir)

    # Print opts
    for k, v in vars(opts).items():
        print("'{}': {}".format(k, v))
    print()
    return tb_logger
