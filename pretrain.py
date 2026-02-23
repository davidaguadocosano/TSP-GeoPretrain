import time
from tqdm import tqdm

from envs import get_generator, envs, scenarios
from nets import load_model, models
from utils2 import *

GRAPH_ENCODERS = list(models()['encoders']['graph_encoders'].keys())
IMAGE_ENCODERS = list(models()['encoders']['image_encoders'].keys())
DECODERS = models()['decoders']
ENVS = list(envs().keys())
SCENARIOS = scenarios()
BASELINES = list(bls().keys())


def get_options(args: list = None) -> argparse.Namespace:
    """
    Parse command line arguments to configure training options for the neural network model.

    Args:
        args (list): List of command line arguments. If None, defaults to sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="Random seed to use")
    
    ##################################################### MODEL #####################################################
    
    # Graph encoder
    parser.add_argument('--graph_encoder', type=str, default='gtn', help=f"Graph encoders: {', '.join(GRAPH_ENCODERS)}. You may indicate a path to load instead")
    
    # Image encoder
    parser.add_argument('--image_encoder', type=str, default=None, help=f"Image encoders: {', '.join(IMAGE_ENCODERS)}. You may indicate a path to load instead") #dac
    parser.add_argument('--image_size', type=int, default=64, help=f"Binary map shape (assume squared image)")
    parser.add_argument('--patch_size', type=int, default=16, help=f"Patch size for ViT encoder")
    
    # Decoder
    parser.add_argument('--decoder', type=str, default='tsp-ar', help=f"Decoders: {', '.join(DECODERS)}. You may indicate a path to load instead")
    parser.add_argument('--num_dirs', type=int, default=8, help=f"Number of directions the agent can turn")
    
    # Parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help="Dimension of embeddings")
    parser.add_argument('--hidden_dim_ff', type=int, default=512, help="Dimension of embeddings of transformer's feed forward layers")
    parser.add_argument('--dropout', type=int, default=0.1, help="Apply dropout")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of multi-heads for attention operations")
    parser.add_argument('--num_blocks', type=int, default=3, help="Number of blocks in the encoder/critic network")
    parser.add_argument('--freeze_encoder', type=str2bool, default=False, help="Whether to freeze encoder layers or not (for non-contrastive learning)")
    
    ##################################################### DATASET #####################################################
    
    # Type
    parser.add_argument('--env', type=str, default='tsp', help=f"Problem to solve: {', '.join(ENVS)}")
    parser.add_argument('--scenario', type=str, default='contrastive', help=f"Type of scenario: {', '.join(SCENARIOS)}")
    
    # Number of samples
    parser.add_argument('--train_size', type=int, default=1280000, help="Number of instances per epoch during training")
    parser.add_argument('--val_size', type=int, default=10000, help="Number of instances per epoch during validation")
    
    # Nodes & obstacles
    parser.add_argument('--num_nodes', type=int, default=20, help="Number of visitable nodes")
    parser.add_argument('--num_obs', type=int, default=0, help="Number of avoidable obstacles (0 to not use obstacles)")
    
    ##################################################### TRAINING #####################################################
    
    # Epochs
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--first_epoch', type=int, default=0, help="Initial epoch (relevant for learning rate decay)")
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=2048, help="Number of instances per batch during training")
    parser.add_argument('--batch_size_val', type=int, default=1024, help="Number of instances per batch during validation")
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="Learning rate decay per epoch")
    
    # Gradient norm
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max L2 norm for gradient clipping (0 to disable clipping)")
    
    # Baselines
    parser.add_argument('--baseline', type=str, default=None, help=f"Baseline to train with: ','.join({BASELINES})")
    parser.add_argument('--bl_alpha', type=float, default=0.05, help="Significance in the t-test for updating rollout baseline")
    parser.add_argument('--bl_exp', type=float, default=0.8, help="Exponential moving average baseline decay")
    parser.add_argument('--bl_warmup', type=int, default=None, help="Number of epochs to warmup a rollout baseline. None means 1 exponential warmup epoch.")
    
    ##################################################### MISCELLANEOUS #####################################################
    
    # Device
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--use_distributed', type=str2bool, default=False, help="True to use distributed data parallel training (requires use_cuda=True)")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of parallel workers loading data batches")
    
    # Save model
    parser.add_argument('--save_dir', default='outputs', help="Directory to save trained models to")
    parser.add_argument('--cp_freq', type=int, default=1, help="Save checkpoint every n epochs, 0 to save no checkpoints")
    
    # Load model
    parser.add_argument('--load_path', type=str, default='', help="Path to load model parameters and optimizer state from")
    parser.add_argument('--resume', type=str2bool, default=False, help="Resume training beggining from the same epoch of load_path")
    
    # Logging
    parser.add_argument('--log_step', type=int, default=50, help="Log info every log_step steps")
    
    # Debug mode
    parser.add_argument('--debug', type=str2bool, default=False, help="Activate debug mode")
    
    ##################################################### END #####################################################
    
    # Parse arguments
    opts = parser.parse_args(args)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    opts.use_cuda = opts.use_cuda if cuda_available else False
    opts.use_distributed = opts.use_distributed if cuda_available else False
    print(f"[*] Device: {'CUDA' if opts.use_cuda else 'CPU'}{'' if cuda_available else ', since CUDA is not available'}")
    
    # Debug mode
    if opts.debug:
        opts.use_distributed = False
        opts.num_workers = 0
    
    # Filename
    time_txt = time.strftime("%Y-%m-%d-%H-%M-%S")  # time.strftime("%Y%m%dT%H%M%S")
    encoder = {'graph': opts.graph_encoder, 'image': opts.image_encoder}.get(opts.scenario, '')
    opts.model = f"{opts.graph_encoder}_{opts.image_encoder}" if encoder=='' else f"{encoder}_{opts.decoder}"
    opts.save_dir = os.path.join(
        opts.save_dir, f"{opts.env}-{opts.scenario}_{opts.num_nodes}", f"{opts.model}_{time_txt}",
    )
    
    # Set seed for reproducibility
    set_seed(seed=opts.seed)
    return opts


def main(rank=0, opts=None, world_size=1):

    # Tensorboard logger
    tb_logger = config_logger(opts=opts)
    
    # Configure device
    if opts.use_distributed:
        setup(rank=rank, world_size=world_size)
        opts.device = rank
    else:
        opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Load model
    model, checkpoint, first_epoch = load_model(opts=opts, train=True)
    from nets.encoders.gat_encoder import GraphAttentionEncoder
    model = GraphAttentionEncoder(n_layers=3, n_heads=8, hidden_dim=128)

    # ---------------------------------------------------------------------
    
    # Optimizer
    optimizer = load_optimizer(
        opts=opts,
        model=model,
        baseline=None,
        checkpoint=checkpoint
    )
    
    # Load learning rate scheduler, decay by lr_decay once per epoch
    lr_scheduler = load_lr_scheduler(
        optimizer=optimizer,
        lr_decay=opts.lr_decay
    )
    
    # Validation dataloader
    val_dataloader = get_generator(
        env=opts.env,
        num_samples=opts.val_size,
        num_nodes=opts.num_nodes,
        num_obs=opts.num_obs,
        image_size=opts.image_size,
        batch_size=opts.batch_size_val,
        use_cuda=opts.use_cuda,
        use_distributed=opts.use_distributed,
        num_workers=opts.num_workers
    )

    #dac
    val_history = []  # Lista para guardar el rendimiento de cada época
    
    # Train loop
    for epoch in range(opts.num_epochs):

        # Measure training time
        start_time = time.time()
    
        # Train dataloader
        train_dataloader = get_generator(
            env=opts.env,
            num_samples=opts.train_size,
            num_nodes=opts.num_nodes,
            num_obs=opts.num_obs,
            image_size=opts.image_size,
            batch_size=opts.batch_size,
            use_cuda=opts.use_cuda,
            use_distributed=opts.use_distributed,
            num_workers=opts.num_workers
        )
        
        # Current training step
        step = epoch * opts.train_size

        # Tensorboard info
        if not opts.debug:
            tb_logger.log_value(
                name='learnrate_pg0',
                value=optimizer.param_groups[0]['lr'],
                step=step
            )
        
        # Bath iteration
        print(f"\nStart train epoch {epoch}, lr={optimizer.param_groups[0]['lr']}")
        for batch_id, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            
            #dac----------------------------------------------------------------------------
            if epoch == first_epoch and batch_id == 0:
                from utils.plot_utils import save_rotation_check
                # Extraemos el primer ejemplo del batch y lo pasamos a numpy
                # Recordamos que 'nodes' y 'nodes_rotated' vienen del generador modificado
                n_orig = batch['nodes'][0].detach().cpu().numpy()
                n_rot = batch['nodes_rotated'][0].detach().cpu().numpy()
                save_rotation_check(n_orig, n_rot, opts.save_dir)
            #-------------------------------------------------------------------------------

            # Move batch to device
            batch = move_to(var=batch,device=opts.device)
            
            # Make predictions
            output = model(batch)
            
            # Calculate training loss
            loss = contrastive_loss(*output[:2])
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            grad_norms = clip_grad_norms(param_groups=optimizer.param_groups,max_norm=opts.max_grad_norm)
            optimizer.step()

            # Logging
            if step % int(opts.log_step) == 0:
                log_values(
                    loss=loss,
                    grad_norms=grad_norms,
                    epoch=epoch,
                    batch_id=batch_id,
                    step=step,
                    tb_logger=tb_logger,
                    use_critic=opts.baseline == 'critic',
                    debug=opts.debug
                )
            step += 1

        # Measure training time and report results
        epoch_duration = time.time() - start_time
        print(f"Finished epoch {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))}")

        # Save trained model
        if (opts.cp_freq != 0 and epoch % opts.cp_freq == 0) or epoch == opts.epochs - 1:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                save_dir=opts.save_dir,
                epoch=epoch
            )
            
        # Perform validation
        loss_val = validate(
            model=model,
            dataloader=val_dataloader,
            loss_func=contrastive_loss,
            device=opts.device
        ).mean().item()

        #dac
        val_history.append(loss_val) # Guardar el valor en el historial

        # Tensorboard info
        if not opts.debug:
            tb_logger.log_value(
                name='val_avg_reward',
                value=loss_val,
                step=step
            )

        # Update lr_scheduler
        lr_scheduler.step()
    
    # End training
    if opts.use_distributed:
        cleanup()
    
    #dac
    from utils.plot_utils import save_training_results
    label_name = 'Loss (Contrastive)'
    save_training_results(val_history, label_name, opts.save_dir, 'training_performance')
    
    print('Finished')


if __name__ == "__main__":
    
    # Get options
    opts = get_options()
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    
    # Main
    if opts.use_cuda and opts.use_distributed and world_size > 1:
        torch.multiprocessing.spawn(
            main, args=(opts, world_size), nprocs=world_size
        )
    else:
        main(opts=opts)
