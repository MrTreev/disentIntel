import os
import argparse
from torch.backends import cudnn
import torch

from solver import Solver
from data_loader import get_loader
from hparams import hparams, hparams_debug_string
global plslog
plslog = False


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    vcc_loader = get_loader(hparams)

    # Solver for training
    solver = Solver(vcc_loader, config, hparams)

    solver.train()


def intel(config):
    from intelligibility import (
        create_UASpeech_custom_v2,
        cut_vad,
        inference,
        dtw_diff_condense,
        scatter_plot_regression,
        n_utterances_t_times,
        reset_meta_dir,
    )
    # ~ Correlation testing (perform functions below in the listed order)
    # 1) prepare a custom UASpeech corpus, that will be used in all the following steps
    # 2) cut 15% of the audio durations at the beginning and end, then perform VAD
    if not os.path.exists(os.path.join(hparams.uaspeech_custom_3_audio_dir)):
        create_UASpeech_custom_v2()
        cut_vad(0.15, 0.0)
    # 3) perform inference with the previously trained SpeechSplit model
    inference(config)
    # 4) to test across the 4 available reference speaker pairs manually create the 4 reference
    #    dir's and copy files from inference dir in: control, reference, pathological dirs
    # 5) calculate all the required metrics
    dtw_diff_condense()
    # 6) scatter plot including regression line
    scatter_plot_regression()
    # 7) get correlation results for different amount of data used
    n_utterances_t_times(760, 1)  # 20,1000 and 760,1 -> change "pathological_only"

    # Some auxilary functions (to help with intuition about the data/representations)
    reset_meta_dir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( "--train_base", action="store_true", default=False)
    parser.add_argument( "--intel_assess", action="store_true", default=False)
    parser.add_argument( "--trained_model", type=str, default="run/models/57000-G.ckpt")

    # Training configuration.
    parser.add_argument( "--num_iters", type=int, default=1000000, help="number of total iterations")
    parser.add_argument( "--g_lr", type=float, default=0.0001, help="learning rate for G")
    parser.add_argument( "--beta1", type=float, default=0.9, help="beta1 for Adam optimizer")
    parser.add_argument( "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer")
    parser.add_argument( "--resume_iters", type=int, default=None, help="resume training from this step")

    # Miscellaneous.
    parser.add_argument("--use_tensorboard", action="store_false", default=True)
    parser.add_argument("--device_id", type=int, default=0)

    # Directories.
    parser.add_argument("--log_dir", type=str, default="run/logs")
    parser.add_argument("--model_save_dir", type=str, default="run/models")
    parser.add_argument("--sample_dir", type=str, default="run/samples")

    # Step size.
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=1000)
    parser.add_argument("--model_save_step", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=8)

    parser.add_argument("--train_new", action="store_true", default=False)

    config = parser.parse_args()

    if not config.train_new:
        config.resume_iters = int(sorted(
            os.listdir(path=config.model_save_dir)
        )[-1].split("-")[0])
        config.num_iters = int(config.num_iters - config.resume_iters)

    print("Config:")
    [print(f"  {key}: {value}") for key, value in config.__dict__.items()]
    print(hparams_debug_string())
    print("GPUs:")
    [print(f"  {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]
    if config.train_new:
        main(config)
    if config.intel_assess:
        intel(config)
