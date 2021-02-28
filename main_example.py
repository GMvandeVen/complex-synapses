#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import random
import torch
from torch import optim

# Custom-written libraries
import utils
from param_stamp import get_param_stamp, get_param_stamp_from_args
from models.encoder import Classifier
from eval import callbacks as cb, evaluate
from train import train_stream
from data.load import prepare_datasets
from data.labelstream import TaskBasedStream, RandomStream
from data.datastream import DataStream
import complex_synapses as cs
from visual import visual_plt



parser = argparse.ArgumentParser('./main_example.py',
                                 description='Test the complex synapse optimizer on a continual learning experiment.')
parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./store/plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir', help="default: %(default)s")

# Data-stream parameters
data_params = parser.add_argument_group('Data-stream Parameters')
data_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['splitMNIST', 'permMNIST'])
data_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
data_params.add_argument('--stream', type=str, default='task-based', choices=['task-based', 'random'])
data_params.add_argument('--tasks', type=int, default=5, help="# of tasks")
data_params.add_argument('--iters', type=int, default=500, help="# of batches (if `task-based`, this is per task)")
data_params.add_argument('--batch', type=int, default=128, help="batch-size")

# Model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, default=400, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer "
                                                                   " (instead of a 'multi-headed' one)")

# Training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
train_params.add_argument('--decay', type=float, default=0., help="weight decay (has no effect when used with --cs)")
# -prameters specific to complex synapses
cs_params = parser.add_argument_group('Complex Synapse Parameters (Benna & Fusi, 2016; Nat Neuro)')
cs_params.add_argument('--cs', action='store_true', help="use the complex synapse optimizer")
cs_params.add_argument('--beakers', type=int, default=5, help="# of beakers")
cs_params.add_argument('--alpha', type=float, default=500., help="fastest timescale of the complex synapse")
cs_params.add_argument('--beta', type=float, default=2500., help="slowest timescale of the complex synapse")

# Evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--loss-log', type=int, default=100, metavar="N", help="# iters after which to plot loss")
eval_params.add_argument('--eval-log', type=int, default=100, metavar="N", help="# iters after which to plot accuracy")
eval_params.add_argument('--eval-n', type=int, default=1024, help="# samples for evaluating accurcy on-the-fly")



def run(args, verbose=False):

    # Create plots- and results-directories, if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it printed to screen and exit
    if utils.checkattr(args, "get_stamp"):
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)


    #-------------------------------------------------------------------------------------------------#

    #-----------------------#
    #----- DATA-STREAM -----#
    #-----------------------#

    # Find number of classes per task
    if args.experiment=="splitMNIST" and args.tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
    classes_per_task = 10 if args.experiment=="permMNIST" else int(np.floor(10/args.tasks))

    # Print information on data-stream to screen
    if verbose:
        print("\nPreparing the data-stream...")
        print(" --> {}-incremental learning".format(args.scenario))
        ti = "{} classes".format(args.tasks*classes_per_task) if args.stream=="random" and args.scenario=="class" else (
            "{} tasks, with {} classes each".format(args.tasks, classes_per_task)
        )
        print(" --> {} data stream: {}\n".format(args.stream, ti))

    # Set up the stream of labels (i.e., classes, domain or tasks) to use
    if args.stream=="task-based":
        labels_per_batch = True if ((not args.scenario=="class") or classes_per_task==1) else False
        # -in Task- & Domain-IL scenario, each label is always for entire batch
        # -in Class-IL scenario, each label is always just for single observation
        #    (but if there is just 1 class per task, setting `label-per_batch` to ``True`` is more efficient)
        label_stream = TaskBasedStream(
            n_tasks=args.tasks, iters_per_task=args.iters if labels_per_batch else args.iters*args.batch,
            labels_per_task=classes_per_task if args.scenario=="class" else 1
        )
    elif args.stream=="random":
        label_stream = RandomStream(labels=args.tasks*classes_per_task if args.scenario=="class" else args.tasks)
    else:
        raise NotImplementedError("Stream type '{}' not currently implemented.".format(args.stream))

    # Load the data-sets
    (train_datasets, test_datasets), config, labels_per_task = prepare_datasets(
        name=args.experiment, n_labels=label_stream.n_labels, classes=(args.scenario=="class"),
        classes_per_task=classes_per_task, dir=args.d_dir, exception=(args.seed<10)
    )

    # Set up the data-stream to be presented to the network
    data_stream = DataStream(
        train_datasets, label_stream, batch=args.batch, return_task=(args.scenario=="task"),
        per_batch=labels_per_batch if (args.stream=="task-based") else args.labels_per_batch,
    )


    #-------------------------------------------------------------------------------------------------#

    #-----------------#
    #----- MODEL -----#
    #-----------------#

    # Define model
    # -how many units in the softmax output layer? (e.g., multi-headed or not?)
    softmax_classes = label_stream.n_labels if args.scenario=="class" else (
        classes_per_task if (args.scenario=="domain" or args.singlehead) else classes_per_task*label_stream.n_labels
    )
    # -set up model and move to correct device
    model = Classifier(
        image_size=config['size'], image_channels=config['channels'],
        classes=softmax_classes, fc_layers=args.fc_lay, fc_units=args.fc_units,
    ).to(device)
    # -if using a multi-headed output layer, set the "label-per-task"-list as attribute of the model
    model.multi_head = labels_per_task if (args.scenario=="task" and not args.singlehead) else None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- OPTIMIZER -----#
    #---------------------#

    # Define optimizer (only include parameters that "requires_grad")
    optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    if not args.cs:
        # Use the chosen 'standard' optimizer
        if args.optimizer == "sgd":
            model.optimizer = optim.SGD(optim_list, weight_decay=args.decay)
        elif args.optimizer=="adam":
            model.optimizer = optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=args.decay)
    else:
        # Use the "complex synapse"-version of the chosen optimizer
        if args.optimizer=="sgd":
            model.optimizer = cs.ComplexSynapse(optim_list, n_beakers=args.beakers, alpha=args.alpha, beta=args.beta,
                                                verbose=verbose)
        elif args.optimizer=="adam":
            model.optimizer = cs.AdamComplexSynapse(optim_list, betas=(0.9, 0.999), n_beakers=args.beakers,
                                                    alpha=args.alpha, beta=args.beta, verbose=verbose)


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(args, model.name, verbose=verbose)

    # Print some model-characteristics on the screen
    if verbose:
        utils.print_model_info(model, title="MAIN MODEL")

    # Prepare for keeping track of performance during training for storing & for later plotting in pdf
    metrics_dict = evaluate.initiate_metrics_dict(n_labels=label_stream.n_labels, classes=(args.scenario == "class"))

    # Prepare for plotting in visdom
    if args.visdom:
        env_name = "{exp}-{scenario}".format(exp=args.experiment, scenario=args.scenario)
        graph_name = "CS" if args.cs else "Normal"
        visdom = {'env': env_name, 'graph': graph_name}
    else:
        visdom = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Callbacks for reporting on and visualizing loss
    loss_cbs = [
        cb.def_loss_cb(
            log=args.loss_log, visdom=visdom, tasks=label_stream.n_tasks,
            iters_per_task=args.iters if args.stream=="task-based" else None,
            task_name="Episode" if args.scenario=="class" else ("Task" if args.scenario=="task" else "Domain")
        )
    ]

    # Callbacks for reporting and visualizing accuracy
    eval_cbs = [
        cb.def_eval_cb(log=args.eval_log, test_datasets=test_datasets, scenario=args.scenario,
                       iters_per_task=args.iters if args.stream=="task-based" else None,
                       classes_per_task=classes_per_task, metrics_dict=metrics_dict, test_size=args.eval_n,
                       visdom=visdom, provide_task_info=(args.scenario=="task"))
    ]
    # -evaluate accuracy before any training
    for eval_cb in eval_cbs:
        if eval_cb is not None:
            eval_cb(model, 0)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    # Keep track of training-time
    if args.time:
        start = time.time()
    # Train model
    if verbose:
        print("\nTraining...")
    train_stream(model, data_stream, iters=args.iters*args.tasks if args.stream=="task-based" else args.iters,
                 eval_cbs=eval_cbs, loss_cbs=loss_cbs)
    # Get total training-time in seconds, and write to file and screen
    if args.time:
        training_time = time.time() - start
        time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
        time_file.write('{}\n'.format(training_time))
        time_file.close()
        if verbose:
            print("=> Total training time = {:.1f} seconds\n".format(training_time))


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1 if args.scenario=="task" else None,
    ) for i in range(len(test_datasets))]
    average_precs = sum(precs) / len(test_datasets)
    # -print to screen
    if verbose:
        print("\n Precision on test-set:")
        for i in range(len(test_datasets)):
            print(" - {} {}: {:.4f}".format(args.scenario, i + 1, precs[i]))
        print('=> Average precision over all {} {}{}s: {:.4f}\n'.format(
            len(test_datasets), args.scenario, "e" if args.scenario=="class" else "", average_precs
        ))


    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # Average precision on full test set
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs))
    output_file.close()
    # -metrics-dict
    file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
    utils.save_object(metrics_dict, file_name)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = visual_plt.open_pdf(plot_name)

        # -show metrics reflecting progression during training
        figure_list = []  #-> create list to store all figures to be plotted

        # -generate all figures (and store them in [figure_list])
        key = "class" if args.scenario=='class' else "task"
        plot_list = []
        for i in range(label_stream.n_labels):
            plot_list.append(metrics_dict["acc_per_{}".format(key)]["{}_{}".format(key, i+1)])
        figure = visual_plt.plot_lines(
            plot_list, x_axes=metrics_dict["iters"], xlabel="Iterations", ylabel="Accuracy",
            line_names=['{} {}'.format(args.scenario, i+1) for i in range(label_stream.n_labels)]
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [metrics_dict["ave_acc"]], x_axes=metrics_dict["iters"], xlabel="Iterations", ylabel="Accuracy",
            line_names=['average (over all {}{}s)'.format(args.scenario, "e" if args.scenario=="class" else "")],
            ylim=(0,1)
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [metrics_dict["ave_acc_so_far"]], x_axes=metrics_dict["iters"], xlabel="Iterations", ylabel="Accuracy",
            line_names=['average (over all {}{}s so far)'.format(args.scenario, "e" if args.scenario=="class" else "")],
            ylim=(0,1)
        )
        figure_list.append(figure)

        # -add figures to pdf (and close this pdf).
        for figure in figure_list:
            pp.savefig(figure)

        # -close pdf
        pp.close()

        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))


if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -run experiment
    run(args, verbose=True)