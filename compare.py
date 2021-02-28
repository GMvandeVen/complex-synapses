#!/usr/bin/env python3
import os
import argparse
import numpy as np
from param_stamp import get_param_stamp_from_args
import utils
from visual import visual_plt as plt
import main_example


parser = argparse.ArgumentParser('./compare.py',
                                 description='Compare training a DNN on split MNIST with and wihtout complex synapses.')
parser.add_argument('--seed', type=int, default=0, help='random seed for first set of experiments')
parser.add_argument('--n-seeds', type=int, default=1, help='number of times to repeat experiment')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./store/plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir', help="default: %(default)s")

# Data-stream parameters
data_params = parser.add_argument_group('Data-stream Parameters')
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
train_params.add_argument('--decay', type=float, default=0., help="weight decay (only for w/o complex synapses)")
# -prameters specific to complex synapses
cs_params = parser.add_argument_group('Complex Synapse Parameters (Benna & Fusi, 2016; Nat Neuro)')
cs_params.add_argument('--beakers', type=int, default=5, help="# of beakers")
cs_params.add_argument('--alpha', type=float, default=500., help="fastest timescale of the complex synapse")
cs_params.add_argument('--beta', type=float, default=2500., help="slowest timescale of the complex synapse")

# Evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--loss-log', type=int, default=100, metavar="N", help="# iters after which to plot loss")
eval_params.add_argument('--eval-log', type=int, default=100, metavar="N", help="# iters after which to plot accuracy")
eval_params.add_argument('--eval-n', type=int, default=1024, help="# samples for evaluating accurcy on-the-fly")



def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_example.run(args)
    # -get average precisions
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir,  param_stamp))
    # -return tuple with the results
    return (dict, ave)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments
    args = parser.parse_args()
    args.pdf = False
    args.experiment = "splitMNIST"

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"WITHOUT COMPLEX SYNAPSES"----###

    args.cs = False
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="Standard")


    ###----"WITH COMPLEX SYNAPSES"----###

    args.cs = True
    CS = {}
    CS = collect_all(CS, seed_list, args, name="With complex synapses")


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    prec = {}
    prec_so_far = {}
    ave_prec = {}

    ## Create lists for all extracted <dicts> and <lists> with fixed order
    for seed in seed_list:

        i = 0
        prec[seed] = [
            NONE[seed][i]["ave_acc"], CS[seed][i]["ave_acc"],
        ]
        prec_so_far[seed] = [
            NONE[seed][i]["ave_acc_so_far"], CS[seed][i]["ave_acc_so_far"],
        ]
        i = 1
        ave_prec[seed] = [
            NONE[seed][i], CS[seed][i],
        ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}   -   {}".format(args.experiment, scheme)
    x_axes = NONE[args.seed][0]["iters"]

    # select names / colors / ids
    names = ["Standard", "Compex Synapses" ]
    colors = ["grey", "red"]
    ids = [0, 1]

    # open pdf
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # print results to screen
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:19s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:19s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)

    # line-plot (average accuracy so far)
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(prec_so_far[args.seed][id])):
            all_entries = [prec_so_far[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if args.n_seeds > 1:
                new_sem_line.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    ylim = (0, 1.02) if args.scenario == "class" else None
    ylabel = "Test accucary ({} so far)".format("based on digits" if args.scenario=="class" else "on tasks")
    figure = plt.plot_lines(ave_lines, x_axes=x_axes,
                            line_names=names, colors=colors, title=title,
                            xlabel="# iterations so far", ylabel=ylabel,
                            list_with_errors=sem_lines if args.n_seeds > 1 else None, ylim=ylim)
    figure_list.append(figure)

    # line-plot (average accuracy)
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(prec[args.seed][id])):
            all_entries = [prec[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if args.n_seeds>1:
                new_sem_line.append(np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    ylim = (0,1.02) if args.scenario=="class" else None
    ylabel = "Test accucary ({})".format("based on all digits" if args.scenario=="class" else "on all tasks")
    figure = plt.plot_lines(ave_lines, x_axes=x_axes,
                            line_names=names, colors=colors, title=title,
                            xlabel="# iterations so far", ylabel=ylabel,
                            list_with_errors=sem_lines if args.n_seeds > 1 else None, ylim=ylim)
    figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))