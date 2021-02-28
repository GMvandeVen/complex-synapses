import numpy as np
from data.load import prepare_datasets
from data.labelstream import TaskBasedStream, RandomStream
from models.encoder import Classifier



def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''

    if args.experiment=="splitMNIST" and args.tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
    classes_per_task = 10 if args.experiment=="permMNIST" else int(np.floor(10/args.tasks))
    if args.stream=="task-based":
        labels_per_batch = True if ((not args.scenario=="class") or classes_per_task==1) else False
        label_stream = TaskBasedStream(
            n_tasks=args.tasks, iters_per_task=args.iters if labels_per_batch else args.iters*args.batch,
            labels_per_task=classes_per_task if args.scenario=="class" else 1
        )
    elif args.stream=="random":
        label_stream = RandomStream(labels=args.tasks*classes_per_task if args.scenario=="class" else args.tasks)
    else:
        raise NotImplementedError("Stream type '{}' not currently implemented.".format(args.stream))
    config = prepare_datasets(
        name=args.experiment, n_labels=label_stream.n_labels, classes=(args.scenario=="class"),
        classes_per_task=classes_per_task, dir=args.d_dir, exception=(args.seed==1), only_config=True,
    )
    softmax_classes = label_stream.n_labels if args.scenario=="class" else (
        classes_per_task if (args.scenario=="domain" or args.singlehead) else classes_per_task*label_stream.n_labels
    )
    model_name = Classifier(
        image_size=config['size'], image_channels=config['channels'], classes=softmax_classes,
        fc_layers=args.fc_lay, fc_units=args.fc_units,
    ).name
    param_stamp = get_param_stamp(args, model_name, verbose=False)
    return param_stamp



def get_param_stamp(args, model_name, verbose=True):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for protocol
    task_stamp = "{exp}-{str}-{sce}IL-i{ite}-b{bat}".format(
        exp=args.experiment, sce=args.scenario, str=args.stream, ite=args.iters, bat=args.batch,
    )
    if verbose:
        print(" --> data-stream:   "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for training
    cs_stamp = (
        "-decay{}".format(args.decay) if args.decay>0. else ""
    ) if (not args.cs) or args.beakers==0 else "-{}beta{}-beakers{}".format(
        "alpha{}-".format(args.alpha) if args.beakers>1 else "", args.beta, args.beakers,
    )
    training_stamp = "lr{lr}-{optim}{cs_stamp}".format(
        lr=args.lr, optim=args.optimizer if not args.cs else ("cs" if args.optimizer=="sgd" else "adam-cs"),
        cs_stamp=cs_stamp,
    )
    if verbose:
        print(" --> training:      " + training_stamp)

    # --> combine
    param_stamp = "{}--{}--{}{}".format(
        task_stamp, model_stamp, training_stamp, "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp