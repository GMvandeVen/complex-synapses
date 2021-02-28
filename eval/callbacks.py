from eval import evaluate


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def def_eval_cb(log, test_datasets, scenario, provide_task_info=False, iters_per_task=None, classes_per_task=None,
                test_size=None, visdom=None, summary_graph=True, metrics_dict=None, shuffle=True):
    '''Initiates function for evaluating performance of classifier (in terms of precision).

    Args:
        log (int): model will be evaluated after every ``log`` iterations
        test_datasets (list of Datasets): also if only one, it should be presented as a list
        scenario (str)
        provide_task_info (bool, optional): whether info about task-ID should be provided to model during testing
        iters_per_task (int, optional): should be provided with `dissociated` streams
        classes_per_task (int, optional): should be provided with `dissociated` streams according to Class-IL scenario
    '''

    # If selected, import module for plotting into visdom
    if visdom is not None:
        from visual import visual_visdom

    # Define the callback-function
    def eval_cb(classifier, iter, **kwargs):
        '''Callback-function, to evaluate performance of classifier.'''

        # Find current task/domain/episode (if using dissociated stream).
        tasks_so_far = None if (iters_per_task is None) else int((iter-1)/iters_per_task)+1

        if iter % log == 0:
            # Evaluate the classifier on all tasks / classes
            (precs, average_precs) = evaluate.precision(classifier, test_datasets, up_to=None, shuffle=shuffle,
                                                        provide_task_info=provide_task_info,
                                                        test_size=iter if test_size=="iters" else test_size)
            # -also get average accuracy only based on tasks/classes seen so far
            labels_so_far = tasks_so_far * classes_per_task if scenario == "class" else tasks_so_far
            ave_precs_so_far = sum([precs[task_id] for task_id in range(labels_so_far)]) / labels_so_far

            # Send results to visdom server for plotting
            names = ['{} {}'.format(scenario, i+1) for i in range(len(test_datasets))]
            if visdom is not None:
                visual_visdom.visualize_scalars(
                    precs, names=names, title="acc_per_task ({})".format(visdom["graph"]),
                    iteration=iter, env=visdom["env"], ylabel="test precision"
                )
                # -if requested: summary graph (i.e., average accuracy over all tasks / classes)
                if len(test_datasets)>1 and summary_graph:
                    visual_visdom.visualize_scalars(
                        [average_precs], names=["ave"], title="ave_acc ({})".format(visdom["graph"]),
                        iteration=iter, env=visdom["env"], ylabel="test precision"
                    )
                # -if requested: summary graph based only on all tasks / classes so far (if dissociated stream)
                if len(test_datasets)>1 and summary_graph and tasks_so_far is not None:
                    visual_visdom.visualize_scalars(
                        [ave_precs_so_far], names=["ave"], title="ave_acc_so_far ({})".format(visdom["graph"]),
                        iteration=iter, env=visdom["env"], ylabel="test precision"
                    )

            # Append results to metrics-dict
            if metrics_dict is not None:
                key = "class" if scenario=="class" else "task"
                for task_id, _ in enumerate(names):
                    metrics_dict["acc_per_{}".format(key)]["{}_{}".format(key, task_id+1)].append(precs[task_id])
                metrics_dict["ave_acc"].append(average_precs)
                metrics_dict["ave_acc_so_far"].append(ave_precs_so_far)
                metrics_dict["iters"].append(iter)

    ## Return the callback-function (except if neither visdom nor metrics-dict is selected!).
    return eval_cb if (visdom is not None) or (metrics_dict is not None) else None


##------------------------------------------------------------------------------------------------------------------##

###############################################################
## Callback-functions for keeping track of training-progress ##
###############################################################

def def_loss_cb(log, visdom, progress_bar=True, iters_per_task=None, tasks=None, task_name="Task"):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    # If selected, import module for plotting into visdom
    if visdom is not None:
        from visual import visual_visdom

    # Define the callback-function
    def cb(bar, iter, loss_dict):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        # Find current task/domain/episode (if using dissociated stream).
        task = None if (iters_per_task is None) else int((iter-1)/iters_per_task)+1

        # Progress-bar.
        if progress_bar and bar is not None:
            task_stm = "" if (task is None) else " {}: {}{} |".format(
                task_name, task, "/{}".format(tasks) if tasks is not None else ""
            )
            bar.set_description(
                '  <CLASSIFIER>   |{t_stm} training loss: {loss:.3} | training precision: {prec:.3} |'
                    .format(t_stm=task_stm, loss=loss_dict['loss'], prec=loss_dict['precision'])
            )
            bar.update(1)

        # Log the loss to visdom for on-the-fly plotting.
        if (visdom is not None) and (iter % log == 0):
            plot_data = [loss_dict['loss']]
            names = ['loss']
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iter,
                title="Loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function.
    return cb
