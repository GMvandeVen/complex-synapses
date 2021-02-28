import tqdm



def train_stream(model, datastream, iters=2000, loss_cbs=list(), eval_cbs=list()):
    '''Incrementally train a model on a stream of data.

    Args:
        model (Classifier): model to be trained, must have a built-in `train_a_batch`-method
        datastream (DataStream): iterator-object that returns for each iteration the training data
        iters (int, optional): max number of iterations, could be smaller if `datastream` runs out (default: ``2000``)
        *_cbs (list of callback-functions, optional): for evaluating training-progress (defaults: empty lists)
    '''

    # Define tqdm progress bar(s)
    progress = tqdm.tqdm(range(1, iters + 1))

    for batch_id, (x,y,t) in enumerate(datastream, 1):

        if batch_id > iters:
            break

        x = x.to(model._device())
        y = y.to(model._device())

        loss_dict = model.train_a_batch(x, y, t)

        # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
        for loss_cb in loss_cbs:
            if loss_cb is not None:
                loss_cb(progress, batch_id, loss_dict)
        for eval_cb in eval_cbs:
            if eval_cb is not None:
                eval_cb(model, batch_id)

    # Close progres-bar(s)
    progress.close()