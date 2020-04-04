import os
from os.path import join
import shutil
import logging
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def create_loss_plot(exp_dir, epochs, train_losses, test_losses, fig_name='loss'):
    """Plot losses and save.

    Args:
        exp_dir (str): experiment directory.
        epochs (list): list of epochs (x-axis of loss plot).
        train_losses (list): list with train loss during each epoch.
        test_losses (list): list with test loss during each epoch.

    """
    f = plt.figure()
    plt.title(f"{fig_name} plot")
    plt.xlabel("Epoch")
    plt.ylabel(f"{fig_name}")
    plt.grid(True)
    plt.plot(epochs, train_losses, 'b', marker='o', label=f'train {fig_name}')
    plt.plot(epochs, test_losses, 'r', marker='o', label=f'val {fig_name}')
    plt.legend()
    plt.savefig(join(exp_dir, f'{fig_name}.png'))
    plt.close(f)


def save_checkpoint(state, target_dir, file_name='checkpoint.pth.tar',
                    backup_as_best=False,):
    """Save checkpoint to disk.

    Args:
        state: object to save.
        target_dir (str): Full path to the directory in which the checkpoint
            will be stored.
        backup_as_best (bool): Should we backup the checkpoint as the best
            version.
        file_name (str): the name of the checkpoint.

    """
    best_model_path = os.path.join(target_dir, 'model_best.pth.tar')
    target_model_path = os.path.join(target_dir, file_name)

    os.makedirs(target_dir, exist_ok=True)
    torch.save(state, target_model_path)
    if backup_as_best:
        shutil.copyfile(target_model_path, best_model_path)


def cal_num_batches(len_data, batch_size):
    return (len_data + batch_size - 1) // batch_size


def create_confusion_matrix_plot(exp_dir, y_true, y_pred, class_names):
    # f = plt.figure()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot non-normalized confusion matrix
    ax1 = plot_confusion_matrix(ax1, y_true, y_pred, classes=class_names,
                                title='Confusion matrix')

    # Plot normalized confusion matrix
    ax2 = plot_confusion_matrix(ax2, y_true, y_pred, classes=class_names, normalize=True,
                                title='Normalized confusion matrix')
    fig.tight_layout()
    fig.legend()
    fig.savefig(join(exp_dir, 'confusion_matrix.png'))
    plt.close(fig)


def plot_confusion_matrix(ax, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    From scikit-learn example: 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    # fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # def __str__(self):
    #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #     return fmtstr.format(**self.__dict__)


def save_metrics(pathfile, metrics):
    with open(pathfile, 'w') as f:
        json.dump(metrics, f)


def setup_logging(log_path=None, log_level='DEBUG', logger=None, fmt=None):
    """Prepare logging for the provided logger.

    Args:
        log_path (str, optional): full path to the desired log file.
        debug (bool, optional): log in verbose mode or not.
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used.
        fmt (str, optional): format for the logging message.

    """
    log_format = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'
    if fmt is None:
        fmt = log_format

    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info('Log file is %s', log_path)
