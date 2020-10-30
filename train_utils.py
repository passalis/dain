import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from lob_loader import get_wf_lob_loaders
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def lob_epoch_trainer(model, loader, lr=0.0001, optimizer=optim.RMSprop):
    model.train()

    dean_params = [{'params': model.base.parameters()}]
    if model.dean.mode in ['adaptive_avg', 'adaptive_scale', 'full']: 
        dean_params.append({'params': model.dean.mean_layer.parameters(), 'lr': lr * model.dean.mean_lr})
    if model.dean.mode in ['adaptive_scale', 'full']:
        dean_params.append({'params': model.dean.scaling_layer.parameters(), 'lr': lr * model.dean.scale_lr})
    if model.dean.mode == 'full':
        dean_params.append({'params': model.dean.gating_layer.parameters(), 'lr': lr * model.dean.gate_lr})
    
    model_optimizer = optimizer(dean_params, lr=lr)

    criterion = CrossEntropyLoss()
    train_loss, counter = 0, 0

    for (inputs, targets) in loader:
        model_optimizer.zero_grad()

        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        targets = torch.squeeze(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        model_optimizer.step()

        train_loss += loss.item()
        counter += inputs.size(0)

    loss = (loss / counter).cpu().data.numpy()
    return loss


def lob_evaluator(model, loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    for (inputs, targets) in tqdm(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        predicted_labels.append(predicted.cpu().numpy())
        true_labels.append(targets.cpu().data.numpy())

    true_labels = np.squeeze(np.concatenate(true_labels))
    predicted_labels = np.squeeze(np.concatenate(predicted_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                           average='macro')
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    metrics = {}
    metrics['accuracy'] = np.sum(true_labels == predicted_labels) / len(true_labels)

    metrics['precision'], metrics['recall'], metrics['f1'] = precision, recall, f1

    metrics['precision_avg'], metrics['recall_avg'], metrics['f1_avg'] = precision_avg, recall_avg, f1_avg

    metrics['kappa'] = kappa

    return metrics


def train_evaluate_anchored(model, epoch_trainer=lob_epoch_trainer, evaluator=lob_evaluator,
                            horizon=0, window=5, batch_size=128, train_epochs=20, verbose=True,
                            use_resampling=True, learning_rate=0.0001, splits=[6, 7, 8], normalization='std'):
    """
    Trains and evaluates a model for using an anchored walk-forward setup
    :param model: model to train
    :param epoch_trainer: function to use for training the model (please refer to lob.model_utils.epoch_trainer() )
    :param evaluator: function to use for evaluating the model (please refer to lob.model_utils.epoch_trainer() )
    :param horizon: the prediction horizon for the evaluation (0, 5 or 10)
    :param window: the window to use
    :param batch_size: batch size to be used
    :param train_epochs: number of epochs for training the model
    :param verbose:
    :return:
    """

    results = []

    for i in splits:
        print("Evaluating for split: ", i)
        train_loader, test_loader = get_wf_lob_loaders(window=window, horizon=horizon, split=i, batch_size=batch_size,
                                                       class_resample=use_resampling, normalization=normalization)
        current_model = model()
        current_model.cuda()
        for epoch in range(train_epochs):
            loss = epoch_trainer(model=current_model, loader=train_loader, lr=learning_rate)
            if verbose:
                print("Epoch ", epoch, "loss: ", loss)

        test_results = evaluator(current_model, test_loader)
        print(test_results)
        results.append(test_results)

    return results


def get_average_metrics(results):
    precision, recall, f1 = [], [], []
    kappa = []
    acc = []
    for x in results:
        acc.append(x['accuracy'])
        precision.append(x['precision_avg'])
        recall.append(x['recall_avg'])
        f1.append(x['f1_avg'])
        kappa.append(x['kappa'])

    print("Precision = ", np.mean(precision))
    print("Recall = ", np.mean(recall))
    print("F1 = ", np.mean(f1))
    print("Cohen = ", np.mean(kappa))

    return acc, precision, recall, f1, kappa