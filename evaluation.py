"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import ipdb
from torch.autograd.variable import Variable


def eval_custom(model, loader, variable="states", use_cuda=False, num_samples=1, custom_state=None):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        model: the RL policy (actor + critic)
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    total_loss, total_acc = 0, 0
    total = 0 # number of samples

    if use_cuda:
        model.actor.cuda()
        model.critic.cuda()
    model.actor.eval()
    model.critic.eval()

    if variable == "states":
        with torch.no_grad():
            for batch_idx, (inputs) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                if use_cuda:
                    inputs = inputs.cuda()
                loss = -model.critic.q1_forward(inputs, model.actor(inputs)).mean()
                total_loss += loss.item()*batch_size
            total_loss = total_loss / total
            total_acc = total_loss
    elif variable == "actions":
        # get first sample from batch.
        # perturb /w noise from normal dist (0 mean, 0.1 var), and sample loss "num_samples" times
        # Note: same state is sample every time (intentional + tested), since we want to
        # keep the state fixed and vary the action.
        if(custom_state):
            state = custom_state
        else:
            ptr = iter(loader)
            state = ptr.next()[0].unsqueeze(0).cuda()
        action = model.actor(state)
        m = torch.distributions.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.1]))
        with torch.no_grad():
            for _ in range(num_samples):
                action = model.actor(state)
                noise = m.sample(action.size()).squeeze(0).T.cuda() # hacky reshaping. fix this.
                loss = -model.critic.q1_forward(state, action + noise).mean()
                total_loss += loss.item()
            total_loss = total_loss / num_samples
            total_acc = total_loss
    return total_loss, total_acc


def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total
