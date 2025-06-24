import torch


def freeze_model(m):
    for param in m.parameters():
        param.requires_grad = False

    m.eval()
    return m
