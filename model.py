import torch
from torch import nn
import torch.nn.functional as F


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(3200, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True)
        ])

        self.classify = nn.Linear(64, 10)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.classify(x)
        return x

    def get_repr_and_pred(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        for layer in self.fc_layers:
            x = layer(x)
        pred = self.classify(x)
        return x, pred


class JDDA(nn.Module):
    def __init__(self):
        super(JDDA, self).__init__()
        self.base_model = Lenet()
        self.repr_s, self.repr_t = None, None
        self.pred_s = None

    def get_CORAL_loss(self):
        source, target = self.repr_s, self.repr_t
        d = source.data.shape[1]
        batch_size = source.data.shape[0]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (batch_size - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (batch_size - 1)

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        # loss = loss / (4 * d * d)

        return loss

    def forward(self, xs, xt):
        repr_s, pred_s = self.base_model.get_repr_and_pred(xs)
        repr_t, _ = self.base_model.get_repr_and_pred(xt)
        self.repr_s, self.repr_t = repr_s, repr_t
        self.pred_s = pred_s

    def predict(self, x):
        return self.base_model(x)

    def get_all_loss(self, ys, loss_func):
        classification_loss = loss_func(self.pred_s, ys)
        discriminative_loss = self.get_discriminative_loss(ys)
        coral_loss = self.get_CORAL_loss()
        return classification_loss, coral_loss, discriminative_loss

    def get_discriminative_loss(self, ys, method="InstanceBased"):
        batch_size = self.repr_s.shape[0]
        if method == "InstanceBased":
            source = self.repr_s
            dist = torch.sum((torch.unsqueeze(source, 2) - torch.transpose(source, 1, 0)) ** 2, 1)

            label_indicator = torch.unsqueeze(ys, 1) - ys
            label_indicator[label_indicator != 0] = 1
            label_indicator = ((label_indicator == 0) + 0)
            margin0 = 0.0
            margin1 = 100.0

            F0 = torch.max(dist-margin0, torch.Tensor([margin0]).expand_as(dist).to(source.device)) ** 2
            F1 = torch.max(margin1 - dist, torch.Tensor([margin1]).expand_as(dist).to(source.device)) ** 2

            intra_loss = torch.mean(F0 * label_indicator)
            inter_loss = torch.mean(F1 * (1.0 - label_indicator))

            discriminative_loss = (intra_loss + inter_loss) / batch_size / batch_size

        return discriminative_loss



