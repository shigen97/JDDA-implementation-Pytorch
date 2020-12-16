import torch
from torch import nn, optim
import torch.nn.functional as f
import numpy as np
from load_data import get_dataloader
from model import JDDA
from itertools import cycle


class_num = 10
batch_size = 128
total_iters = 200200
lr = 0.0001
discriminative_loss_param = 0.03 ##0.03 for InstanceBased method, 0.01 for CenterBased method
domain_loss_param = 8
device = torch.device('cuda:2')

source_dataloader = get_dataloader('mnist', batch_size=batch_size, split='train')
target_dataloader = get_dataloader('mnistm', batch_size=batch_size, split='train')
test_dataloader = get_dataloader('mnistm', batch_size=batch_size, split='test')


model = JDDA()
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# since length of svhn and mnist is different
# origin author train steps 200200
# and drop the data that batch size is not 128


epoch_classification_loss = 0.0
epoch_coral_loss = 0.0
epoch_discriminative_loss = 0.0

for step, ((xs, ys), (xt, _)) in enumerate(zip(cycle(source_dataloader), cycle(target_dataloader))):

    model.train()
    xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

    model.forward(xs, xt)

    classification_loss, coral_loss, discriminative_loss = model.get_all_loss(ys, loss_func)

    total_loss = classification_loss + \
                 coral_loss * domain_loss_param + \
                 discriminative_loss * discriminative_loss_param

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    epoch_classification_loss += classification_loss.item()
    epoch_coral_loss += coral_loss.item()
    epoch_discriminative_loss += discriminative_loss.item()

    if (step + 1) >= total_iters:
        break
    elif (step + 1) % 200 == 0:
        print((step + 1) % 200 == 0)

        epoch_total_loss = epoch_classification_loss + \
                           domain_loss_param * epoch_coral_loss + \
                           discriminative_loss_param * epoch_discriminative_loss

        print("***************", step + 1, "***************")
        print(
            "TotalLoss={} \n"
            "SourceLoss={} \n"
            "DomainLoss={} \n"
            "DiscriminativeLoss={}\n ".
            format(epoch_total_loss / step,
                   epoch_classification_loss / step,
                   epoch_coral_loss / step,
                   epoch_discriminative_loss / step))

        epoch_classification_loss = 0.0
        epoch_coral_loss = 0.0
        epoch_discriminative_loss = 0.0

        acc_num = 0
        total_number = len(test_dataloader.dataset)
        with torch.no_grad():
            model.eval()
            for _, (xt, yt) in enumerate(test_dataloader):

                xt = xt.to(device)
                yt = yt.to(device)

                pred = model.predict(xt)
                prediction = torch.argmax(pred, 1)

                acc_num += (prediction == yt).sum().item()
        acc = acc_num / total_number
        print('acc=', acc)












