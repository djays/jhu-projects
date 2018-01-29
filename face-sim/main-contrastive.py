import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from FacesDS import FacesDS
from SiameseNetworkB import SiameseNetwork

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Face Similarity using Contrastive Loss")
parser.add_argument("--train_set", default='train.txt', help="Location of file with training set")
parser.add_argument("--test_set", default='test.txt', help="Location of file with testing set")
parser.add_argument("--cpu", action='store_true', default=False, help="Run only on CPU")
parser.add_argument("--save", help="File to store the model.", default="siam.model")
parser.add_argument("--save_every_epoch", help="Enable storing the model per epoch if there is an improvement over the previous", default=False)
parser.add_argument("--load", help="File to load the model from.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training. (default=16)")
parser.add_argument("--epochs", default=50, type=int, help="Epochs through the data. (default=10)")
parser.add_argument("--learning_rate", "-lr", default=1e-5, type=float)


class CLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(CLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        dist = torch.clamp(self.margin - euclidean_distance, min=0.0)
        loss_contrastive = label * torch.pow(euclidean_distance, 2) + (1 - label) * torch.pow(dist, 2)
        return loss_contrastive.mean()


def train(options):
    model = SiameseNetwork()

    if not options.cpu:
        model = model.cuda()

    # Adamax ?
    optimizer = torch.optim.Adam(model.parameters(), options.learning_rate)
    criterion = CLoss(1.0)

    if not options.cpu:
        criterion.cuda()

    # For tracking average loss
    min_loss = float("inf")

    train_loader = DataLoader(FacesDS(options.train_set, True), batch_size=options.batch_size, shuffle=True)

    for epoch in range(options.epochs):
        logging.info("At {0}-th epoch.".format(epoch))
        model.train()
        training_loss = loss_train_data(model, train_loader, options, criterion, optimizer)
        logging.info("Training loss after epoch {0}: {1}".format(epoch, training_loss))

        # # validation
        # model.eval()
        # dev_avg_loss = accuracy(model, valid_loader, options)
        # logging.info("Accuracy is {0} at the end of epoch {1}".format(dev_avg_loss, epoch))
        # if options.save_every_epoch and min_loss > dev_avg_loss:
        #     min_loss = dev_avg_loss
        #     torch.save(model.state_dict(),
        #                open(options.save + ".bce_{0:.2f}.epoch_{1}".format(dev_avg_loss, epoch), 'wb'))

    torch.save(model.state_dict(), open(options.save, 'wb'))
    return model


def loss_train_data(model, train_loader, options, criterion, optimizer):
    for i_batch, batch in enumerate(train_loader):
        batch_im1, batch_im2, batch_label = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        if not options.cpu:
            batch_im1 = batch_im1.cuda()
            batch_im2 = batch_im2.cuda()
            batch_label = batch_label.cuda()

        im_em1, im_em2 = model(batch_im1, batch_im2)

        optimizer.zero_grad()
        loss = criterion(im_em1, im_em2, batch_label)
        loss.backward()
        training_loss = loss.data[0]
        optimizer.step()

    return training_loss


def accuracy(model, loader, options):
    acc = 0.0

    for _, valid_batch in enumerate(loader):
        batch_im1, batch_im2, batch_label = Variable(valid_batch[0], volatile=True), Variable(valid_batch[1], volatile=True), Variable(valid_batch[2], volatile=True)

        if not options.cpu:
            batch_im1 = batch_im1.cuda()
            batch_im2 = batch_im2.cuda()
            batch_label = batch_label.cuda()

        em1, em2 = model(batch_im1, batch_im2)
        e_dist = nn.functional.pairwise_distance(em1, em2)

        # Using similarity metric that has been empirically determined
        # Using 14.0 with current model can bu
        similarity_metric = 12.0    #(batch_im1.size()[1]/2.0) ** 0.5
        prediction = (e_dist < similarity_metric).float()

        acc += np.count_nonzero((prediction == batch_label).data.cpu().numpy()) / float(options.batch_size)

    return acc / float(len(loader))


def test(options):
    model = SiameseNetwork()
    if not options.cpu:
        model = model.cuda()
    model.load_state_dict(torch.load(open(options.load, 'rb')))
    model.eval()

    # Check training loss
    train_loader = DataLoader(FacesDS(options.train_set), batch_size=options.batch_size, shuffle=True)
    training_acc = accuracy(model, train_loader, options)
    logging.info("Training Accuracy : %s" % (training_acc,))

    # Check Testing loss
    test_loader = DataLoader(FacesDS(options.test_set), batch_size=options.batch_size, shuffle=True)
    testing_acc = accuracy(model, test_loader, options)
    logging.info("Testing Accuracy : %s" % (testing_acc,))


def app(options):
    if options.load:
        test(options)
    else:
        train(options)


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    app(args)
