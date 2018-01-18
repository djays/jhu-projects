import argparse
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from FacesDS import FacesDS
from SiameseNetwork import SiameseNetwork

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Face Similarity")
parser.add_argument("--train_set", default='train.txt', help="Location of file with training set")
parser.add_argument("--test_set", default='test.txt', help="Location of file with testing set")
parser.add_argument("--cpu", action='store_true', default=False, help="Run only on CPU")
parser.add_argument("--save", help="File to store the model.", default="siam.model")
parser.add_argument("--save_every_epoch", help="Enable storing the model per epoch if there is an improvement over the previous", default=False)
parser.add_argument("--load", help="File to load the model from.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training. (default=16)")
parser.add_argument("--epochs", default=30, type=int, help="Epochs through the data. (default=10)")
parser.add_argument("--learning_rate", "-lr", default=5e-5, type=float)


def train(options):
    model = SiameseNetwork()

    if not options.cpu:
        model = model.cuda()

    # Adamax ?
    optimizer = torch.optim.Adam(model.parameters(), options.learning_rate)
    criterion = torch.nn.BCELoss()

    # For tracking average loss
    min_loss = float("inf")

    train_loader = DataLoader(FacesDS(options.train_set, True), batch_size=options.batch_size, shuffle=True)

    for epoch in range(options.epochs):
        logging.info("At {0}-th epoch.".format(epoch))
        model.train()
        training_loss = loss_train_data(model, train_loader, options, criterion, optimizer)
        logging.info("Training loss after epoch {0}: {1}".format(epoch, training_loss))

    # TODO: test this
    torch.save(model.state_dict(), open(options.save, 'wb'))
    return model


def loss_train_data(model, train_loader, options, criterion, optimizer):
    for i_batch, batch in enumerate(train_loader):
        batch_im1, batch_im2, batch_label = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        if not options.cpu:
            batch_im1 = batch_im1.cuda()
            batch_im2 = batch_im2.cuda()
            batch_label = batch_label.cuda()

        prediction = model(batch_im1, batch_im2)

        optimizer.zero_grad()
        loss = criterion(prediction, batch_label)
        loss.backward()
        training_loss = loss.data[0]
        optimizer.step()
    return training_loss


def accuracy(model, test_loader, options):
    acc = 0.0
    for i_batch, valid_batch in enumerate(test_loader):
        batch_im1, batch_im2, batch_label = Variable(valid_batch[0], volatile=True), Variable(valid_batch[1], volatile=True), Variable(valid_batch[2], volatile=True)

        if not options.cpu:
            batch_im1 = batch_im1.cuda()
            batch_im2 = batch_im2.cuda()
            batch_label = batch_label.cuda()

        prediction = model(batch_im1, batch_im2)
        prediction = (prediction > 0.5).float()

        acc += np.count_nonzero((prediction == batch_label).data.cpu().numpy()) / float(options.batch_size)

    return acc / float(len(test_loader))


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
