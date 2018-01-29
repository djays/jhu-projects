import argparse
import logging

import dill
import torch
from torch import cuda
from torch.autograd import Variable

import model as model
from utils import tensor
from utils import rand

from model import NMT
import random

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="NMT Train")
parser.add_argument("--data_file", default="hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="words",
#parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = words)")
parser.add_argument("--trg_lang", default="phoneme",
#parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = phoneme)")
parser.add_argument("--model_file", required=True,
                    help="Location to dump the models.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=SGD)")
parser.add_argument("--learning_rate", "-lr", default=0.1, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def init_model(vocab_sizes, use_cuda):
    nmt = NMT(vocab_sizes, use_cuda)

    model.load_state("intermediate_ds_new", nmt)  # TODO Remove this
    #model.load_state(os.path.join('data', 'model.param'), nmt, generator)  # TODO Uncomment this
    return nmt


def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  """src_train = get_lm_input(src_train)
  src_dev = get_lm_input(src_dev)
  src_test = get_lm_input(src_test)

  trg_train = get_lm_output(trg_train)
  trg_dev = get_lm_output(trg_dev)
  trg_test = get_lm_output(trg_test)"""

  batched_train_src, batched_train_src_mask, sort_index = tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
  batched_train_trg, batched_train_trg_mask = tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  batched_dev_src, batched_dev_src_mask, sort_index = tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_trg, batched_dev_trg_mask = tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)

  (src_vocab_size, trg_vocab_size) = len(src_vocab), len(trg_vocab)

  nmt = NMT((src_vocab_size, trg_vocab_size), use_cuda) # TODO: add more arguments as necessary
  #nmt = init_model((src_vocab_size, trg_vocab_size), use_cuda)
  if use_cuda > 0:
    nmt.cuda()
  else:
    nmt.cpu()

  criterion = torch.nn.NLLLoss()
  optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

  # main training loop
  last_dev_avg_loss = float("inf")
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range
    for i, batch_i in enumerate(rand.srange(len(batched_train_src))):
      #if random.random() > 0.5:
      if False: #i > 1500:  # TODO REMOVE THIS !!!!!!!!!!!!!!!!!!!!
        #model.save("intermediate_ds_new", nmt);
        break;
      if i % 200 == 0:
        model.save("intermediate_ds", nmt);
        pass

      train_src_batch = Variable(batched_train_src[batch_i])  # of size (src_seq_len, batch_size)
      train_trg_batch = Variable(batched_train_trg[batch_i])  # of size (src_seq_len, batch_size)
      train_src_mask = Variable(batched_train_src_mask[batch_i])
      train_trg_mask = Variable(batched_train_trg_mask[batch_i])
      if use_cuda:
        train_src_batch = train_src_batch.cuda()
        train_trg_batch = train_trg_batch.cuda()
        train_src_mask = train_src_mask.cuda()
        train_trg_mask = train_trg_mask.cuda()

      sys_out_batch, translated_sentence_wd_index = nmt(train_src_batch, train_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary
      train_trg_mask = train_trg_mask.view(-1)
      train_trg_batch = train_trg_batch.view(-1)
      train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
      train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
      loss = criterion(sys_out_batch, train_trg_batch)
      logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    for batch_i in range(len(batched_dev_src)):
      #if random.random() > 0.5:
      if False: #i > 1500:  # TODO REMOVE THIS !!!!!!!!!!!!!!!!!!!!
        #model.save("intermediate_ds_new", nmt);
        break;
      dev_src_batch = Variable(batched_dev_src[batch_i], volatile=True)
      dev_trg_batch = Variable(batched_dev_trg[batch_i], volatile=True)
      dev_src_mask = Variable(batched_dev_src_mask[batch_i], volatile=True)
      dev_trg_mask = Variable(batched_dev_trg_mask[batch_i], volatile=True)
      if use_cuda:
        dev_src_batch = dev_src_batch.cuda()
        dev_trg_batch = dev_trg_batch.cuda()
        dev_src_mask = dev_src_mask.cuda()
        dev_trg_mask = dev_trg_mask.cuda()

      sys_out_batch, translated_sentence_wd_index = nmt(dev_src_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary
      actual_trg_max_len = dev_trg_mask.data.shape[0]
      dev_trg_mask = dev_trg_mask.view(-1)
      dev_trg_batch = dev_trg_batch.view(-1)
      dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
      dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size)

      # TODO Remove this!!!!!!
      predicted_trg_max_len = sys_out_batch.data.shape[0]
      if actual_trg_max_len > predicted_trg_max_len:
        sys_out_batch = torch.cat((sys_out_batch, torch.ones((actual_trg_max_len-predicted_trg_max_len,options.batch_size,trg_vocab_size))))
      else:
        sys_out_batch = sys_out_batch[0:actual_trg_max_len]
      # TODO Remove this ^^^ !!!!!!

      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)
      loss = criterion(sys_out_batch, dev_trg_batch)
      logging.debug("dev loss at batch {0}: {1}".format(batch_i, loss.data[0]))
      dev_loss += loss
    #if True: break
    dev_avg_loss = dev_loss / len(batched_dev_src)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))

    if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
      logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
      break
    torch.save(nmt, open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    last_dev_avg_loss = dev_avg_loss

  import datetime
  current_dt_tm = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  model.save("%s_%s"%(options.model_file, current_dt_tm), nmt);


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
