import argparse
import logging

import torch
from torch import cuda
from torch.autograd import Variable

import model as model
from utils import tensor

from model import NMT

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="NMT Translator")
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
parser.add_argument("--translation_file", default="translation_file.txt",
                    help="translation_file. (default=translation_file.txt)")
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

def init_model(vocab_sizes, use_cuda, model_file_name):
    nmt = NMT(vocab_sizes, use_cuda)

    model.load_state(model_file_name, nmt)
    return nmt


def get_translated_sentence(translated_sentence_wd_index, vocabList):
  translated_sentence = []
  for i in range(translated_sentence_wd_index.shape[1]):
    translated_sentence.append(' '.join([vocabList[j] for j in translated_sentence_wd_index[:,i] if j not in (0, 2, 3)]))
  return (translated_sentence)


def main(options):

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
    trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

    model_file_name = options.model_file

    translation_file = open(options.translation_file, 'w')

    batched_test_src, batched_test_src_mask, sort_index = tensor.advanced_batchize(src_test, options.batch_size, src_vocab.stoi["<blank>"])

    (src_vocab_size, trg_vocab_size) = len(src_vocab), len(trg_vocab)

    nmt = init_model((src_vocab_size, trg_vocab_size), use_cuda, model_file_name)
    if use_cuda > 0:
        nmt.cuda()
    else:
        nmt.cpu()

    batch_translated_sentences = []
    for batch_i in range(len(batched_test_src)):
        dev_src_batch = Variable(batched_test_src[batch_i], volatile=True)
        if use_cuda:
            dev_src_batch = dev_src_batch.cuda()

        sys_out_batch, translated_sentence_wd_index = nmt(dev_src_batch)
        translated_sent = get_translated_sentence(translated_sentence_wd_index, trg_vocab.itos)
        batch_translated_sentences.append(translated_sent)

    for translated_sentences in batch_translated_sentences:
        for translated_sentence in translated_sentences:
            translation_file.write("%s\n" % translated_sentence)
    print "done"



if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
