import argparse
import dill
import logging

import torch
from torch.autograd import Variable
from torch import cuda

from model_final import RNNLM, BiRNNLM, BiGRULM


parser = argparse.ArgumentParser(description="Find the missing words using the given model")
parser.add_argument("--data_file", required=True,
                    help="File for training set.")
parser.add_argument("--model_file1", required=True,
                    help="File for model to use.")
parser.add_argument("--model_file2", required=False,
                    help="File for model to use.")
parser.add_argument("--result_file", default="results/results_dump.txt")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


def main(options):
    # use_cuda = (len(options.gpuid) >= 1)
    use_cuda = False
    if options.gpuid:
      use_cuda = True
      cuda.set_device(options.gpuid[0])

    train, dev, test, vocab = torch.load(open(options.data_file, 'rb'), pickle_module=dill)
    rnnlm = torch.load(open(options.model_file1, 'rb'), pickle_module=dill)
    rnnlm1 = BiGRULM(len(vocab))
    if use_cuda:
      rnnlm1.cuda()
    rnnlm1.load_state_dict(rnnlm)

    vocab_size = len(vocab)
    f_ = open(options.result_file, 'wb')
    for sentence_i in test:
        sent_tensor = torch.ones((len(sentence_i), 1)).long()
        sent_tensor[:, 0] = sentence_i
        if use_cuda:
          sent_tensor = sent_tensor.cuda()
        sys_out1 = rnnlm1(Variable(sent_tensor))
        sys_out = sys_out1  # + sys_out2

        answer = []
        full_sentence = []
        original_sentence = []
        for j, word in enumerate(sentence_i):
            original_sentence.append(vocab.itos[word])
            if word == vocab.stoi["<blank>"]:
                idx = sys_out[j, :][0, :].data.max(0)[1][0]
                word = vocab.itos[idx]
                while word in ("<unk>", "<pad>"):
                    new_vec = sys_out[j, :][0, :].data
                    new_vec[idx] = float('-10000')
                    idx = new_vec.max(0)[1][0]
                    word = vocab.itos[idx]

                answer.append(word)
                full_sentence.append(word)
            else:
                full_sentence.append(vocab.itos[word])
        f_.write(" ".join(answer).encode('utf-8').strip())
        f_.write("\n".encode('utf-8'))
    f_.close()


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
