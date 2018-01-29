import torch
import torch.nn as nn
from torch.autograd import Variable
import math


def __logsumexp(value):
    m, _ = torch.max(value, dim=2, keepdim=True)
    value_ = value - m
    return m + torch.log(torch.sum(torch.exp(value_), dim=2, keepdim=True))


def log_softmax(a):
    return a - __logsumexp(a).expand_as(a)


'''
Unidirectional RNN Language Model
'''
class RNNLM(nn.Module):
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.recurrent_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __init__(self, vocab_size):
        super(RNNLM, self).__init__()
        self.embedding_size = 32
        self.recurrent_size = 16  ##size of the hidden layer.
        self.vocab_size = vocab_size
        # Create Randomly initialized input tensor
        self.embedding = nn.Parameter(torch.randn(vocab_size, self.embedding_size), requires_grad=True)
        # Create weights, there are 3 sets, embedding - recurrent, recurrent - output, recurrent - recurrent
        # stdv = 1.0/math.sqrt(self.recurrent_size)
        # weight btw the input layer/embedding-layer and hidden layer
        self.U = nn.Parameter(torch.randn(self.recurrent_size, self.embedding_size), requires_grad=True)
        self.U = nn.Parameter(torch.randn(self.recurrent_size, self.embedding_size), requires_grad=True)
        # weight btw the hidden layer and output
        self.V = nn.Parameter(torch.randn(self.vocab_size, self.recurrent_size), requires_grad=True)
        # weight of the hidden unit being passed.
        self.W = nn.Parameter(torch.randn(self.recurrent_size, self.recurrent_size), requires_grad=True)
        # Create bias terms
        self.bias_1 = nn.Parameter(torch.randn(self.recurrent_size), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.randn(self.recurrent_size), requires_grad=True)
        self.bias_3 = nn.Parameter(torch.randn(self.vocab_size), requires_grad=True)

        # Hidden state
        self.h = nn.Parameter(torch.randn(self.recurrent_size))
        self.reset_parameters()

    def forward(self, input_batch, train=1):
        # Get vectors from word_ids, size (S, B, E)
        input_vec = self.embedding[input_batch.data, :]

        # Step == seq_len
        batch_size = input_vec.size(1)
        input_size = input_vec.size(0)

        hidden_layer = Variable(torch.FloatTensor(input_size, batch_size, self.recurrent_size))
        h = torch.stack([self.h] * input_vec.size(1))

        for step in range(input_size):
            hidden_layer[step] = h
            i_h = torch.addmm(self.bias_1, input_vec[step], self.U.t())
            h_h = torch.addmm(self.bias_2, h, self.W.t())
            h = torch.tanh(i_h + h_h)

        h_out = hidden_layer.matmul(self.V.t())
        return log_softmax(h_out)


'''
BiDirectional RNN Language Model
'''
class BiRNNLM(nn.Module):
    def logsumexp(value, dim):
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value_ = value - m
        return m + torch.log(torch.sum(torch.exp(value_), dim=dim, keepdim=True))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.recurrent_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __init__(self, vocab_size):
        super(BiRNNLM, self).__init__()

        self.embedding_size = 32
        self.recurrent_size = 8  ##size of the hidden layer.
        self.vocab_size = vocab_size
        # Create Randomly initialized input tensor
        self.embedding = nn.Parameter(torch.randn(vocab_size, self.embedding_size), requires_grad=True)

        self.U_f = nn.Parameter(torch.randn(self.recurrent_size, self.embedding_size), requires_grad=True)
        self.W_f = nn.Parameter(torch.randn(self.recurrent_size, self.recurrent_size), requires_grad=True)

        self.U_b = nn.Parameter(torch.randn(self.recurrent_size, self.embedding_size), requires_grad=True)
        self.W_b = nn.Parameter(torch.randn(self.recurrent_size, self.recurrent_size), requires_grad=True)

        self.V = nn.Parameter(torch.randn(self.vocab_size, 2 * self.recurrent_size), requires_grad=True)

        # Create bias terms
        self.bias_1_f = nn.Parameter(torch.randn(self.recurrent_size), requires_grad=True)
        self.bias_2_f = nn.Parameter(torch.randn(self.recurrent_size), requires_grad=True)

        self.bias_1_b = nn.Parameter(torch.randn(self.recurrent_size), requires_grad=True)
        self.bias_2_b = nn.Parameter(torch.randn(self.recurrent_size), requires_grad=True)

        self.bias_3 = nn.Parameter(torch.randn(self.vocab_size), requires_grad=True)

        self.reset_parameters()

    def forward(self, input_batch):
        embeddings = self.embedding[input_batch.data, :]
        seq_length = input_batch.size()[0]
        batch_size = input_batch.size()[1]

        hidden_state = Variable(torch.zeros(batch_size, self.recurrent_size))
        forward_pass = Variable(torch.FloatTensor(seq_length, batch_size, self.recurrent_size))

        if embeddings.is_cuda:
            hidden_state = hidden_state.cuda()
            forward_pass = forward_pass.cuda()

        for i, embed in enumerate(embeddings):
            forward_pass[i] = hidden_state
            if i == seq_length - 1:
                break
            hi = embed.matmul(self.U_f.t()) + self.bias_1_f
            hh = hidden_state.matmul(self.W_f.t()) + self.bias_2_f
            hidden_state = torch.tanh(hi + hh)

        hidden_state = Variable(torch.zeros(batch_size, self.recurrent_size))
        backward_pass = Variable(torch.FloatTensor(seq_length, batch_size, self.recurrent_size))

        if embeddings.is_cuda:
            hidden_state = hidden_state.cuda()
            backward_pass = backward_pass.cuda()

        for i, embed in enumerate(reversed(embeddings)):
            # print(seq_length-t-1)
            backward_pass[seq_length - i - 1] = hidden_state
            if i == seq_length - 1:
                break
            hi = embed.matmul(self.U_b.t()) + self.bias_1_b
            hh = hidden_state.matmul(self.W_b.t()) + self.bias_2_b
            hidden_state = torch.tanh(hi + hh)

        total_h = torch.cat((forward_pass, backward_pass), 2)
        a = total_h.matmul(self.V.t())
        return log_softmax(a)


'''
BiDirectional GRU Language Model
'''

class BiGRULM(nn.Module):

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.recurrent_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __init__(self, vocab_size):
        super(BiGRULM, self).__init__()
        # enhanced architecture
        self.embedding_size = 512
        self.recurrent_size = 128
        self.vocab_size = vocab_size
        # trainable start state?

        # Create Randomly initialized input tensor
        self.embedding = nn.Parameter(torch.randn(vocab_size, self.embedding_size))

        # Create weights, there are 3 sets, embedding - recurrent, recurrent - recurrent, recurrent - output
        # Since the hidden units actually have 3 weights each for update/input gate, reset gate and new
        self.U = nn.Parameter(torch.randn(3 * self.recurrent_size, self.embedding_size))
        self.W = nn.Parameter(torch.randn(3 * self.recurrent_size, self.recurrent_size))

        self.U_b = nn.Parameter(torch.randn(3 * self.recurrent_size, self.embedding_size))
        self.W_b = nn.Parameter(torch.randn(3 * self.recurrent_size, self.recurrent_size))

        self.V = nn.Parameter(torch.randn(self.vocab_size, self.recurrent_size * 2))

        # Create bias terms
        self.bias_1 = nn.Parameter(torch.randn(3 * self.recurrent_size))
        self.bias_2 = nn.Parameter(torch.randn(3 * self.recurrent_size))

        self.bias_1_b = nn.Parameter(torch.randn(3 * self.recurrent_size))
        self.bias_2_b = nn.Parameter(torch.randn(3 * self.recurrent_size))

        self.reset_parameters()

    def forward(self, input_batch):
        # Get vectors from word_ids, size (S, B, E)
        embeddings = self.embedding[input_batch.data, :]
        seq_length = input_batch.size()[0]
        batch_size = input_batch.size()[1]

        hidden_state = Variable(torch.zeros(batch_size, self.recurrent_size))
        forward_pass = Variable(torch.FloatTensor(seq_length, batch_size, self.recurrent_size))

        if embeddings.is_cuda:
            hidden_state = hidden_state.cuda()
            forward_pass = forward_pass.cuda()

        # Forward layer
        for i, embed in enumerate(embeddings):
            forward_pass[i] = hidden_state
            if i == seq_length - 1:
                break
            # Fused ADD+MM
            gi = torch.addmm(self.bias_1, embed, self.U.t())
            gh = torch.addmm(self.bias_2, hidden_state, self.W.t())

            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            input_gate = torch.sigmoid(i_i + h_i)
            reset_gate = torch.sigmoid(i_r + h_r)

            new_gate = torch.tanh(i_n + reset_gate * h_n)
            hidden_state = new_gate + input_gate * (hidden_state - new_gate)

        hidden_state = Variable(torch.zeros(batch_size, self.recurrent_size))
        backward_pass = Variable(torch.FloatTensor(seq_length, batch_size, self.recurrent_size))

        if embeddings.is_cuda:
            hidden_state = hidden_state.cuda()
            backward_pass = backward_pass.cuda()

        for i, embed in enumerate(reversed(embeddings)):
            # print(seq_length-t-1)
            backward_pass[seq_length - i - 1] = hidden_state
            if i == seq_length - 1:
                break

            gi = torch.addmm(self.bias_1_b, embed, self.U_b.t())
            gh = torch.addmm(self.bias_2_b, hidden_state, self.W_b.t())

            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            input_gate = torch.sigmoid(i_i + h_i)
            reset_gate = torch.sigmoid(i_r + h_r)

            new_gate = torch.tanh(i_n + reset_gate * h_n)
            hidden_state = new_gate + input_gate * (hidden_state - new_gate)

        total_h = torch.cat((forward_pass, backward_pass), 2)
        # Bias

        return log_softmax(total_h.matmul(self.V.t()))
