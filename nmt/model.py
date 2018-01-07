import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np

class NMT(nn.Module):

    def __init__(self, vocab_sizes, use_cuda):
        super(NMT, self).__init__()

        self.hidden_size = 512, 1024  # encoder & decoder hidden sizes
        self.emb_sizes = 300, 300  # encoder & decoder embed sizes

        self.SOS_TOKEN = 2  # TODO Parameterize this
        self.EOS_TOKEN = 3  # TODO Parameterize this
        self.BLANK_TOKEN = 1  # TODO Parameterize this
        self.teacher_forcing_ratio = 0.5  # TODO Change this to enable self-learning during training

        dropout_p = 0.05

        self.sent_encoder = SentenceEncoder(self.emb_sizes[0], vocab_sizes[0], self.hidden_size[0], use_cuda)

        self.word_decoder = WordDecoder(self.emb_sizes[1], vocab_sizes[1], self.hidden_size[1], use_cuda, dropout_p)
        self.use_cuda = use_cuda

        # Move models to GPU
        if self.use_cuda:
            self.sent_encoder.cuda()
            self.word_decoder.cuda()

        self.o = 0

    def forward(self, *nmt_args):
        src = nmt_args[0]

        # Run words through encoder
        encoder_outputs, encoder_hidden = self.sent_encoder(src)  # encoder_out: [seq, batch, 2* enc_hidden]; hidden_state: [1, batch, 2 * enc_hidden]

        # Initialize decoder input, hidden and context variables
        # decoder_input = Variable(torch.LongTensor([[self.SOS_TOKEN]]))  # TODO Revisit this
        batch_size = encoder_outputs.data.shape[1]
        decoder_input = Variable(torch.ones(1, batch_size)).long() * self.SOS_TOKEN

        decoder_context = Variable(torch.zeros(1, batch_size, self.word_decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        translated_sentence_wd_index = []  # TODO remove this.
        seq_batch_wd_prob = []

        if nmt_args.__len__() > 1:  # Train
            target = nmt_args[1]

            # Get size of input and target sentences
            target_length = target.data.shape[0]

            # Choose whether to use teacher forcing
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio
            if use_teacher_forcing:
                # Teacher forcing: Use the ground-truth target as the next input
                for di in range(target_length):
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = self.word_decoder(decoder_input,
                                                                                                 decoder_context,
                                                                                                 decoder_hidden,
                                                                                                 encoder_outputs)
                    decoder_input = target[di].unsqueeze(0)  # Next target is next input
                    seq_batch_wd_prob.append(decoder_output)

                    batch_most_prob_wd_id = decoder_output.data.topk(1, 1, largest=True)[1].t()
                    translated_sentence_wd_index.append(batch_most_prob_wd_id.squeeze(0))

            else:
                # Without teacher forcing: use network's own prediction as the next input
                # TODO Revisit this
                """max_src_sentence_length = encoder_outputs.data.shape[0]
                max_tgt_sentence_length = 3 * max_src_sentence_length"""

                incomplete_sentence_flag = torch.ones(1, batch_size)

                for di in range(target_length): # TODO Consider changing this to max_tgt_sentence_length. Set to target length to avoid issues in train mask
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = self.word_decoder(decoder_input,
                                                                                                 decoder_context,
                                                                                                 decoder_hidden,
                                                                                                 encoder_outputs)

                    seq_batch_wd_prob.append(decoder_output)

                    # Choose top word from output

                    batch_most_prob_wd_id = decoder_output.data.topk(1, 1, largest=True)[1].t()
                    #batch_most_prob_wd_id *= incomplete_sentence_flag
                    translated_sentence_wd_index.append(batch_most_prob_wd_id.squeeze(0))
		    #print(type(batch_most_prob_wd_id == self.EOS_TOKEN))
                    incomplete_sentence_flag[(batch_most_prob_wd_id == self.EOS_TOKEN).cpu()] = 0

                    """if incomplete_sentence_flag.eq(0).all():
                        break;"""

                    decoder_input = Variable(batch_most_prob_wd_id)
                    if self.use_cuda:
                        decoder_input=decoder_input.cuda()

            translated_sentence_wd_index = torch.stack(translated_sentence_wd_index)
            seq_batch_wd_prob = torch.stack(seq_batch_wd_prob)


        else:  # Inference
            # TODO Revisit this
            max_src_sentence_length = encoder_outputs.data.shape[0]
            max_tgt_sentence_length = 1.5 * max_src_sentence_length+3
	    if self.use_cuda:	
            	incomplete_sentence_flag = torch.ones(1, batch_size).long().cuda() #long()
	    else:
		incomplete_sentence_flag = torch.ones(1,batch_size).long()	    
#incomplete_sentence_flag = [1]*batch_size
            # Run through decoder
            for tgt_sent_length in range(int(max_tgt_sentence_length)):

                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.word_decoder(decoder_input,
                                                                                             decoder_context,
                                                                                             decoder_hidden,
                                                                                             encoder_outputs)
                seq_batch_wd_prob.append(decoder_output)

                # Choose top word from output

                batch_most_prob_wd_id = decoder_output.data.topk(1, 1, largest=True)[1].t()

		#print('Type batch_most_prob_wd_id: ',type(batch_most_prob_wd_id),' ',type(incomplete_sentence_flag))
                batch_most_prob_wd_id = batch_most_prob_wd_id*incomplete_sentence_flag
                translated_sentence_wd_index.append(batch_most_prob_wd_id.squeeze(0))

                incomplete_sentence_flag[batch_most_prob_wd_id == self.EOS_TOKEN] = 0
                if incomplete_sentence_flag.eq(0).all():
                    break;

                decoder_input = Variable(batch_most_prob_wd_id)
                if self.use_cuda:
                    decoder_input=decoder_input.cuda()

            translated_sentence_wd_index = torch.stack(translated_sentence_wd_index)
            seq_batch_wd_prob = torch.stack(seq_batch_wd_prob)

            # TODO Revisit this
            if tgt_sent_length > max_tgt_sentence_length:
                print(
                    "Discontinuing translation because target sentence length %i exceeded twice the max. length of src sentence" % (
                        tgt_sent_length))

        return seq_batch_wd_prob, translated_sentence_wd_index


class SentenceEncoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, use_cuda):
        super(SentenceEncoder, self).__init__()
        self.rnn = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.embeddings = Embeddings(vocab_size, embed_size)
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

    def forward(self, input):
        embedding = self.embeddings(input)

        batch_size = embedding.data.shape[1]
        init_hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if self.use_cuda:
            init_hidden = init_hidden.cuda()

        encoder_out, hidden_state = self.rnn(embedding, init_hidden)
        bi_dir_hidden_state = torch.cat([hidden_state[0:1], hidden_state[1:2]], 2)
        return encoder_out, bi_dir_hidden_state  # TODO In the case of stacked encoder, pass hidden and cell state of the top layer for the final time-step


class WordDecoder(nn.Module):
    def __init__(self, embedd_size, vocab_size, hidden_size, use_cuda, dropout_p):
        super(WordDecoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, embedd_size)
        self.gru = nn.GRU(embedd_size + hidden_size, hidden_size, dropout=dropout_p)
        self.use_cuda = use_cuda
        self.attn = Word_Attention(hidden_size, use_cuda)
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size

    #def forward(self, encoder_outputs, hidden_state, cell_state, generator, tgt_sent=None):
    def forward(self, word_input_1b, last_context_1bh, last_hidden_1bh, encoder_outputs_sbh):
        # TODO Revisit this
        word_embedded_1be = self.embeddings(word_input_1b)

        # Combine embedded input word and last context, run through RNN
        # rnn_input_sbi size: 1, batch, (input_size = embed size + context(hidden) size)
        rnn_input_1bi = torch.cat((word_embedded_1be, last_context_1bh), 2)  # TODO Revisit this
        rnn_output_1bh, hidden_1bh = self.gru(rnn_input_1bi, last_hidden_1bh)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights_b1s = self.attn(rnn_output_1bh, encoder_outputs_sbh).unsqueeze(1)
        context_b1h = attn_weights_b1s.bmm(encoder_outputs_sbh.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        context_1bh = context_b1h.transpose(0,1)  # B x S=1 x N -> B x N
        lin_out = self.out(torch.cat((rnn_output_1bh, context_1bh), 2))
        output_bv = nn.functional.log_softmax(lin_out.squeeze(0))

        # Return final output, hidden state, and attention weights (for visualization)
        return output_bv, context_1bh, hidden_1bh, attn_weights_b1s.squeeze(1)


class Word_Attention(nn.Module):
    def __init__(self, hidden_size, use_cuda):
        super(Word_Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.use_cuda = use_cuda

    def forward(self, hidden_1bh, encoder_outputs_sbh):
        """seq_len = len(encoder_outputs_sbh)
        batch_size = hidden_1bh.data.shape[1]

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, 1, seq_len))  # B x 1 x S TODO Revisit this
        if self.use_cuda: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            energy = self.attn(encoder_outputs_sbh[i,0:1,:])
        attn_energies[i] = hidden_1bh.dot(energy)"""

        _, t_batch, t_dim = hidden_1bh.size()
        hidden_1bh = self.attn(hidden_1bh)
        hidden_bh1 = hidden_1bh.view(t_batch, t_dim, 1)  
        encoder_outputs_bsh = encoder_outputs_sbh.transpose(0, 1)
        attn_energies_bs1 = torch.bmm(encoder_outputs_bsh, hidden_bh1)

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return nn.functional.softmax(attn_energies_bs1.squeeze(2))

'''
Embeddings and features
'''

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedd_size):
        super(Embeddings, self).__init__()
        self.emb_luts = nn.Sequential(nn.Embedding(vocab_size, embedd_size))
        # print(vocab_size, embedd_size)

    def forward(self, input):
        return self.emb_luts(input)


def save(name, model):
    save_state = model.state_dict().copy()
    print(sorted(save_state.keys()))
    torch.save(save_state, name)


def load_state(filename, model):
    state = torch.load(filename)
    model.load_state_dict(state)
