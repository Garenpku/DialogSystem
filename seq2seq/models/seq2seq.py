import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, dialog_encoder=None, decode_function=F.log_softmax, cpt_vocab=None,
                 hidden_size=128,
                 mid_size=64, dialog_hidden=128):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dialog_encoder = dialog_encoder
        self.decode_function = decode_function
        self.cpt_vocab = cpt_vocab
        if self.cpt_vocab:
            self.cpt_embedding = nn.Embedding(len(cpt_vocab.itos), hidden_size)
        self.layer_u = torch.nn.Linear(hidden_size * 2, mid_size)
        self.layer_c = torch.nn.Linear(dialog_hidden, mid_size)
        self.layer_e = torch.nn.Linear(hidden_size, mid_size)
        self.layer_att = torch.nn.Linear(mid_size, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.forget_u = torch.nn.Linear(hidden_size * 2, mid_size, bias=False)
        self.forget_c = torch.nn.Linear(dialog_hidden, mid_size, bias=False)
        self.forget_o = torch.nn.Linear(hidden_size, mid_size, bias=False)
        self.forget = torch.nn.Linear(mid_size, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden = hidden_size

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def extract_per_utt(self, input_variable, encoder_outputs, eou_index):
        input_index = input_variable.numpy() if not torch.cuda.is_available() else input_variable.cpu().numpy()
        eou_pos = [np.where(line == eou_index)[0] for line in input_index]
        utt_hidden = [torch.cat([encoder_outputs[j][i].unsqueeze(0) for i in eou_pos[j]], 0) for j in
                      range(input_variable.shape[0])]
        max_num_utt = max([len(line) for line in utt_hidden])
        for i in range(input_variable.shape[0]):
            if torch.cuda.is_available():
                utt_hidden[i] = torch.cat(
                    [utt_hidden[i], torch.zeros([max_num_utt - len(utt_hidden[i]), len(utt_hidden[0][0])]).cuda()])
            else:
                utt_hidden[i] = torch.cat(
                    [utt_hidden[i], torch.zeros([max_num_utt - len(utt_hidden[i]), len(utt_hidden[0][0])])])
        utt_hidden = [line.unsqueeze(0) for line in utt_hidden]
        return torch.cat(utt_hidden, 0), eou_pos

    # return size: batch * num_sentence * num_concept_per_sentence * embedding
    def concept_mapping(self, concept, vocab):
        pad_index = vocab.stoi['<pad>']
        eou_index = vocab.stoi['<EOU>']
        np_concept = concept.numpy() if not torch.cuda.is_available() else concept.cpu().numpy()
        end_pos = []
        for line in np_concept:
            pos = np.where(line == pad_index)[0]
            if len(pos):
                end_pos.append(pos[0])
            else:
                end_pos.append(len(line))
        np_concept = [np_concept[i][:end_pos[i]] for i in range(len(np_concept))]
        concept_batch = []
        embedding_batch = []
        for i in range(len(np_concept)):
            concept_d = []
            utt_pos = np.where(np_concept[i] == eou_index)[0]
            utt_pos = np.concatenate([[-1], utt_pos])
            for j in range(1, len(utt_pos)):
                concept_d.append(np_concept[i][utt_pos[j - 1] + 1:utt_pos[j]])
            if torch.cuda.is_available():
                concept_mapped = [self.cpt_embedding(torch.tensor(line).cuda()) for line in concept_d]
            else:
                concept_mapped = [self.cpt_embedding(torch.tensor(line)) for line in concept_d]
            concept_batch.append(concept_d)
            embedding_batch.append(concept_mapped)
        return concept_batch, embedding_batch

    def state_track(self, concept, embedding, dialog, utterance):

        max_sentence = max([len(line) for line in embedding])
        one = torch.ones((1, self.hidden))
        batch_size = len(concept)
        g = torch.ones([batch_size, 1])
        if torch.cuda.is_available():
            one = one.cuda()
            g = g.cuda()
        concept = [[list(line) for line in sample] for sample in concept]

        # batch padding
        for j in range(batch_size):
            if len(embedding[j]) < max_sentence:
                embedding[j].extend((max_sentence - len(embedding[j])) * [one])
                for k in range(max_sentence - len(concept[j])):
                    concept[j].append(['<pad>'])
        for i in range(max_sentence):
            max_concepts = 0
            for j in range(batch_size):
                num = embedding[j][i].shape[0]
                max_concepts = max_concepts if max_concepts >= num else num
            for j in range(batch_size):
                num = embedding[j][i].shape[0]
                if num < max_concepts:
                    embedding[j][i] = torch.cat([embedding[j][i], torch.cat([one] * (max_concepts - num))])
                    concept[j][i].extend((max_concepts - num) * ['<pad>'])
        embedding_per_step = []
        for i in range(max_sentence):
            emb = []
            for j in range(batch_size):
                emb.append(embedding[j][i].unsqueeze(0))
            embedding_per_step.append(torch.cat(emb, 0))

        # calculating state
        for i in range(max_sentence):
            c = dialog[:, i-1] if i != 0 else torch.zeros_like(dialog[:, 0])
            u = utterance[:, i]
            cpt = embedding_per_step[i]
            res_u = self.layer_u(u).unsqueeze(1)
            res_c = self.layer_c(c).unsqueeze(1)
            res_e = self.layer_e(cpt)
            distribution = self.softmax(self.layer_att(res_u + res_c + res_e).reshape(batch_size, -1))
            o = torch.bmm(distribution.unsqueeze(1), cpt).squeeze()
            res_f_u = self.forget_u(u)
            res_f_c = self.forget_c(c)
            res_f_o = self.forget_o(o)
            if i != 0:
                g = self.sigmoid(self.forget(res_f_c + res_f_u + res_f_o))
                state = torch.cat([state * g, distribution * (1 - g)], 1)
            else:
                state = distribution

        # filtered state template
        concept_linear = []
        embedding_linear = []
        dict_linear = []
        states = []
        for k in range(batch_size):
            i_to_concept = []
            i_to_embedding = []
            concept_to_i = {}
            index = 0
            for i in range(len(concept[k])):
                if not len(concept[i]):
                    continue
                for cnt, cpt in enumerate(concept[k][i]):
                    if cpt not in i_to_concept:
                        i_to_concept.append(cpt)
                        i_to_embedding.append(embedding[k][i][cnt].unsqueeze(0))
                        concept_to_i[cpt] = index
                        index += 1
            i_to_embedding = torch.cat(i_to_embedding, 0)
            concept_linear.append(i_to_concept)
            embedding_linear.append(i_to_embedding)
            dict_linear.append(concept_to_i)
            prob_dist = torch.zeros((len(i_to_concept)))
            if torch.cuda.is_available():
                prob_dist = prob_dist.cuda()
            states.append(prob_dist)

        # generate final state
        for i in range(batch_size):
            cnt = 0
            cpt_dict = dict_linear[i]
            for j in range(len(concept[i])):
                for k, cpt in enumerate(concept[i][j]):
                    if cpt != '<pad>':
                        states[i][cpt_dict[cpt]] += state[i][cnt]
                    cnt += 1
            assert cnt == state.shape[1]

        concept_rep = []
        for i in range(batch_size):
            concept_rep.append(torch.mm(states[i].unsqueeze(0), embedding_linear[i]))

        return states, concept_linear, embedding_linear, concept_rep

    def single_turn_state_track(self, concept_batch, embedding_batch, dialog):
        concept_linear = []
        emb_linear = []
        zero = torch.zeros((1, self.hidden))
        if torch.cuda.is_available():
            zero = zero.cuda()
        for i in range(len(concept_batch)):
            res = []
            emb = []
            for j in range(len(concept_batch[i])):
                res.extend(list(concept_batch[i][j]))
                emb.extend(embedding_batch[i][j].unsqueeze(0))
            if len(emb) != 0:
                emb = torch.cat(emb, 0)
            else:
                emb = zero
            concept_linear.append(res)
            emb_linear.append(emb)
        max_len = max([len(line) for line in concept_linear])
        for i in range(len(concept_batch)):
            if len(emb_linear[i]) < max_len:
                tmp = torch.cat(((max_len - len(emb_linear[i])) * [zero]))
                # print(tmp.shape)
                # print(emb_linear[i].shape)
                emb_linear[i] = torch.cat([emb_linear[i], tmp]).unsqueeze(0)
            else:
                emb_linear[i] = emb_linear[i].unsqueeze(0)
        emb_linear = torch.cat(emb_linear, 0)
        c = dialog[:, -1]
        res_c = self.layer_c(c)
        res_c = res_c.reshape(res_c.shape[0], 1, res_c.shape[-1])
        res_e = self.layer_e(emb_linear)
        #res = self.softmax(self.layer_att(res_e + res_c).reshape(emb_linear.shape[0], emb_linear.shape[1]))
        res = self.layer_att(res_e + res_c).squeeze()
        s = torch.sum(res, dim=-1).reshape((-1, 1))
        res /= s
        o = torch.bmm(res.unsqueeze(1), emb_linear).squeeze()

        """
        replace = torch.zeros_like(res)
        states = []
        i_to_concept = []
        i_to_embedding = []
        for i in range(len(concept_batch)):
            state = {}
            tmp_concept = []
            tmp_embedding = []
            for j in range(len(concept_linear[i])):
                concept = concept_linear[i][j]
                if concept not in state:
                    state[concept] = res[i][j]
                else:
                    state[concept] = state[concept] + res[i][j]
                if concept not in tmp_concept:
                    tmp_concept.append(concept)
                    tmp_embedding.append(emb_linear[i][j].unsqueeze(0))
            tmp_embedding = torch.cat(tmp_embedding)
            i_to_concept.append(tmp_concept)
            i_to_embedding.append(tmp_embedding)
            states.append(state)
        """
        #return states, i_to_concept, i_to_embedding, o
        return o

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, concept=None, vocabs=None, use_concept=False):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        if use_concept:
            src_vocab = vocabs.src_vocab
            tgt_vocab = vocabs.tgt_vocab
            cpt_vocab = vocabs.cpt_vocab
            eou_index = src_vocab.stoi['<EOU>']
            utt_hidden, eou_pos = self.extract_per_utt(input_variable, encoder_outputs, eou_index)
            dialog_output, (context, _) = self.dialog_encoder(utt_hidden)
            concept_batch, embedding_batch = self.concept_mapping(concept, cpt_vocab)
            batch_state, batch_concepts, batch_embeddings, o = self.state_track(concept_batch, embedding_batch, dialog_output, utt_hidden)
            #batch_state, batch_concepts, batch_embeddings, o = self.single_turn_state_track(concept_batch, embedding_batch, dialog_output)
            #o = self.single_turn_state_track(concept_batch, embedding_batch, dialog_output)
            o = torch.cat(o).unsqueeze(1)
            result = self.decoder(inputs=target_variable,
                                  encoder_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  function=self.decode_function,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  batch_state=batch_state,
                                  batch_concepts=batch_concepts,
                                  batch_embeddings=batch_embeddings,
                                  context=context.squeeze(),
                                  cpt_vocab=cpt_vocab,
                                  tgt_vocab=tgt_vocab,
                                  use_copy=use_concept,
                                  concept_rep=o)
        else:
            result = self.decoder(inputs=target_variable,
                                  encoder_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  function=self.decode_function,
                                  teacher_forcing_ratio=teacher_forcing_ratio)
        return result
