import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from collections import namedtuple

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data', default='/Users/mac/PycharmProjects/DialogSystem/ubuntu_valid.tsv')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data', default='/Users/mac/PycharmProjects/DialogSystem/ubuntu_valid.tsv')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--save_file', action='store_true', dest='save_file',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--concept', default=False,
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    cpt = SourceField()
    max_len = 200

    fields = [('src', src), ('tgt', tgt)]
    if opt.concept:
        fields.append(('cpt', cpt))

    def len_filter(example):
        try:
            return len(example.src) <= max_len and len(example.tgt) <= max_len
        except:
            print(example.src)
            exit()
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=fields,
        filter_pred=None
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=fields,
        filter_pred=None
    )
    # this is for the bug of UBUNTU corpus
    example_train = []
    example_dev = []
    for example in train.examples:
        try:
            a = example.src
            b = example.tgt
            if len_filter(example):
                example_train.append(example)
        except:
            continue
        #if len(example) == len(fields):
    for example in dev.examples:
        try:
            a = example.src
            b = example.tgt
            if len_filter(example):
                example_dev.append(example)
        except:
            continue
    train.examples = example_train
    dev.examples = example_dev

    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    if opt.concept:
        cpt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = NLLLoss(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        dialog_hidden_size = 128
        dropout = 0.5
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id, embedding=hidden_size, use_concept=opt.concept)
        dialog_encoder = torch.nn.LSTM(input_size=hidden_size*2 if bidirectional else hidden_size,
                                       hidden_size=dialog_hidden_size, batch_first=True, dropout=dropout)
        if opt.concept:
            seq2seq = Seq2seq(encoder, decoder, dialog_encoder=dialog_encoder, cpt_vocab=cpt.vocab, hidden_size=dialog_hidden_size)
        else:
            seq2seq = Seq2seq(encoder, decoder, dialog_encoder=dialog_encoder, hidden_size=dialog_hidden_size)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=1, expt_dir=opt.expt_dir)

    if opt.concept:
        VOCAB = namedtuple('vocabs', ("src_vocab", "tgt_vocab", "cpt_vocab"))
        vocabs = VOCAB(src.vocab, tgt.vocab, cpt.vocab)
    else:
        vocabs = []

    seq2seq = t.train(seq2seq, train,
                      num_epochs=30, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume,
                      use_concept=opt.concept,
                      vocabs=vocabs,
                      save_file=opt.save_file)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
