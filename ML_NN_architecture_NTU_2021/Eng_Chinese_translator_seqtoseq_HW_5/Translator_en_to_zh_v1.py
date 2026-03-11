'''
Propose:
Train a Transformer encoder decoder network for translating English to Chinese. 
The network is trained on the NTU course dataset, which original from ted2020.
The input to the network is the English sentence, and the output is the corresponding Chinese sentence. 
The network is trained using soften-cross-entropy loss, and the performance is evaluated using BLEU score.
Worth noting that the dataset has been preprocessed and tokenized, and has been split into training and testing sets.
(And they are binary files.)
For decoder training, we use teacher forcing,
which means that the input to the decoder is the ground truth token.
Author: Dr. Ka Hou Leong
Date: 5/3/2026
Version: 0.1
ML library: PyTorch
'''
import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils

import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


config = Namespace(
    datadir = "./project_data_binary",
    savedir = "./checkpoints/transformer",
    source_lang = "en",
    target_lang = "zh",
    run_in_background = False, # make tqdm to be silent mode
    show_model_summary = False, # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
    allow_device = True,  # Set to False if you want to force using CPU.
    use_pin_memory = False,  # Set to True if you use GPU. False for CPU and MPS.

    # cpu threads when fetching & processing data.
    num_workers=0,
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,

    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,

    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,

    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,

    # beam size for beam search
    beam=5,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2,
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols.
    post_process = "sentencepiece",

    # checkpoints
    keep_last_epochs=5,
    resume=None, # if resume from checkpoint name (under config.savedir)

    # logging
    #use_wandb=False,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
'''
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)
'''

# Automatically choose the device to use.
if config.allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_env = utils.CudaEnvironment()
        utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
        config.num_workers = 0 # MPS does not support multi-process data loading, so we set num_workers to 0.
        config.use_pin_memory = False # MPS does not support pin_memory, so we set it to False.
    else:
        device = torch.device("cpu")
        print("Using CPU")
        config.use_pin_memory = False
        config.num_workers = 0 # You do not multiple process to move data when using CPU, so we set num_workers to 0.
else:
    device = torch.device("cpu")
    print("Using CPU")
    config.use_pin_memory = False
    config.num_workers = 0 # You do not multiple process to move data when using CPU, so we set num_workers to 0.

from fairseq.tasks.translation import TranslationConfig, TranslationTask

# use TranslationTask to load the binary dataset.
# also for dataloader and beam search
## setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap", # memory-mapped binary dataset
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)  # type: ignore

logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

# Print one sample to check consistancy
sample = task.dataset("valid")[1]
pprint.pprint(sample)
print('A')
pprint.pprint(
    "Source: " + \
    task.source_dictionary.string(
        sample['source'],
        config.post_process,
    )
)
pprint.pprint(
    "Target: " + \
    task.target_dictionary.string(
        sample['target'],
        config.post_process,
    )
)

# A helper function to move data to the appropriate device. It can handle tensors, dicts, lists and tuples.
def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x

# Define dataloader
# func: 
# ensure there are N token in each batch
# shuffling data
# filter out extremely long sentences
# padding to ensure the length of sentences is a constant 
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond
        # first call of this method has no effect.
    )
    return batch_iterator

demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=config.num_workers, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter) # type: ignore
pprint.pprint(sample)

# import fairseq's encoder and decoder base class for building our own encoder and decoder.
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.models.transformer import base_architecture
arch_args = Namespace(
    encoder_embed_dim=256, # dim for embedding (encoder and decoder can have different embedding dim)
    encoder_ffn_embed_dim=1024, # feedforward network hidden dim in the encoder
    encoder_layers=4, # number of encoder layers
    decoder_embed_dim=256, # dim for embedding (encoder and decoder can have different embedding dim)
    decoder_ffn_embed_dim=1024, # feedforward network hidden dim in the decoder
    decoder_layers=4, # number of decoder layers
    share_decoder_input_output_embed=True, # whether input and output word embedding in the decoder are shared. 
    dropout=0.3, # 
    encoder_attention_heads=4, # attention heads in the encoder
    encoder_normalize_before=True, # controls where Layer Normalization is applied inside each encoder layer
    decoder_attention_heads=4, # attention heads in the decoder
    decoder_normalize_before=True, # controls where Layer Normalization is applied inside each decoder layer
    activation_fn="relu",
    max_source_positions=1024, # maximum source sequence length
    max_target_positions=1024, # maximum target sequence length
)
# compile the default architecture settings for Transformer. 
base_architecture(arch_args)

class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens: bool = True): # type: ignore
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

def build_model(args, task):
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # Word embedding for encoder and decoder. The embedding layer converts the input token ids into dense vectors.
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())

    # Transformer encoder and decoder.
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)

    # initialize the model parameters. You can use other initialization methods if you like.
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

    # initialize the model parameters. You can use other initialization methods if you like.
    model.apply(init_params)
    return model


# Copy from NTU course. Not necessary to modify it.
# Feature: linearly growing learning rate in the warmup stage, 
# and then decaying learning rate proportional to the inverse square root of the step number.
class NoamOpt:
    "Optim wrapper that implements rate."
    # warmup: warmup steps
    # model_size: the d_model in the paper, which is the embedding dimension of the model (encoder in this project).
    # factor: proportionality factor for learning rate. You can tune the maximum learning rate by this factor.
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    # property is a python decorator that allows you to access the method like an attribute. For example, you can use optimizer.param_groups instead of optimizer.param_groups().
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

from fairseq.data import iterators

def train_one_epoch(epoch_itr, model, task, criterion, optimizer, config, device, accum_steps=1):

    itr = epoch_itr.next_epoch_itr(shuffle=True)
    # gradient accumulation: update model parameters after accum_steps batches.
    itr = iterators.GroupedIterator(itr, accum_steps) # every element in itr is a list containing accum_steps number of batches.
    epoch = epoch_itr.epoch


    pad_idx = task.target_dictionary.pad()

    training_loss = []
    number_of_tokens = [] # effective tokens only
    model.train()
    progress = tqdm.tqdm(itr, desc=f"Epoch {epoch}", disable=config.run_in_background)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0 #Number of effective tokens in these accum_steps batches (no padding)
        optimizer.zero_grad()
        for step, sample in enumerate(samples):
            '''
            # Used to check variable shapes
            print("ABC")
            print(sample["net_input"]["src_tokens"].shape)
            print(sample["nsentences"], sample["ntokens"])
            print("CBA")
            '''
            # move batch to device
            # sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            # record the number of effective tokens in the current batch (exclude padding tokens)
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            net = sample["net_input"]
            # move data to device
            src_tokens = net["src_tokens"].to(device)
            src_lengths = net["src_lengths"].to(device)
            prev_output_tokens = net["prev_output_tokens"].to(device)
            target = sample["target"].to(device)

            # forward pass
            logits, extra = model(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                prev_output_tokens=prev_output_tokens,
            )

            # flatten for CE loss
            # from [B, T, V] to [B*T, V]
            logits = logits.reshape(-1, logits.size(-1))
            # from [B, T] to [B*T]
            target = target.reshape(-1)

            loss = criterion(logits, target) * sample_size_i # total loss for all tokens in this batch.

            # normalize for gradient accumulation
            # loss = loss / accum_steps # accum_steps is a constant, it doesn't matter for training.
            accum_loss += loss.item() # total loss for all tokens in this batch 
            loss.backward()

        # total gradient for accum_steps batches -> average gradient
        optimizer.multiply_grads(1.0 / max(1, sample_size)) # sample_size is the total number of effective tokens in these accum_steps batches. We use it to normalize the gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)

        optimizer.step()
        optimizer.zero_grad()

        # logging
        training_loss.append(accum_loss) # total loss for all tokens in these accum_steps batches
        number_of_tokens.append(sample_size) # total effective tokens in these accum_steps batches
        mean_loss = accum_loss / sample_size # means loss per token in these accum_steps batches
        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(
            loss=f"{mean_loss:.4f}",
            lr = f"{current_lr:.5e}",
        )
    loss_print = np.sum(training_loss) / np.sum(number_of_tokens) # average loss per token for this epoch
    logger.info(f"average training loss (per token): {loss_print:.4f}")
    return loss_print

def decode(toks, dictionary):
    # Convert vector back to words.
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model, seq_generator):
    # sample: like a sentence 
    gen_out = seq_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # decode the sentence, and collect the source sentence, hypothesis sentence and reference sentence for evaluation.
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0: take the first one in the beam tree, which has the top score.
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return srcs, hyps, refs


import shutil
import sacrebleu

def validate(model, task, criterion, seq_generator):
    logger.info('begin validation')
    # No iterators.GroupedIterator here, so element in itr is a batch
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    number_of_tokens = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", disable=config.run_in_background)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            sample_size = sample["ntokens"]
            # sample is a dictionary containing the batch data, including "net_input" and "target".
            # move_to_device moves the batch data to the appropriate device. It can handle tensors, dicts, lists and tuples.
            # since sample is a dictinoary, move_to_device will recursively move all tensors in the dictionary to the appropriate device.
            sample = move_to_device(sample, device)
            # forward pass
            logits, extra =  model.forward(**sample["net_input"]) # type: ignore
            number_of_tokens.append(sample_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), sample["target"].reshape(-1)) # type: ignore
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss.item() * sample_size)  # total loss for all tokens in this batch

            # inference
            s, h, r = inference_step(sample, model, seq_generator)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = np.sum(stats["loss"]) / np.sum(number_of_tokens) # average loss per token for the validation set
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # calculate BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs


    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])

    # show bleu results
    logger.info(f"average validation loss (per token):\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats

def validate_and_save(model, task, criterion, optimizer, epoch, seq_generator, save=True):
    stats = validate(model, task, criterion, seq_generator)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score # type: ignore python trick to add a static variable to the function
            torch.save(check, savedir/f"checkpoint_best.pt")

        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")

model = build_model(arch_args, task)

if config.show_model_summary:
    logger.info(model)
    net = sample["net_input"]
    summary(model, input_data=[
        net["src_tokens"], # src_tokens
        net["src_lengths"], # src_lengths
        net["prev_output_tokens"], # prev_output_tokens
    ], col_names=["input_size", "output_size", "num_params", "trainable"], depth=3)
    print("Model summary shown. Please set 'show_model_summary' to False if you don't want to see the model summary.")
    logits, extra = model(
        src_tokens=net["src_tokens"],
        src_lengths=net["src_lengths"],
        prev_output_tokens=net["prev_output_tokens"],
    )
    print("Show how the batch data looks like and the model output shape. ")
    pprint.pprint(sample)
    print("Above is a sample batch data.")
    print(logits.shape)  # expect (B, T, V)
    print("The expect model output shape is (B, T, V), where B is the batch size, T is the target sequence length, and V is the target vocabulary size.")
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()

model = model.to(device) # Move the model to the appropriate device
# The sequence generator is used for beam search decoding during evaluation. 
sequence_generator = task.build_generator([model], config)

pad_idx = task.target_dictionary.pad() # Padding index is 1

# Define the loss function. We use cross-entropy loss with label smoothing and ignore the padding index.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=pad_idx).to(device)

optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim,
    factor=config.lr_factor,
    warmup=config.lr_warmup,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
try_load_checkpoint(model, optimizer, name=config.resume)
while epoch_itr.next_epoch_idx <= config.max_epoch:
    # train for one epoch
    train_one_epoch(epoch_itr, model, task, criterion, optimizer, config, device=device)
    stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch, seq_generator=sequence_generator, save=True)
    logger.info("end of epoch {}".format(epoch_itr.epoch))
    epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
