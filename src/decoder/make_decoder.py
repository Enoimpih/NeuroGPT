#!/usr/bin/env python3
import torch

def make_decoder(
    architecture: str='GPT',
    num_hidden_layers: int = 4,
    embed_dim: int = 768,
    output_dim: int = 1024,
    num_attention_heads: int = 12,
    intermediate_dim_factor: int=4,
    n_positions: int = 512,
    hidden_activation: str='gelu_new',
    dropout: float = 0.1
    ) -> torch.nn.Module:
    """
    Make a decoder object.
    
    The decoder contains the core
    model architecture used for learning.

    Args:
    -----
    architecture: str
        The model architecture to use.
        One of: 'GPT', 'BERT', 'NetBERT', autoencoder',
        'PretrainedGPT', 'PretrainedBERT', 'LinearBaseline'.
    num_hidden_layers: int
        The number of hidden layers of the model.
        Does not apply to 'PretrainedGPT', 'PretrainedBERT', 
        'LinearBaseline'. 
        For 'autoencoder', num_hidden_layers represents 
        the number of hidden layers of the encoder and decoder
        model.
    embed_dim: int
        The dimension of the used embedding space (see src.embedder).
    output_dim: int
        The dimension of the output projection (needs to match
        in_dim of src.embedder for upstream learning).
    num_attention_heads: int
        The number of attention heads of transformer models. Does
        not apply to any other model architecture as well as the
        'PretrainedGPT' and 'PretrainedBERT' architectures.
    intermediate_dim_factor: int
        Scales feed-forward transformer layer dimension relative to '
        embed_dim: intermediate_dim_factor * embed_dim
    n_positions: int
        The maximum number of sequence elements that
        the model can handle (in sequence elements).
    hidden_activation: str
        Type of hidden activation of transformer layers
        One of 'gelu', 'gelu_new', 'relu', 'silu'.
        Does not apply to non-transformer models.
    dropout: float
        Dropout ratio for attendion heads and residual layers
        of transofmer models and between LSTM layers of 
        encoder / decoder parts of autoencoder models. 

    Core methods:
    -----
    forward(batch: Dict):
        Forward pass of the model, generates Dict containing
        predicted output seqeuences, given input batch
        (as generated by src.embedder.prep_batch).
    decode(outputs: Dict):
        Make decoding prediction, given outputs generated by
        caling forward().    
    switch_decoding_mode(is_decoding_mode: bool):
        Switch model to decoding mode (is_decoding_mode=True).
        Relevant for adaptation of pre-trained models
        to downstream decoding tasks.
    """

    kwargs = {
        "num_hidden_layers": num_hidden_layers,
        "embed_dim": embed_dim,
        "output_dim": output_dim,
        "num_attention_heads": num_attention_heads,
        "intermediate_dim_factor": intermediate_dim_factor,
        "n_positions": n_positions,
        "hidden_activation": hidden_activation,
        "dropout": dropout
    }

    if architecture == 'GPT':
        from decoder.gpt import GPTModel
        return GPTModel(**kwargs)

    elif architecture == 'PretrainedGPT2':
        from decoder.gpt import PretrainedGPT2
        return PretrainedGPT2(**kwargs)

    else:
        raise ValueError(f'{architecture}-architecture unkown.')