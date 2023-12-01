import torch
import torch.nn as nn
import math


class InteractionEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InteractionEncoder, self).__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, interaction):
        return self.net(interaction)


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size, nhead, n_layers, max_length, pad_token, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.seq_len = max_length
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_len)
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_size, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pad_token = pad_token

    def forward(self, src):
        src_padding_mask = (src == self.pad_token)
        embedded_src = self.token_embedding(src.permute(1, 0))
        final_src = self.positional_encoding(embedded_src)
        output = self.encoder(final_src.permute(1, 0, 2), src_key_padding_mask=src_padding_mask)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size, nhead, n_layers, max_length, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.seq_len = max_length
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_len)
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, hidden_size, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.dense = nn.Linear(embedding_dim, vocab_dim)

    def forward(self, target, memory, target_mask, target_padding_mask):
        embedded_target = self.token_embedding(target.permute(1, 0))
        final_target = self.positional_encoding(embedded_target)
        output = self.decoder(final_target.permute(1, 0, 2), memory,
                              tgt_mask=target_mask, tgt_key_padding_mask=target_padding_mask,
                              memory_key_padding_mask=None, memory_mask=None)
        logits = self.dense(output)
        return logits


class RNNBasedEncoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size, n_layers, rnn_type='gru', dropout=0.1):
        super(RNNBasedEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,
                              batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,
                               batch_first=True)
        else:
            print(f'type: {rnn_type} not supported')
            exit()

    def forward(self, src):
        embedded_src = self.token_embedding(src.permute(1, 0))
        if self.rnn_type == 'gru':
            output, hidden = self.rnn(embedded_src.permute(1, 0, 2))
            out = hidden
        elif self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded_src.permute(1, 0, 2))
            out = (hidden, cell)
        else:
            out = None
        return out


class RNNBasedDecoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size, n_layers, rnn_type='gru', dropout=0.1):
        super(RNNBasedDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,
                              batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,
                               batch_first=True)
        else:
            print(f'type: {rnn_type} not supported')
            exit()
        self.dense = nn.Linear(hidden_size, vocab_dim)

    def forward(self, target, memory, *args, **kwargs):
        embedded_target = self.token_embedding(target.permute(1, 0))
        if self.rnn_type == 'gru':
            output, hidden = self.rnn(embedded_target.permute(1, 0, 2), memory)
        elif self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded_target.permute(1, 0, 2), memory)
        else:
            output = None
        logits = self.dense(output)
        return logits


class InteractionTranslator(nn.Module):
    def __init__(self, prot_encoder, mol_encoder, decoder):
        super().__init__()
        self.prot_encoder = prot_encoder
        self.mol_encoder = mol_encoder
        self.decoder = decoder

