"""
The transformer block is multi-headed attn + feedforward layers with add&norm
at the end of each layer.
Reference: https://www.youtube.com/watch?v=U0s0f995w14
Based on: https://arxiv.org/abs/1706.03762
"""
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """ Multi-headed self attention class
    embed_size represents the input encoding embedding size that will get
    distributed across h or num of heads. E.g. input is 256 with 8 heads, then
    each head will take a vector of len 32.
    Output is output of a lin layer with same len as input.
    """
    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads 
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == self.embed_size), \
            "Embed size needs to be div by heads."

        # Lin layer for Q K V before Scaled Dot-Product Attention
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Lin layer after concat output of Scaled Dot-Product Attention
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]  # number of examples per batch
        # Src and trg sentence lengths
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Distribute embedding vector among heads
        # self.heads and self.head_dim are at dim 3 and 4 because input is in
        # (N, value_len, embed_len), so reshaping doesn't affect other dims
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        
        # send inputs thru linear layers after defining them
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # energy is the output of Q dot K, i.e. similarity(Q, K)
        # Q.shape = (N, query_len, heads, heads_dim)
        # K.shape = (N, key_len, heads, heads_dim)
        # energy.shape = (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e28"))
        
        # dim=3 means normalization is across key_len, i.e. src sentence
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # attention.shape = (N, heads, query_len, key_len)
        # values.shape = (N, value_len, heads, heads_dim)
        # out.shape = (N, query_len, heads, head_dim)
        # N.B. key_len = value_len after dot product (see paper)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size)  # reshape is the concat op after attention

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    """ This is the Multiheaded Attention + Feed Forward layers (with add&norm at the
    end of each layer) and skip connections module described in the paper."""
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)  # layer norm = batch norm just for ea. ex.
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)

        x = self.dropout(self.norm1(attention + queries))  # first skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # second forward connection
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # A bit magical how position_embedding fn will auto correlate with words and their positions
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)  # Q K V are all going to be the same in the encoder

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion)
        self. dropout = nn.Dropout(dropout)

    def forward(self, x, values, keys, src_mask, trg_mask):
        """ src_mask is optional, if we need to pad input w/ pad tokens
        trg_mask is mandatory"""
        attention = self.attention(x, x, x, trg_mask)
        queries = self.dropout(self.norm(attention + x))
        out = self.transformer_block(values, keys, queries, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                trg_vocab_size,
                embed_size,
                num_layers,
                heads,
                forward_expansion,
                dropout,
                device,
                max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            # enc_out, enc_out are the values and keys sent to DecoderBlock's forward()
            # val and key are the same thing and are produced by the Encoder
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Following line sets non src_pad_idx ele to 1 and else to 0
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # Lower triangular matrix
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encode_src = self.encoder(src, src_mask)
        out = self.decoder(trg, encode_src, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    max_length = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)