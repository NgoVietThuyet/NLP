import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size phải chia hết cho heads"

        # Các lớp Linear để tạo Q, K, V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Lớp kết hợp cuối cùng
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # N: Batch size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 1. Chia vector thành các heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 2. Chiếu qua Linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 3. Tính Attention Score (Q nhân K chuyển vị)
        # Einsum: n=batch, h=heads, q=query_len, k=key_len, d=head_dim
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # 4. Masking (Che đi các vị trí không hợp lệ)
        if mask is not None:
            # mask == 0 là các vị trí cần che, gán giá trị rất nhỏ (-1e20)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 5. Softmax & Scaling
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # 6. Nhân với V
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values])
        
        # 7. Nối lại (Concat) và đi qua lớp cuối
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        
        return out

class TransformerBlock(nn.Module):
    """Một tầng Encoder"""
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Attention -> Add & Norm
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query)) # Residual Connection
        
        # Feed Forward -> Add & Norm
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # Residual Connection
        return out

class DecoderBlock(nn.Module):
    """Một tầng Decoder (Khác Encoder là có 2 lần Attention)"""
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # 1. Masked Self Attention (Decoder tự nhìn chính nó, nhưng che tương lai)
        # trg_mask: che các từ chưa xuất hiện
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x)) # Add & Norm

        # 2. Cross Attention (Decoder nhìn vào Encoder) & Feed Forward
        # value, key lấy từ ENCODER (src). query lấy từ DECODER (trg).
        # Đoạn này dùng lại class TransformerBlock để tiết kiệm code vì logic giống nhau
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,     # Số lượng từ vựng đầu vào (nguồn)
        trg_vocab_size,     # Số lượng từ vựng đầu ra (đích)
        src_pad_idx,        # Index của token padding
        trg_pad_idx,        # Index của token padding
        embed_size=512,     # Kích thước vector
        num_layers=6,       # Số tầng (block)
        forward_expansion=4,# Hệ số mở rộng FFN
        heads=8,            # Số heads
        dropout=0,
        device="cpu",
        max_length=100      # Độ dài câu tối đa
    ):
        super(Transformer, self).__init__()

        # --- ENCODER COMPONENTS ---
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_length, embed_size)
        
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        # --- DECODER COMPONENTS ---
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)
        
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size, heads, forward_expansion, dropout
                )
                for _ in range(num_layers)
            ]
        )

        # --- OUTPUT ---
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def make_src_mask(self, src):
        # Tạo mask che phần padding của input
        # src shape: (N, src_len)
        # mask shape: (N, 1, 1, src_len) -> broadcasting cho các heads
        mask = (src != 0).unsqueeze(1).unsqueeze(2) # Giả sử pad_idx = 0
        return mask.to(self.device)

    def make_trg_mask(self, trg):
        # Tạo mask hình tam giác để che tương lai (Look-ahead mask)
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape

        # Tạo ma trận vị trí: [0, 1, 2, ..., seq_len]
        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(0).expand(N, src_seq_length).to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(0).expand(N, trg_seq_length).to(self.device)
        )

        # 1. Embeddings + Positional Encodings
        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )
        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        # Tạo masks
        src_mask = self.make_src_mask(src) # Mask padding
        trg_mask = self.make_trg_mask(trg) # Mask tam giác

        # 2. Chạy qua ENCODER
        enc_out = embed_src
        for layer in self.encoder_layers:
            # Trong Encoder, src đóng vai trò là cả Value, Key, Query
            enc_out = layer(enc_out, enc_out, enc_out, src_mask)

        # 3. Chạy qua DECODER
        out = embed_trg
        for layer in self.decoder_layers:
            # Trong Decoder:
            # - x (input của layer) đóng vai trò Query
            # - enc_out (output của encoder) đóng vai trò Key và Value (Cross Attention)
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        # 4. Output final prediction
        out = self.fc_out(out)

        return out

# --- PHẦN CHẠY THỬ (TEST) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Giả lập tham số
    src_vocab_size = 10000
    trg_vocab_size = 10000
    src_pad_idx = 0
    trg_pad_idx = 0
    
    # Khởi tạo mô hình
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device
    ).to(device)

    # Giả lập dữ liệu đầu vào
    # Batch size = 2, câu nguồn dài 10 từ, câu đích dài 8 từ
    # Các số nguyên đại diện cho index của từ trong từ điển
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 0], 
                      [1, 8, 7, 3, 4, 5, 6, 7, 2, 0]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], 
                        [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # Chạy mô hình
    out = model(x, trg[:, :-1]) # trg input bỏ từ cuối để dự đoán
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {trg[:, :-1].shape}")
    print(f"Output shape: {out.shape}")
    
    # Output mong đợi: (N, trg_len, trg_vocab_size)
    # Tức là (2, 7, 10000)