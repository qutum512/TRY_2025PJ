# model.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, offset:offset + L, :]

class TransformerSeq(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_proj = nn.Linear(1, d_model)
        self.tgt_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, src_seq, tgt_seq):
        B, Tp, _ = src_seq.shape
        B, Tf, _ = tgt_seq.shape

        src = self.src_proj(src_seq)
        tgt = self.tgt_proj(tgt_seq)
        src = self.pos_enc(src, offset=0)
        tgt = self.pos_enc(tgt, offset=Tp - 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Tf).to(src.device)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        return self.pred_head(out)

    @torch.no_grad()
    def rollout(self, src_seq, steps: int):
        # autoregressive rollout using decoder only (keeps same pos enc offsets as training)
        self.eval()
        B, Tp, _ = src_seq.shape
        device = src_seq.device
        src = self.src_proj(src_seq)
        src = self.pos_enc(src, offset=0)
        memory = self.transformer.encoder(src)

        preds = []
        pos_idx = Tp - 1
        next_tok_val = src_seq[:, -1:, :]  # raw scalar values shape (B,1,1)
        tgt_seq_emb = torch.empty((B, 0, self.src_proj.out_features), device=device)
        full_mask = nn.Transformer.generate_square_subsequent_mask(steps).to(device)

        for _ in range(steps):
            next_tok_emb = self.tgt_proj(next_tok_val)
            # add positional embedding for this next token
            next_tok_emb = next_tok_emb + self.pos_enc.pe[:, pos_idx:pos_idx + 1, :]
            tgt_seq_emb = torch.cat([tgt_seq_emb, next_tok_emb], dim=1)
            Tf = tgt_seq_emb.size(1)
            tgt_mask = full_mask[:Tf, :Tf]
            out = self.transformer.decoder(tgt=tgt_seq_emb, memory=memory, tgt_mask=tgt_mask)
            pred_k = self.pred_head(out[:, -1:, :])
            preds.append(pred_k)
            next_tok_val = pred_k
            pos_idx += 1
        return torch.cat(preds, dim=1)
