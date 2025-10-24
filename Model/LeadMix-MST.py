import torch
import torch.nn as nn
import torch.nn.functional as F

def dw_conv1d(cin, cout, k, d=1):
    return nn.Sequential(
        nn.Conv1d(cin, cin, kernel_size=k, padding=d*(k//2), dilation=d, groups=cin, bias=False),
        nn.Conv1d(cin, cout, kernel_size=1, bias=False)
    )

class SE1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signal."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c, max(1, c//r), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, c//r), c, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class LeadMixBlock(nn.Module):
    def __init__(self, c_in, c_out, num_leads, heads=4, ff=2.0, drop=0.1):
        super().__init__()
        self.ms1 = dw_conv1d(c_in, c_out//2, 3, d=1)
        self.ms2 = dw_conv1d(c_in, c_out//2, 3, d=2)
        self.proj = nn.Conv1d(c_out, c_out, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=8, num_channels=c_out)
        self.se = SE1D(c_out)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

        self.lead_embed = nn.Parameter(torch.randn(num_leads, c_out))
        self.qkv = nn.Linear(c_out, c_out*3, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim=c_out, num_heads=heads, batch_first=True, dropout=drop)
        self.ffn = nn.Sequential(
            nn.Linear(c_out, int(c_out*ff)),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(int(c_out*ff), c_out),
        )
        self.norm_tokens1 = nn.LayerNorm(c_out)
        self.norm_tokens2 = nn.LayerNorm(c_out)

    def forward(self, x):
        res = x
        
        y = torch.cat([self.ms1(x), self.ms2(x)], dim=1)
        y = self.proj(y)
        y = self.act(self.gn(y))
        y = self.se(y)
        y = self.drop(y)

        B, C, L = y.size()
        g = y.mean(dim=-1) 
        tokens = self.lead_embed.unsqueeze(0).expand(B, -1, -1) 

        gv = g.unsqueeze(1).expand(-1, tokens.size(1), -1)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        k = k + gv
        v = v + gv
        h,_ = self.mha(self.norm_tokens1(q), self.norm_tokens1(k), self.norm_tokens1(v))
        tokens = tokens + h
        tokens = tokens + self.ffn(self.norm_tokens2(tokens))

        scale = torch.tanh(tokens).mean(dim=1).unsqueeze(-1)
        shift = torch.tanh(tokens).amax(dim=1).unsqueeze(-1)
        y = y * (1.0 + 0.1*scale) + 0.1*shift

        return res + y 

class LeadMixMST(nn.Module):
    def __init__(self, input_length, num_leads, base=48, depth=3, heads=4, drop=0.1):
        super().__init__()
        self.drop_path = drop

        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, base, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        
        blocks = []
        c = base
        for i in range(depth):
            blocks += [
                LeadMixBlock(c, c, num_leads=num_leads, heads=heads, drop=drop),
                nn.Conv1d(c, c*2, 3, stride=2, padding=1, bias=False),  
                nn.GroupNorm(8, c*2),
                nn.SiLU(),
            ]
            c *= 2
        self.body = nn.Sequential(*blocks)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(c, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x).squeeze(-1)
        x = F.dropout(x, p=self.drop_path, training=self.training)
        return self.fc(x)
