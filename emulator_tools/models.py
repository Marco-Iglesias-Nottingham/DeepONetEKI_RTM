import torch
import torch.nn as nn
import torch.nn.functional as F

class TrunkSharedGateAffinePreAct(nn.Module):
    def __init__(self, trunk_input_dim: int, Dp: int, Df: int, num_layers: int = 6):
        super().__init__()
        assert num_layers >= 1
        self.Dp, self.Df = Dp, Df
        self.output_dim = Dp + Df
        self.num_layers = num_layers

        self.encoder = nn.Linear(trunk_input_dim, self.output_dim)
        self.blocks  = nn.ModuleList([nn.Linear(self.output_dim, self.output_dim)
                                      for _ in range(num_layers - 1)])

        self.a = nn.ParameterList([nn.Parameter(torch.ones(self.output_dim))  for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(self.output_dim)) for _ in range(num_layers)])

        # Heads: sized to their slices
        self.out_p = nn.Linear(self.Dp, 1)
        self.out_f = nn.Linear(self.Df, 1)

    def _apply_gate_pre_act(self, h_pre: torch.Tensor, g_shared: torch.Tensor, layer_idx: int) -> torch.Tensor:
        g_layer = self.a[layer_idx] * g_shared + self.b[layer_idx]
        return h_pre * g_layer

    def forward(self, x: torch.Tensor, g_shared: torch.Tensor):
        h = self.encoder(x)
        h = F.gelu(self._apply_gate_pre_act(h, g_shared, 0))

        for i, layer in enumerate(self.blocks, start=1):
            h_pre = layer(h)
            h = F.gelu(self._apply_gate_pre_act(h_pre, g_shared, i))

        # Slice channels before the heads
        h_p = h[..., :self.Dp]
        h_f = h[...,  self.Dp:]

        p_out = self.out_p(h_p)              # e.g. regression/logi
        f_out = self.out_f(h_f) # e.g. probability
        #f_out = torch.sigmoid(self.out_f(h_f))  # e.g. probability
        return p_out, f_out
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)
    
    
class FieldUNetEncoderToVector(nn.Module):
    """
    Assumes ~120x120 inputs → bottleneck ~15x15 (adjust head in_features if different).
    """
    def __init__(self, input_channels=2, output_dim=100):
        super().__init__()
        self.enc1 = DoubleConv(input_channels, 64);  self.pool1 = nn.MaxPool2d(2)  # 120→60
        self.enc2 = DoubleConv(64, 128);             self.pool2 = nn.MaxPool2d(2)  # 60→30
        self.enc3 = DoubleConv(128, 256);            self.pool3 = nn.MaxPool2d(2)  # 30→15
        self.bottleneck = DoubleConv(256, 512)                                      # 15x15
        self.flatten = nn.Flatten()
        self.head = nn.Linear(512 * 15 * 15, output_dim)

    def forward(self, x):
        x = self.enc1(x); x = self.pool1(x)
        x = self.enc2(x); x = self.pool2(x)
        x = self.enc3(x); x = self.pool3(x)
        x = self.bottleneck(x)
        return self.head(self.flatten(x))


class ScalarMLPToVector(nn.Module):
    def __init__(self, scalar_dim: int, output_dim: int, hidden: int = 128, layers: int = 3, dropout: float = 0.0):
        super().__init__()
        sizes = [scalar_dim] + [hidden]*(layers-1) + [output_dim]
        seq = []
        for i in range(len(sizes)-1):
            seq.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                if dropout > 0: seq.append(nn.Dropout(dropout))
                seq.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*seq)
    def forward(self, s): return self.net(s)
    
# --- Branch: emit two gates and concat ---
class BranchNetPressureFrontSplit(nn.Module):
    def __init__(self, input_channels=2, scalar_dim=3, Dp=64, Df=36,
                 scalar_hidden=128, scalar_layers=3, scalar_dropout=0.0,
                 normalize=False):
        super().__init__()
        self.Dp, self.Df = Dp, Df
        D = Dp + Df

        self.field_branch  = FieldUNetEncoderToVector(input_channels=input_channels, output_dim=D)
        self.scalar_branch = ScalarMLPToVector(scalar_dim=scalar_dim, output_dim=D,
                                               hidden=scalar_hidden, layers=scalar_layers,
                                               dropout=scalar_dropout)
        self.normalize = normalize

    def forward(self, field_in, scalar_in):
        g_f = self.field_branch(field_in)   # [B, D]
        g_s = self.scalar_branch(scalar_in) # [B, D]
        if self.normalize:
            g_f = F.layer_norm(g_f, g_f.shape[-1:])
            g_s = F.layer_norm(g_s, g_s.shape[-1:])
        g = g_f * g_s                       # [B, D]

        # Split for clarity (optional) then re-concat — downstream still just needs g
        g_p, g_f = g[..., :self.Dp], g[..., self.Dp:]
        return torch.cat([g_p, g_f], dim=-1)  # [B, D]


# --- Top-level wrapper: pass (Dp, Df) and tile as before ---
class DeepONetPressureFront(nn.Module):
    def __init__(self, branch_input_channels, scalar_dim, trunk_input_dim,
                 Dp: int, Df: int, num_layers: int = 6):
        super().__init__()
        self.Dp, self.Df = Dp, Df
        self.output_dim = Dp + Df
        self.num_layers = num_layers

        self.branch_net = BranchNetPressureFrontSplit(
            input_channels=branch_input_channels,
            scalar_dim=scalar_dim,
            Dp=Dp, Df=Df
        )
        self.trunk_net = TrunkSharedGateAffinePreAct(
            trunk_input_dim=trunk_input_dim,
            Dp=Dp, Df=Df,
            num_layers=num_layers
        )


    def forward(self, branch_field_input, branch_scalar_input, trunk_input, apply_mask=True):
        B, N, Dtrunk = trunk_input.shape

        g = self.branch_net(branch_field_input, branch_scalar_input)  # [B, Dp+Df]
        g = g.unsqueeze(1).expand(-1, N, -1).reshape(B * N, self.output_dim)

        x = trunk_input.view(B * N, Dtrunk)

        raw_p, f_out = self.trunk_net(x, g)   # [BN,1], [BN,1]
        return raw_p.view(B, N, 1), f_out.view(B, N, 1)
