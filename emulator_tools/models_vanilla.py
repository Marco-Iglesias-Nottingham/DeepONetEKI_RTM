import torch
import torch.nn as nn
import torch.nn.functional as F

class TrunkSharedGateAffinePreAct(nn.Module):
    """
    Classic trunk feature extractor (NO branch interaction inside).
    Keeps the same architecture: encoder + (num_layers-1) linear blocks + GELU.
    Produces trunk features split into Dp and Df channels.
    """
    def __init__(self, trunk_input_dim: int, Dp: int, Df: int, num_layers: int = 6):
        super().__init__()
        assert num_layers >= 1
        self.Dp, self.Df = Dp, Df
        self.output_dim = Dp + Df
        self.num_layers = num_layers

        self.encoder = nn.Linear(trunk_input_dim, self.output_dim)
        self.blocks  = nn.ModuleList(
            [nn.Linear(self.output_dim, self.output_dim) for _ in range(num_layers - 1)]
        )

        # Heads remain (same as before)
        self.out_p = nn.Linear(self.Dp, 1)
        self.out_f = nn.Linear(self.Df, 1)

    def forward_features(self, x: torch.Tensor):
        """
        Returns trunk features (h_p, h_f) with shapes:
          h_p: [*, Dp], h_f: [*, Df]
        """
        h = self.encoder(x)
        h = F.gelu(h)

        for layer in self.blocks:
            h = F.gelu(layer(h))

        h_p = h[..., :self.Dp]
        h_f = h[...,  self.Dp:]
        return h_p, h_f

    def forward_heads(self, h_p: torch.Tensor, h_f: torch.Tensor):
        """
        Applies the same heads as before to produce scalar outputs.
        """
        p_out = self.out_p(h_p)
        f_out = self.out_f(h_f)
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

        g_p, g_f = g[..., :self.Dp], g[..., self.Dp:]
        return torch.cat([g_p, g_f], dim=-1)  # [B, D]


class DeepONetPressureFront(nn.Module):
    """
    Classic DeepONet coupling: branch and trunk interact ONLY at the end.
    Trunk produces features; branch produces coefficients; combine by elementwise product
    then apply the same heads as before.
    """
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

        # Branch coefficients (per-sample), split into p/f parts
        g = self.branch_net(branch_field_input, branch_scalar_input)  # [B, Dp+Df]
        g_p, g_f = g[..., :self.Dp], g[..., self.Dp:]                 # [B, Dp], [B, Df]

        # Tile over trunk points to match [B*N, ...]
        g_p = g_p.unsqueeze(1).expand(-1, N, -1).reshape(B * N, self.Dp)
        g_f = g_f.unsqueeze(1).expand(-1, N, -1).reshape(B * N, self.Df)

        x = trunk_input.view(B * N, Dtrunk)

        # Trunk features (no branch interaction inside)
        h_p, h_f = self.trunk_net.forward_features(x)                 # [B*N, Dp], [B*N, Df]

        # Classic DeepONet interaction: only at the end
        h_p = h_p * g_p
        h_f = h_f * g_f

        # Same heads as before
        raw_p, f_out = self.trunk_net.forward_heads(h_p, h_f)         # [B*N,1], [B*N,1]
        return raw_p.view(B, N, 1), f_out.view(B, N, 1)

