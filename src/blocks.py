import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def heinsen_associative_scan_log(log_coeffs, log_values):
    """
    Performs associative scan operation in log space for sequence modeling.

    Parameters
    ----------
    log_coeffs : torch.Tensor
        Logarithmic coefficients tensor of shape (batch, seq_len, ...)
    log_values : torch.Tensor
        Logarithmic values tensor compatible with log_coeffs

    Returns
    -------
    torch.Tensor
        Exponential of computed log-space scan results

    Examples
    --------
    >>> coeffs = torch.randn(32, 100, 64).log_softmax(-1)
    >>> values = torch.randn(32, 100, 64).log_softmax(-1)
    >>> output = heinsen_associative_scan_log(coeffs, values)
    """
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


class G_Act(nn.Module):
    """
    G activation function with for MinGRU.

    Forward pass:
    - For x >= 0: linear function x + 0.5
    - For x < 0: sigmoid function

    Logarithmic pass (log() method):
    - For x >= 0: log(ReLU(x) + 0.5)
    - For x < 0: -softplus(-x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    def log(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class CausalAvg(nn.Module):
    """
    Causal average pooling layer with state retention for sequential processing.

    Parameters
    ----------
    kernel_size : int
        Size of the averaging window. Determines padding for causal operation.

    Examples
    --------
    >>> layer = CausalAvg(kernel_size=5)
    >>> x = torch.randn(32, 10, 64)
    >>> output, state = layer(x)
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

        self.conv = nn.AvgPool2d(kernel_size=(self.kernel_size, 1), stride=1, padding=0)

    def forward(self, x, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(x.shape[0], self.pad, x.shape[-2], x.shape[-1]).to(
                x.device
            )

        y = torch.cat([state, x], dim=1)
        return self.conv(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1), y[:, -self.pad :]


class MinGRU(nn.Module):
    """
    Minimal Gated Recurrent Unit with both parallel and sequential forward pass.

    Parameters
    ----------
    dim : int
        Feature dimension size

    Examples
    --------
    >>> gru = MinGRU(128)
    >>> x = torch.randn(32, 50, 16, 128)
    >>> output, hidden = gru(x, prev_hidden=None)
    """

    def __init__(self, dim: int):
        super().__init__()

        self.linear_gh = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
        )

        self.act = G_Act()

    def forward(self, x, prev_hidden):
        batch_size, seq_len, token, edim = x.shape

        gate, hidden = self.linear_gh(x).chunk(2, -1)

        if seq_len == 1:

            gate = torch.sigmoid(gate)
            hidden = self.act(hidden)
            out = (
                torch.lerp(prev_hidden, hidden, gate)
                if prev_hidden is not None
                else (hidden * gate)
            )

        else:

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = self.act.log(hidden)
            log_values = log_z + log_tilde_h

            if prev_hidden is not None:
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                zero_pad = torch.zeros(
                    batch_size,
                    1,
                    token,
                    edim,
                    device=log_coeffs.device,
                    dtype=log_coeffs.dtype,
                )
                log_coeffs = torch.cat([zero_pad, log_coeffs], dim=1)

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        return out, next_prev_hidden


class MultiGRU(nn.Module):
    """
    Stacked GRU architecture with residual connections.

    Parameters
    ----------
    in_dim : int
        Input feature dimension size
    out_dim : int
        Input feature dimension size
    shortcut : bool
        Whether to use shortcut connection. Default is True.

    Examples
    --------
    >>> mgru = MultiGRU(256)
    >>> x = torch.randn(32, 100, 256)
    >>> output, states = mgru(x)
    """

    def __init__(self, in_dim: int, out_dim: int, shortcut: bool = True):
        super().__init__()

        self.shortcut = shortcut

        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.gru1 = MinGRU(in_dim)
        self.gru2 = MinGRU(in_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x, prev_hiddens=None):
        if prev_hiddens is None:
            prev_hiddens = [None, None]
        y, state1 = self.gru1(self.pre(x), prev_hiddens[0])
        y, state2 = self.gru2(y, prev_hiddens[1])
        y = self.proj(y)
        if self.shortcut:
            y = y + x
        return y, [state1, state2]


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer with learnable adjacency matrix .

    Parameters
    ----------
    in_dim : int
        Input feature dimension size
    out_dim : int
        Input feature dimension size
    A : torch.Tensor
        Adjacency matrix of shape (nodes, nodes) or (batch, nodes, nodes)
    use_uwb_attn : bool
        Whether to use UWB attention mechanism. Default is False.

    Examples
    --------
    >>> adj = torch.randn(24, 24)
    >>> layer = GCNLayer(128, adj)
    >>> x = torch.randn(32, 100, 24, 128)
    >>> output = layer(x)
    """

    def __init__(
        self, in_dim: int, out_dim: int, A: torch.Tensor, use_uwb_attn: bool = False
    ):
        super(GCNLayer, self).__init__()

        self.use_uwb_attn = use_uwb_attn
        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if A.shape[0] == A.shape[1]:
            self.A = (A + torch.eye(A.shape[0], device=A.device)).unsqueeze(-1)

        else:
            self.A = A.unsqueeze(-1)

        self.W = nn.Parameter(torch.empty(*A.shape, out_dim)).to(A.device)
        nn.init.xavier_uniform_(self.W)

        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, uwb: torch.Tensor = None):
        a = self.W * self.A
        y = self.pre(x)

        if uwb is not None and self.use_uwb_attn:
            a = torch.einsum("btmn,mnd->btmnd", uwb, a)
            y = torch.einsum("btnd,btmnd->btmd", y, a)
        else:
            y = torch.einsum("btnd,mnd->btmd", y, a)

        return self.proj(y)


class GCNBlock(nn.Module):
    """
    Residual GCN block containing two GCN layers.

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    output_dim : int
        Output feature dimension
    As : tuple[torch.Tensor, torch.Tensor]
        Pair of adjacency matrices for consecutive layers
    use_uwb_attn : bool
        Whether to use UWB attention mechanism. Default is False.

    Examples
    --------
    >>> adj1 = torch.randn(24, 24)
    >>> adj2 = torch.randn(24, 24)
    >>> block = GCNBlock(128, 128, (adj1, adj2))
    >>> x = torch.randn(32, 100, 24, 128)
    >>> output = block(x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        As: tuple[torch.Tensor, torch.Tensor],
        use_uwb_attn: bool = False,
    ):
        super().__init__()
        self.layer1 = GCNLayer(input_dim, output_dim, As[0], use_uwb_attn)
        self.layer2 = GCNLayer(output_dim, output_dim, As[1], use_uwb_attn)

    def forward(self, x: torch.Tensor, uwb: torch.Tensor = None):
        y = self.layer1(x, uwb)
        return self.layer2(y, uwb) + x
