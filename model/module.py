from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn

__all__ = ['Cbhg', 'PreNet', 'DecoderRnn', 'ContentBasedAttention', 'SeqModule']


@dataclass
class AttentionOutput:
    context_vector: torch.Tensor
    attention_weight: torch.Tensor


class SeqModule(ABC):
    """Abstract base class for pytorch module handling sequential data"""

    @property
    @abstractmethod
    def num_output_features(self):
        """Force subclasses to have `out_num_features` to make inference
        on number of output features more feasible in each network module"""


class Cbhg(nn.Module, SeqModule):
    """
    CBHG Encoder module for Tacotron.
    """

    def __init__(self,
                 input_emb_dim: int = 128,
                 conv_bank_K: int = 16,
                 conv_bank_c: int = 128,
                 maxpool_kernel_size: int = 2,
                 maxpool_stride: int = 1,
                 proj_kernel_size: int = 3,
                 proj_channel_dims: Tuple[int, int] = (128, 128),
                 num_highway_layer: int = 4,
                 highway_fc_dim: int = 128,
                 num_gru_cells: int = 128):
        """
        Initialize CBHG encoder
        Args:
            input_emb_dim: input embedding dimension for the CBHG
            conv_bank_K: number of filter groups to make for 1d conv bank
            conv_bank_c: number of features to encode for 1d conv bank
            maxpool_kernel_size: kernel size for maxpool layer
            maxpool_stride: stride for maxpool layer
            proj_kernel_size: kernel size for 1d conv layer in
                the projection module
            proj_channel_dims: number of output features for the two layer
                in the projection module
            num_highway_layer: number of fully connected layers in the highway
                module
            highway_fc_dim: size of fully connected layer in the highway module
            num_gru_cells: number of GRU cells (per direction)
        """
        super(Cbhg, self).__init__()

        self.conv1d_bank = Conv1dBank(conv_bank_K, conv_bank_c, input_emb_dim)

        maxpool_pad = get_pad(maxpool_kernel_size)
        self.max_pool = nn.Sequential(
            maxpool_pad,
            nn.MaxPool1d(maxpool_kernel_size, stride=maxpool_stride)
        )

        proj_input_num_features = self.conv1d_bank.num_output_features

        self.conv1d_proj = Conv1dProjection(
            proj_kernel_size, proj_input_num_features, proj_channel_dims)

        # the output of the projection should have the same number of features
        # as the dimension of the fc layers in the highway network (so that
        # dimension is preserved along the highway network). If they are not the
        # same, make a linear layer mapping the projection output into the
        # highway network
        self.proj_highway_linear = None
        if proj_channel_dims[-1] != highway_fc_dim:
            self.proj_highway_linear = nn.Linear(
                proj_channel_dims[-1], highway_fc_dim)

        self.highway_net = HighwayNetwork(num_highway_layer, highway_fc_dim)

        gru_input_num_features = self.highway_net.num_output_features

        # GRU will output hidden state of dimension `2 * num_gru_cells` due
        # to bi-directionality
        self.gru = nn.GRU(input_size=gru_input_num_features,
                          hidden_size=num_gru_cells,
                          batch_first=True,
                          bidirectional=True)

        # GRU need not be initialized, as initializing GRU already initializes
        # its weight and bias

        # Expected number of features from this network
        self.__num_output_features = num_gru_cells * 2

    @property
    def num_output_features(self):
        return self.__num_output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CBHG module
        Args:
            x: character embedding or mel spectrogram, in case the cbhg module
                is used as the encoder or the post-processing unit respectively
               x at this stage is always assumed to have dimension of
                (batch, seq, feature)

        Returns:

        """
        # permute dimension into (batch, feature, seq)
        x = x.permute(0, 2, 1)

        # residual connection between input and output upto conv1d_proj
        x = x + self.conv1d_proj(self.max_pool(self.conv1d_bank(x)))

        # nn.Linear, as well as nn.GRU accepts input of dimension
        # (batch, seq, feature). But our x here so far has dimension of
        # (batch, feature, seq). Transpose accordingly
        x = x.permute(0, 2, 1)

        if self.proj_highway_linear is not None:
            x = self.proj_highway_linear(x)

        # output hidden layers from all time steps
        return self.gru(self.highway_net(x))[0]


class Conv1dBank(nn.Module, SeqModule):
    def __init__(self, K: int, c: int, emb_dim: int):
        super(Conv1dBank, self).__init__()
        conv_bank = []
        # create conv filter bank with variable filter size
        for k in range(1, K + 1):
            # Get padding module accordingly such that time dimension is
            # preserved for all the conv filters
            padding_module = get_pad(k)

            # each layer in the conv bank consists of 1D conv
            conv_batch_relu = nn.Sequential(
                padding_module,
                nn.Conv1d(in_channels=emb_dim, kernel_size=k,
                          out_channels=c, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            )

            conv_bank.append(conv_batch_relu)

        self.conv_bank = nn.ModuleList(conv_bank)

        self.__num_output_features = c * K
        # Randomly initialize fc layers
        self._init_layer()

    @property
    def num_output_features(self):
        return self.__num_output_features

    def _init_layer(self) -> None:
        for module in self.conv_bank:
            for layer in module:
                if isinstance(layer, nn.Conv1d):
                    nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input has dim of (batch, feature, seq)

        # Get output of each conv layer in the conv bank, and concatenate
        # then along output channel axis
        stack = []
        for conv_layer in self.conv_bank:
            stack.append(conv_layer(x))
        return torch.cat(stack, dim=1)


class Conv1dProjection(nn.Module, SeqModule):
    def __init__(self, kernel_size: int, input_num_features: int,
                 channel_dims: Tuple[int, int]):
        super(Conv1dProjection, self).__init__()
        layer0_dim, layer1_dim = channel_dims

        padding_module = get_pad(kernel_size)

        self.layer1 = nn.Sequential(
            padding_module,
            nn.Conv1d(in_channels=input_num_features, kernel_size=kernel_size,
                      out_channels=layer0_dim, bias=False),
            nn.BatchNorm1d(layer0_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            padding_module,
            nn.Conv1d(in_channels=layer0_dim, kernel_size=kernel_size,
                      out_channels=layer1_dim, bias=False),
            nn.BatchNorm1d(layer1_dim)
        )

        self.__num_output_features = layer1_dim

    @property
    def num_output_features(self):
        return self.__num_output_features

    def _init_layer(self) -> None:
        for module in [self.layer1, self.layer2]:
            for layer in module:
                if isinstance(layer, nn.Conv1d):
                    nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class HighwayNetwork(nn.Module, SeqModule):
    def __init__(self, n_layer: int = 4, fc_dim: int = 128):
        super(HighwayNetwork, self).__init__()

        # Make highway layers consisting of fully connected layers and ReLU
        # activations, and gates
        network_layers = []
        gates = []
        for i in range(n_layer):
            fc_relu = nn.Sequential(
                nn.Linear(fc_dim, fc_dim, bias=False),
                nn.ReLU(inplace=True)
            )
            network_layers.append(fc_relu)

            gate = nn.Sequential(
                nn.Linear(fc_dim, fc_dim),
                nn.Sigmoid()
            )
            gates.append(gate)

        self.network_layers = nn.ModuleList(network_layers)

        self.gates = nn.ModuleList(gates)

        self.__num_output_features = fc_dim

        # Randomly initialize fc layers
        self._init_layer()

    @property
    def num_output_features(self):
        return self.__num_output_features

    def _init_layer(self) -> None:
        for layer in self.gates:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # Highway network needs to be initialized with negative value
                nn.init.constant_(layer.bias, -1)

        for layer in self.network_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, gate in zip(self.network_layers, self.gates):
            # the output of gate is scalar tensor
            x = (1 - gate(x)) * x + gate(x) * layer(x)

        return x


class DecoderRnn(nn.Module, SeqModule):
    def __init__(self, num_input_features: int = 256, num_cells: int = 256):
        super(DecoderRnn, self).__init__()
        # We might need a fully connected layer in case the dimension of
        # the concatenated vector between the attention rnn output and
        # the context vector do not match the number of cells in GRU units in
        # the decoder RNN
        self.fc = None
        if num_input_features != num_cells:
            self.fc = nn.Linear(num_input_features, num_cells)

        self.gru1 = nn.GRU(input_size=num_cells,
                           hidden_size=num_cells,
                           batch_first=True)

        self.gru2 = nn.GRU(input_size=num_cells,
                           hidden_size=num_cells,
                           batch_first=True)

        self.__num_out_features = num_cells

    @property
    def num_output_features(self):
        return self.__num_out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc is not None:
            x = self.fc(x)
        # Residual connection between previous and current layers
        layer1_out = self.gru1(x)[0]
        layer2_out = self.gru2(layer1_out + x)[0]
        return x + layer1_out + layer2_out


class ContentBasedAttention(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super(ContentBasedAttention, self).__init__()

        self.fc_encoder = nn.Linear(feature_dim, feature_dim)
        self.fc_decoder = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Parameter(torch.rand(feature_dim))

        self.__init_layers()

    def __init_layers(self):
        for layer in [self.fc_decoder, self.fc_decoder]:
            torch.nn.init.kaiming_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(
            self, decoder_states: torch.Tensor,
            encoder_states: torch.Tensor) -> AttentionOutput:
        """
        Calculate context vector using content-based attention
        u_{ti} = v^{T} * tanh(W_{1} * e_{i} + W_{2} * d_{t})
        a_{ti} = softmax(u_{ti})
        d'_{t} = sum(a_{ti} * e_{i}) along encoder seq axis

        v, W_{1}, W_{2} are learnable parameters

        Args:
            decoder_states: (batch, decoder_seq, feature)
            encoder_states: (batch, encoder_seq, feature)

        Returns:
            attention output consisting of context vector and attention weight

        """
        # Make a dummy dimension to enable column-wise addition
        fc_decoder_out = self.fc_decoder(decoder_states).unsqueeze(2)
        fc_encoder_out = self.fc_encoder(encoder_states).unsqueeze(1)

        # Results in shape of (batch, decoder_seq, encoder_seq, feature)
        states_out = torch.tanh(fc_decoder_out + fc_encoder_out)

        # Dot proeduct along feature axis resulting
        # (batch, decoder_seq, encoder_seq)
        alignment_score = torch.einsum(
            'bdef,f->bde', states_out, self.v)

        # Calculate softmax along `encoder_seq` dimension to calculate attention
        # weight. Resulting dimension is still the same
        # (batch, decoder_seq, encoder_seq)
        attention_weight = torch.softmax(alignment_score, dim=2)

        # Context vector is simply summation of encoder vectors weighted by
        # attention weights. Resulting dim: (batch, decoder_seq, feature)
        context_vector = torch.bmm(attention_weight, encoder_states)

        return AttentionOutput(context_vector, attention_weight)


class PreNet(nn.Module, SeqModule):
    def __init__(self,
                 num_input_features: int = 256,
                 fc_dims: tuple = (256, 128),
                 dropout_rate: float = 0.5):
        super(PreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_input_features, fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.__num_output_features = fc_dims[1]

    @property
    def num_output_features(self):
        return self.__num_output_features

    def _init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # Highway network needs to be initialized with negative value
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_pad(k: int) -> nn.ConstantPad1d:
    """
    Get padding to preserve output dimension given kernel size.
    This is needed since PyTorch has no builtin feature in Conv*d and MaxPool*d
    to pad a value asymmetrically.
    Args:
        k: kernel size of convolutional filter

    Returns:
        Padding size returned in tuple. This is to be passed to `padding`
        options in `nn.Conv2d`
    """

    if (k - 1) % 2 == 0:
        pad = ((k - 1) // 2, (k - 1) // 2)
    else:
        pad = (k // 2, (k - 1) // 2)

    return nn.ConstantPad1d(pad, 0)
