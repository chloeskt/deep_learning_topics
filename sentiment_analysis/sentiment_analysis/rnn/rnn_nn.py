import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


def pack_outputs(state_seq, lengths):
    # Select the last states just before the padding
    last_indices = lengths - 1
    final_states = []
    for b, t in enumerate(last_indices.tolist()):
        final_states.append(state_seq[t, b])
    state = torch.stack(final_states).unsqueeze(0)

    # Pack the final state_seq (h_seq, c_seq e.t.c.)
    state_seq = pack_padded_sequence(state_seq, lengths)

    return state_seq, state


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.W_xh = nn.Linear(self.input_size, self.hidden_size, bias=True)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()

        else:
            raise ValueError(
                "Unrecognized activation. Allowed activations: tanh or relu"
            )

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        # We handle varying length inputs for you
        lenghts = None
        if isinstance(x, PackedSequence):
            x, lengths = pad_packed_sequence(x)

        # State initialization in case of zero, check the below code!
        if h is None:
            h = torch.zeros(
                (1, x.size(1), self.hidden_size), device=x.device, dtype=x.dtype
            )

        h_seq = []

        for xt in x.unbind(0):
            # update the hidden state
            h = self.activation(self.W_hh(h) + self.W_xh(xt))
            h_seq.append(h)

        # Stack the h_seq as a tensor
        h_seq = torch.cat(h_seq, 0)

        # Re-pack the outputs
        if lenghts is not None:
            h_seq, h = pack_outputs(h_seq, h)

        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """

        self.hidden_size = hidden_size
        self.input_size = input_size

        # Input gate
        self.input_Wx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.input_Wh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.input_activation = nn.Sigmoid()

        # Forget gate
        self.forget_Wx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.forget_Wh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.forget_activation = nn.Sigmoid()

        # New output
        self.output_Wx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.output_Wh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.output_activation = nn.Sigmoid()

        # New cell/context vector c_t
        self.cell_Wx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.cell_Wh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cell_activation = nn.Tanh()

        # New context h_t
        self.context_activation = nn.Tanh()

    def forget_gate(self, x, h):
        """
        Forget Gate

        :param x: input at time t
        :param h: hidden state at time t-1
        :return:
        """
        x1 = self.forget_Wx(x)
        h1 = self.forget_Wh(h)
        return self.forget_activation(x1 + h1)

    def input_gate(self, x, h):
        """
        Input Gate

        :param x: input at time t
        :param h: hidden state at time t-1
        :return:
        """
        return self.input_activation(self.input_Wx(x) + self.input_Wh(h))

    def output_gate(self, x, h):
        """
        Output Gate

        :param x: input at time t
        :param h: hidden state at time t-1
        :return:
        """
        return self.output_activation(self.output_Wx(x) + self.output_Wh(h))

    def cell_memory_gate(self, x, h, c, i, f):
        """
        Cell memory gate

        :param x: input at time t
        :param h: hidden state at time t-1
        :param c: cell state at time t-1
        :param i: input state at time t
        :param f: forget state at time t
        :return:
        """
        k = i * self.cell_activation(self.cell_Wx(x) + self.cell_Wh(h))
        e = f * c
        return e + k

    def context_gate(self, o, c):
        """
        Context/Hidden gate

        :param o: output at time t
        :param c: cell state at time t
        :return:
        """
        return o * self.context_activation(c)

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vector of sequence(1, batch_size, hidden_size)
        - c: Final cell state vector of sequence(1, batch_size, hidden_size)
        """
        # Below code handles the batches with varying sequence lengths
        lengths = None
        if isinstance(x, PackedSequence):
            x, lengths = pad_packed_sequence(x)

        # State initialization provided to you
        state_size = (1, x.size(1), self.hidden_size)
        if h is None:
            h = torch.zeros(state_size, device=x.device, dtype=x.dtype)
        if c is None:
            c = torch.zeros(state_size, device=x.device, dtype=x.dtype)
        assert state_size == h.shape == c.shape

        h_seq = []
        c_seq = []

        # process each time step at a time
        for xt in x.unbind(0):
            f = self.forget_gate(xt, h)
            i = self.input_gate(xt, h)
            o = self.output_gate(xt, h)
            c = self.cell_memory_gate(xt, h, c, i, f)
            h = self.context_gate(o, c)

            # add to the lists
            h_seq.append(h)
            c_seq.append(c)

        # Stack the h_seq as a tensor
        h_seq = torch.cat(h_seq, 0)
        # Stack the c_seq as a tensor
        c_seq = torch.cat(c_seq, 0)

        # Handle the padding stuff
        if lengths is not None:
            h_seq, h = pack_outputs(h_seq, lengths)
            c_seq, c = pack_outputs(c_seq, lengths)

        return h_seq, (h, c)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        """
        Inputs:
        - num_embeddings: Number of embeddings
        - embedding_dim: Dimension of embedding outputs
        - pad_idx: Index used for padding (i.e. the <eos> id)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # We handle the padding for you
        self.padding_idx = padding_idx
        self.register_buffer(
            "padding_mask", (torch.arange(0, num_embeddings) != padding_idx).view(-1, 1)
        )

        self.weight = None

        self.weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(self.weight)

        self.weight.data[padding_idx] = 0

    def forward(self, inputs):
        """
        Inputs:
            inputs: A long tensor of indices of size (seq_len, batch_size)
        Outputs:
            embeddings: A float tensor of size (seq_len, batch_size, embedding_dim)
        """

        # Ensure <eos> always return zeros
        # and padding gradient is always 0
        weight = self.weight * self.padding_mask

        embeddings = weight[inputs]

        return embeddings
