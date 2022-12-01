import torch
import torch.nn as nn
import numpy as np

class BLSTMConversionModel(nn.Module):
    """
    Conversion model based on BLSTM
    """
    def __init__(self, in_channels, out_channels, lstm_hidden):
        """
        :param in_channels: input feature dimension,
                            usually (bnfs_dim + f0s_dim) when use bnfs and f0s as inputs
        :param out_channels: mel dimension or your acoustic feature dimnesion
        :param lstm_hidden: parameter dimension
        """
        super(BLSTMConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden = lstm_hidden
        self.blstm1 = nn.LSTM(input_size=in_channels,
                              hidden_size=lstm_hidden,
                              bidirectional=True)
        self.blstm2 = nn.LSTM(input_size=lstm_hidden * 2,
                              hidden_size=lstm_hidden,
                              bidirectional=True)
        self.out_projection = nn.Linear(in_features=2 * lstm_hidden,
                                        out_features=out_channels)

    def forward(self, x):
        """
        :param x: [time, batch, features]
        :return: [time, batch, features]
        """
        # pass to the 1st BLSTM layer
        blstm1_out, _ = self.blstm1(x)
        # pass to the 2nd BLSTM layer
        blstm2_out, _ = self.blstm2(blstm1_out)
        # project to the output dimension
        outputs = self.out_projection(blstm2_out)
        return outputs


class BLSTMResConversionModel(nn.Module):
    """
    Conversion model based on BLSTM with ResidualNet, you need to
    define your ResidualNet Module to be used in this Module.
    """
    def __init__(self, in_channels, out_channels, lstm_hidden, other_params):
        """
        :param in_channels: input feature dimension,
                            usually (bnf_dim + f0s_dim) when use bnfs and f0s as inputs
        :param out_channels: mel dimension or your acoustic feature dimnesion
        :param lstm_hidden: parameter dimension
        :param other_params: other parameters you need to define resnet
        """
        super(BLSTMResConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden = lstm_hidden
        self.blstm1 = nn.LSTM(input_size=in_channels,
                              hidden_size=lstm_hidden,
                              bidirectional=True)
        self.blstm2 = nn.LSTM(input_size=lstm_hidden * 2,
                              hidden_size=lstm_hidden,
                              bidirectional=True)
        self.out_projection = nn.Linear(in_features=2 * lstm_hidden,
                                        out_features=out_channels)
        self.resnet = ResidualNet(out_channels, other_params)

    def forward(self, x):
        """
        :param x: [time, batch, features]
        :return: [time, batch, features]
        """
        # pass to the 1st BLSTM layer
        blstm1_out, _ = self.blstm1(x)
        # pass to the 2nd BLSTM layer
        blstm2_out, _ = self.blstm2(blstm1_out)
        # project to the output dimension
        initial_outs = self.out_projection(blstm2_out)
        residual = self.resnet(initial_outs)
        final_outs = _  # define the final outputs here
        return final_outs


class ResidualNet(nn.Module):
    """
    Residual network used to generate residual information
    for predicted Mel-spectrogram. It's widely accepted that
    this can help generate much more accurate Mel-spectrogram.
    """
    def __init__(self, channels, other_params):
        """
        :param channels: equal to dimension of the Mel-spectrogram
        :param other_params: parameters for the definition of your residual net
        """
        super(ResidualNet, self).__init__()
        self.channels = channels
        # define your module components below
        # e.g. self.dense_layer = nn.Linear(in_features=channels, out_features=channels)

    def forward(self, x):
        """
        :param x: input mel-spectrogram
        :return: output improved mel-spectrogram
        """
        # define your inference process below
        pass


class BLSTMToManyConversionModel(nn.Module):
    """
    BLSTM based any-to-many VC model with SPKEmbedding Module.
    You need to define your SPKEmbedding Module, and 
    might need to change necessary components in this Module.
    """

    def __init__(self, in_channels, out_channels, num_spk, embd_dim, lstm_hidden):
        """
        :param in_channels: input feature dimension,
                            usually (bnf_dim + f0s_dim) when use bnfs and f0s as inputs
        :param out_channels: mel dimension or your acoustic feature dimnesion
        :param lstm_hidden: parameter dimension
        """
        super(BLSTMToManyConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_spk = num_spk
        self.embd_dim = embd_dim
        self.lstm_hidden = lstm_hidden
        self.spk_embed_net = SPKEmbedding(num_spk=num_spk,
                                          embd_dim=embd_dim)
        self.emb_proj1 = nn.Linear(in_features=embd_dim,
                                   out_features=in_channels)
        self.emb_proj2 = nn.Linear(in_features=embd_dim,
                                   out_features=lstm_hidden * 2)
        self.blstm1 = nn.LSTM(input_size=in_channels,
                              hidden_size=lstm_hidden,
                              bidirectional=True)
        self.blstm2 = nn.LSTM(input_size=lstm_hidden * 2,
                              hidden_size=lstm_hidden,
                              bidirectional=True)
        self.out_projection = nn.Linear(in_features=2 * lstm_hidden,
                                        out_features=out_channels)

    def forward(self, x, spk_inds):
        """
        Feel free to include input features you need to extract speaker embedding,
        and much possibly you need to modify the corresponding part in the training script if you do so.
        :param x: [time, batch, features]
        :param spk_inds: indices of speakers, [batch, ]
        :return: [time, batch, features]
        """
        # look up speaker embedding
        spk_embds = self.spk_embed_net(spk_inds)
        spk_embds =spk_embds.repeat(x.shape[0],1,1)
        
        # add speaker embd to the inputs
        blstm1_inputs = _ # give your implementation here
        # pass to the 1st BLSTM layer
        blstm1_outs, _ = self.blstm1(blstm1_inputs)
        # add speaker embd to the outputs of 1st lstm
        blstm2_inputs = _  # give your implementation here
        # pass to the 2nd BLSTM layer
        blstm2_outs, _ = self.blstm2(blstm2_inputs)
        # project to the output dimension
        outputs = self.out_projection(blstm2_outs)
        return outputs


class SPKEmbedding(nn.Module):
    """
    Speaker embedding module.
    You are required to implement this module as asked in the assignment.
    You can implement the basic ideas provided in the assignment.
    Besides, you can also add more components, e.g., to generate speaker embedding
    according to the speaker's acoustic features such as Mel-spectrogram.
    """
    def __init__(self, num_spk, embd_dim):
        """
        Feel free to add parameters to define your own components
        :param num_spk: number of target speakers
        :param embd_dim: output speaker embedding dimension
        """
        super(SPKEmbedding, self).__init__()
        # define your module components below
        # e.g. self.embedding_table = ...
        #from zxt
        
        
    def forward(self, spk_inds):
        """
        Feel free to use other input features to extract your speaker embedding and
        modify the corresponding parts in the BLSTMToManyConversionModel Module.
        :param spk_inds: speaker indices, should be of type torch.Longtensor
        :return: speaker embedding, should be of shape [batch, out_channels]
        """
        # define your inference process below
        # e.g. return self.embedding_table(spk_inds)
        pass


class CustomToOneConversionModel(nn.Module):
    """
    define your custom any-to-one conversion model here
    """
    def __init__(self, in_channels, out_channels, other_params):
        """
        :param in_channels: input feature dimension (e.g. bnf's dim + F0 dim)
        :param out_channels: output feature dimension (e.g. mel-spectrogram's dim)
        :param other_params: other parameters for your custom defined components
        """
        super(CustomToOneConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # define your own conversion model's components below
        # e.g. self.dense_layer = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        """
        :param x: VC input features
        :return: output acoustic features
        """
        # define your model's transform process below
        pass


class Mel2Linear(nn.Module):
    """
    Here defines a module to transform mel-spectrogram into linear-spectrogram
    in a neural way.
    """
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: input mel-spectrogram feature dimensions
        :param out_channels: output linear-spectrogram feature dimensions
        """
        super(Mel2Linear, self).__init__()
        # define your module components below
        # e.g. self.dense_layer = nn.Linear(in_features=in_channels, out_features=out_channels)

    def call(self, x):
        """
        :param x: input mel-spectrogram
        :return: output linear-spectrogram
        """
        # define your transform process below


if __name__ == '__main__':
    # these are just some tests
    x = torch.randn(4, 16, 8)
    mdl = BLSTMConversionModel(in_channels=8, out_channels=9, lstm_hidden=16)
    print(mdl)
    outs = mdl(x)
    print(outs.shape)
