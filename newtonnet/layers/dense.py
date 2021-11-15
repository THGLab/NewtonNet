from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_

from newtonnet.layers.batchrenorm import BatchRenorm1d


class Dense(nn.Linear):
    r"""
    Fully connected linear layer with activation function.

    credit : https://github.com/atomistic-machine-learning/schnetpack under the MIT License

    .. math::
        x = normalize(x)
        y = activation(xW^T + b)
        y = dropout(y)

    Parameters
    ----------
    in_features: int
        number of input feature :math:`x`.
    out_features: int
        number of output features :math:`y`.
    bias: bool, optional (default: True)
        if False, the layer will not adapt bias :math:`b`.
    activation: callable, optional (default: None)
        A torch.nn activation function (e.g., torch.nn.ReLU())
        if None, no activation function is used.
    weight_init: callable, optional (default: torch.nn.init.xavier_uniform_)
        weight initializer
    bias_init: callable, optional (default: torch.nn.init.zeros_)
        bias initializer
    dropout: float, optional (default: None)
        if the value is a float, it will be used as dropout probabilty
    norm: bool, optional (default: None)
        if True, batch normalization will be applied prior to the Linear mapping.


    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 activation=None,
                 weight_init=xavier_uniform_,
                 xavier_init_gain=1.,
                 bias_init=zeros_,
                 dropout=None,
                 norm=None):
        self.weight_init = weight_init
        self.gain = xavier_init_gain
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)
        self.activation = activation
        # initialize linear layer y = xW^T + b

        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self.norm = norm
        if norm:
            self.norm = BatchRenorm1d(
                num_features=out_features
                # track_running_stats=False
            )  #, momentum=0.99, eps=0.001) # momentum and eps are based on Keras default values

    def reset_parameters(self):
        self.weight_init(self.weight, gain=self.gain)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            A tensor with shape (batch_size, in_features)


        Returns
        -------
        torch.Tensor
            A tensor with shape (batch_size, out_features)


        """
        # compute linear layer y = xW^T + b
        x = super(Dense, self).forward(x)

        # batch normalization
        if self.norm:
            x = self.norm(x)

        # add activation function
        if self.activation:
            x = self.activation(x)

        # dropout
        if self.dropout:
            x = self.dropout(x)

        return x
