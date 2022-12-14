from abc import ABC

import torch


def loss_fn(y, theta, signs):
    prob = torch.sigmoid(torch.square(y).sum(dim=1) - theta)
    loss = -torch.log(prob) * signs
    # loss = torch.square(y).sum(dim=1) - theta
    # loss = - loss * signs
    loss = loss.mean()
    return loss


def alternative_loss_fn(y, theta, signs):
    logits = torch.square(y).sum(dim=1) - theta
    logits = -logits * signs
    prob = torch.exp(logits)
    loss = torch.log(1 + prob)
    loss = loss.mean()
    return loss


class BaseFFLayer(torch.nn.Module, ABC):
    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        raise NotImplementedError

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        raise NotImplementedError


class FFLayer(BaseFFLayer):
    """Layer wrapper for efficient forward-forward layers.
    """
    def __init__(self, layer, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str = "loss_fn"):
        super().__init__()
        self.layer = layer
        self.optimizer = getattr(torch.optim, optimizer_name)(layer.parameters(), **optimizer_kwargs)
        if loss_fn_name == "loss_fn":
            self.loss_fn = loss_fn
        elif loss_fn_name == "alternative_loss_fn":
            self.loss_fn = alternative_loss_fn

    def forward(self, x):
        return self.layer(x)

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        """Train the layer with the given target.
        """
        y = self(input_tensor.detach())
        loss = self.loss_fn(y, theta, signs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return y.detach()

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the layer with the given input and theta.
        """
        y = self(input_tensor)
        return y, torch.square(y).sum(dim=1) - theta


class FFNormalization(BaseFFLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l2_norm = torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1, keepdim=True) + 1e-8
        return x / l2_norm

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        with torch.no_grad():
            output = self(input_tensor)
        return output

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        with torch.no_grad():
            output = self(input_tensor)

        return output, torch.zeros(input_tensor.shape[0], device=input_tensor.device)


class LinearReLU(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class FCNetFF(BaseFFLayer):
    def __init__(self, layer_sizes: list, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str = "loss_fn"):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFLayer(LinearReLU(layer_sizes[i], layer_sizes[i + 1]), optimizer_name, optimizer_kwargs, loss_fn_name))
            if i < len(layer_sizes) - 2:
                self.layers.append(FFNormalization())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        """Train the network with the given target.
        """
        for layer in self.layers:
            input_tensor = layer.ff_train(input_tensor, signs, theta)
        return input_tensor

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the network with the given input and theta.
        """
        accumulated_goodness = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        for layer in self.layers:
            input_tensor, goodness = layer.positive_eval(input_tensor, theta)
            accumulated_goodness += goodness
        return input_tensor, torch.sigmoid(accumulated_goodness)
