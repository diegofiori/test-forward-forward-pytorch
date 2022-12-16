from abc import ABC
from typing import List

import torch
import torch.utils.data

from efficientff.utils import ProgressiveTrainingDataset


def loss_fn(y, theta, sign):
    logits = torch.square(y).mean(dim=1) - theta
    loss = -logits * sign
    with torch.no_grad():
        accumulated_logits = logits.mean().item()
    loss = loss.mean()
    return loss, accumulated_logits


def probabilistic_loss_fn(y, theta, sign):
    logits = torch.square(y).mean(dim=1) - theta
    prob = torch.sigmoid(logits)
    loss = -torch.log(prob+ 1e-6) * sign
    with torch.no_grad():
        accumulated_logits = logits.mean().item()
    loss = loss.mean()
    return loss, accumulated_logits


def alternative_loss_fn(y, theta, sign):
    logits = y.pow(2).mean(dim=1) - theta
    with torch.no_grad():
        accumulated_logits = logits.mean().item()
    logits = -logits * sign
    prob = torch.exp(logits)
    loss = torch.log(1 + prob)
    loss = loss.mean()
    return loss, accumulated_logits


class BaseFFLayer(torch.nn.Module, ABC):
    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        raise NotImplementedError

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        raise NotImplementedError

    @property
    def requires_training(self):
        return True


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
        elif loss_fn_name == "probabilistic_loss_fn":
            self.loss_fn = probabilistic_loss_fn

    def forward(self, x):
        return self.layer(x)

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        """Train the layer with the given target.
        """
        # upgrade optimizer for positive goodness
        y = self(input_tensor.detach())
        y_pos = y[torch.where(signs == 1)]
        y_neg = y[torch.where(signs == -1)]
        # y_pos = self(input_tensor.detach()[torch.where(signs == 1)])
        loss_pos, cumulated_logits_pos = self.loss_fn(y_pos, theta, sign=1)
        # self.optimizer.zero_grad()
        # loss_pos.backward()
        # print(loss_pos.item())
        # self.optimizer.step()
        # y_neg = self(input_tensor.detach()[torch.where(signs == -1)])
        loss_neg, cumulated_logits_neg = self.loss_fn(y_neg, theta, sign=-1)
        self.optimizer.zero_grad()
        loss = loss_pos + loss_neg
        loss.backward()
        self.optimizer.step()
        separation = [cumulated_logits_pos, cumulated_logits_neg]
        y = torch.zeros(input_tensor.shape[0], *y_pos.shape[1:], device=input_tensor.device)
        y[torch.where(signs == 1)] = y_pos
        y[torch.where(signs == -1)] = y_neg
        return y.detach(), separation

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
            output = self()
        return output, None

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        with torch.no_grad():
            output = self(input_tensor)

        return output, torch.zeros(input_tensor.shape[0], device=input_tensor.device)
    
    @property
    def requires_training(self):
        return False


class LinearReLU(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class FCNetFFProgressive(BaseFFLayer):
    """FCNet trained using forward-forward algorithm. The network is trained
    in a progressive manner, i.e. the first layer is trained, then the
    second layer, and so on.
    """
    def __init__(self, layer_sizes: list, optimizer_name: str, optimizer_kwargs: dict, epochs: int, loss_fn_name: str = "loss_fn"):
        super().__init__()
        self.epochs = epochs
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFNormalization())
            self.layers.append(FFLayer(LinearReLU(layer_sizes[i], layer_sizes[i + 1]), optimizer_name, optimizer_kwargs, loss_fn_name))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def progressive_train(self, dl: torch.utils.data.DataLoader,  theta: float):
        """Train the network in a progressive manner.
        """
        print("Training the network in a progressive manner.")
        for i, layer in enumerate(self.layers):
            if layer.requires_training:
                for epoch in range(self.epochs):
                    accumulated_separation = None
                    for j, (data, signs) in enumerate(dl):
                        data = data.to(self.device)
                        signs = signs.to(self.device)
                        _, separation = layer.ff_train(data, signs, theta)
                        if accumulated_separation is None:
                            accumulated_separation = separation
                        else:
                            accumulated_separation[0] += separation[0]
                            accumulated_separation[1] += separation[1]
                        if j % 100 == 0:
                            print(f"Epoch: {epoch}, Batch: {j}, Layer: {i}")
                    print(f"Epoch {epoch} of layer {i} done.")
                    accumulated_separation[0] /= len(dl.dataset)
                    accumulated_separation[1] /= len(dl.dataset)
                    print("Goodness: ", accumulated_separation)
                    print(f"Accumulated separation: {(accumulated_separation[0] - accumulated_separation[1])/max(accumulated_separation[0], accumulated_separation[1])}")
                print(f"Finished training layer {i} / {len(self.layers)}.")
            # create a new dataloader for the next layer
            dataset = ProgressiveTrainingDataset(((layer(x.to(self.device)), sign.to(self.device)) for x, sign in dl))
            batch_size = dl.batch_size
            dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print("Finished training the network.")

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        """Train the network with the given target.
        """
        accumulated_separation = None
        for layer in self.layers:
            input_tensor, separation = layer.ff_train(input_tensor, signs, theta)
            if accumulated_separation is None:
                accumulated_separation = separation
            else:
                accumulated_separation[0] += separation[0]
                accumulated_separation[1] += separation[1]
        return input_tensor, accumulated_separation

    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the network with the given input and theta.
        """
        accumulated_goodness = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        for i, layer in enumerate(self.layers):
            input_tensor, goodness = layer.positive_eval(input_tensor, theta)
            if i > 1:
                accumulated_goodness += goodness
        return input_tensor, accumulated_goodness
    
    @property
    def device(self):
        return next(self.parameters()).device


class NormLinearReLU(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = FFNormalization()
        self.linear_relu = LinearReLU(in_features, out_features)

    def forward(self, x):
        return self.linear_relu(self.norm(x))


class RecurrentFFLayer(BaseFFLayer):
    def __init__(self, hidden_size: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str):
        super().__init__()
        self.layer = NormLinearReLU(2 * hidden_size, hidden_size)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.layer.parameters(), **optimizer_kwargs)
        self.loss_fn = eval(loss_fn_name)
    
    def forward(self, x_prev, x_same, x_next):
        x = torch.cat((x_prev, x_next), dim=1)
        new_x = self.layer(x)
        new_x = 0.3 * x_same + 0.7 * new_x
        return new_x
    
    def ff_train(
        self, 
        x_prev: torch.Tensor, 
        x_same: torch.Tensor, 
        x_next: torch.Tensor, 
        signs: torch.Tensor, 
        theta: float,
    ):
        new_x = self(x_prev.detach(), x_same.detach(), x_next.detach())
        y_pos = new_x[signs == 1]
        y_neg = new_x[signs == -1]
        loss_pos, goodness_pos = self.loss_fn(y_pos, theta, 1)
        loss_neg, goodness_neg = self.loss_fn(y_neg, theta, -1)
        loss = loss_pos + loss_neg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return new_x, [goodness_pos, goodness_neg]
    
    @torch.no_grad()
    def positive_eval(self, x_prev: torch.Tensor, x_same: torch.Tensor, x_next: torch.Tensor, theta: float):
        new_x = self(x_prev, x_same, x_next)
        goodness = self.loss_fn(new_x, theta, 1)[1]
        return new_x, goodness


class RecurrentProjectionFFLayer(BaseFFLayer):
    def __init__(self, input_size: int, output_size: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str):
        super().__init__()
        self.layer = NormLinearReLU(input_size, output_size)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.layer.parameters(), **optimizer_kwargs)
        self.loss_fn = eval(loss_fn_name)
    
    def forward(self, x: torch.Tensor):
        return self.layer(x)
    
    def ff_train(
        self, 
        x: torch.Tensor, 
        signs: torch.Tensor, 
        theta: float,
    ):
        new_x = self(x.detach())
        y_pos = new_x[signs == 1]
        y_neg = new_x[signs == -1]
        loss_pos, goodness_pos = self.loss_fn(y_pos, theta, 1)
        loss_neg, goodness_neg = self.loss_fn(y_neg, theta, -1)
        loss = loss_pos + loss_neg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return new_x, [goodness_pos, goodness_neg]
    
    @torch.no_grad()
    def positive_eval(self, x: torch.Tensor, theta: float):
        new_x = self(x)
        goodness = self.loss_fn(new_x, theta, 1)[1]
        return new_x, goodness


class RecurrentProjectedSoftmaxFFLayer(BaseFFLayer):
    def __init__(self, input_size: int, output_size: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str):
        super().__init__()
        self.loss_fn = eval(loss_fn_name)
        self.norm = FFNormalization()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.linear.parameters(), **optimizer_kwargs)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def ff_train(
        self,
        x: torch.Tensor,
        signs: torch.Tensor,
        theta: float,
    ):
        new_x = self(x.detach())
        y_pos = new_x[signs == 1]
        y_neg = new_x[signs == -1]
        loss_pos, goodness_pos = self.loss_fn(y_pos, theta, 1)
        loss_neg, goodness_neg = self.loss_fn(y_neg, theta, -1)
        loss = loss_pos + loss_neg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return new_x, [goodness_pos, goodness_neg]

    @torch.no_grad()
    def positive_eval(self, x: torch.Tensor, theta: float):
        new_x = self(x)
        goodness = self.loss_fn(new_x, theta, 1)[1]
        return new_x, goodness


class RecurrentFCNetFF(BaseFFLayer):
    """Recurrent FCNet trained using forward-forward algorithm.
    """
    def __init__(self, layer_sizes: list, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str = "loss_fn"):
        super().__init__()
        self.time_steps = 10
        self.test_time_steps = 8
        self.storable_time_steps = [2, 3, 4]
        self.states = []
        self.layers = torch.nn.ModuleList()
        self.projector = RecurrentProjectionFFLayer(layer_sizes[0], layer_sizes[1], optimizer_name, optimizer_kwargs, loss_fn_name)
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(RecurrentFFLayer(layer_sizes[i], optimizer_name, optimizer_kwargs, loss_fn_name))
        self.proj_y = RecurrentProjectionFFLayer(layer_sizes[-1], layer_sizes[-2], optimizer_name, optimizer_kwargs, loss_fn_name)
        self.softmax = RecurrentProjectedSoftmaxFFLayer(layer_sizes[-2], layer_sizes[-1], optimizer_name, optimizer_kwargs, loss_fn_name)
        self.num_labels = layer_sizes[-1]
    
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def bottom_up(self, x: torch.Tensor, y: torch.Tensor):
        states = []
        x_proj = self.projector(x)

        for layer in self.layers:
            states.append(x_proj)
            x_proj = layer(x_proj, torch.zeros_like(x_proj, device=self.device), torch.zeros_like(x_proj, device=self.device))
        states.append(x_proj)
        states.append(y)
        y_arg = torch.argmax(y, dim=1)
        x_proj_ = x_proj.clone()
        x_proj_[torch.arange(x_proj.shape[0]), y_arg] = - 1e6
        neg_prob = self.softmax(x_proj_)
        cumulative_neg_prob = torch.cumsum(neg_prob, dim=1)
        neg_samples = torch.argmax(1.* (cumulative_neg_prob > torch.rand(x.shape[0], 1)), dim=1)
        neg_samples = torch.functional.F.one_hot(neg_samples, num_classes=self.num_labels)
        return states, neg_samples

    def forward(self, x: torch.Tensor, prev_states: List[torch.Tensor]):
        x_proj = self.projector(x)
        new_states = []
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_state = prev_states[i + 2]
            else:
                next_state = self.proj_y(prev_states[i + 2].float())
            new_states.append(x_proj)
            x_proj = layer(prev_states[i], prev_states[i+1], next_state)
        new_states.append(x_proj)
        y = self.softmax(x_proj)
        new_states.append(y)
        return new_states

    def ff_train(self, input_tensor: torch.Tensor, labels: torch.Tensor, theta: float):
        """Train the network with the given target.
        """
        with torch.no_grad():
            states, neg_samples = self.bottom_up(input_tensor, labels)
            neg_states, _ = self.bottom_up(input_tensor, neg_samples)
            states = [torch.cat([s, ns], dim=0) for s, ns in zip(states, neg_states)]
            signs = torch.cat([
                torch.ones(input_tensor.shape[0], device=self.device), 
                -torch.ones(input_tensor.shape[0], device=self.device)
            ], dim=0)
            input_tensor = torch.cat([input_tensor, input_tensor], dim=0)
        # states have been created, now we can train the network
        x_proj, accumulated_goodness = self.projector.ff_train(input_tensor, signs, theta)
        for _ in range(self.time_steps):
            new_states = []
            x = x_proj
            for j, layer in enumerate(self.layers):
                if j < len(self.layers) - 1:
                    next_state = states[j + 2]
                else:
                    next_state = self.proj_y(states[j + 2].float())
                new_states.append(x)
                x, goodnesses = layer.ff_train(states[j], states[j+1], next_state, signs, theta)
                accumulated_goodness[0] += goodnesses[0]
                accumulated_goodness[1] += goodnesses[1]
            new_states.append(x)
            with torch.no_grad():
                x_ = states[-2][torch.where(signs == -1)]
                real_y = states[-1][torch.where(signs == 1)]
                x_[torch.arange(x_.shape[0]), torch.argmax(real_y, dim=1)] = -1e6
                y = self.softmax(x_)
                cumulative_y = torch.cumsum(y, dim=1)
                neg_samples = torch.argmax(1. * (cumulative_y > torch.rand(x_.shape[0], 1)), dim=1)
                neg_samples = torch.functional.F.one_hot(neg_samples, num_classes=self.num_labels)
                # replace just negative samples 
                next_labels = states[-1].clone()
                next_labels[torch.where(signs == -1)] = neg_samples
                new_states.append(next_labels)
            states = new_states
        accumulated_goodness[0] /= self.time_steps * len(self.layers) + 1
        accumulated_goodness[1] /= self.time_steps * len(self.layers) + 1
        with torch.no_grad():
            states = [t[:input_tensor.shape[0]//2] for t in states]
        return states, accumulated_goodness

    @torch.no_grad()
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the network with the given input and theta.
        """
        labels = torch.arange(0, self.num_labels, device=self.device)
        labels = torch.functional.F.one_hot(labels, num_classes=self.num_labels)
        original_bs = input_tensor.shape[0]
        input_tensor = input_tensor.unsqueeze(1).repeat(1, self.num_labels, 1).reshape(-1, input_tensor.shape[-1])
        labels = labels.unsqueeze(0).repeat(original_bs, 1, 1).reshape(-1, labels.shape[-1])
        
        states, _ = self.bottom_up(input_tensor, labels)
        x_proj, _ = self.projector.positive_eval(input_tensor, theta)
        accumulated_goodness = torch.zeros(input_tensor.shape[0], device=self.device)

        for time_step in range(self.test_time_steps):
            new_states = []
            x = x_proj
            for j, layer in enumerate(self.layers):
                if j < len(self.layers) - 1:
                    next_state = states[j + 2]
                else:
                    next_state = self.proj_y(states[j + 2].float())
                new_states.append(x)
                x, goodnesses = layer.positive_eval(states[j], states[j+1], next_state, theta)
                if time_step in self.storable_time_steps:
                    accumulated_goodness += goodnesses
            new_states.append(x)
            new_states.append(states[-1])
            states = new_states
        accumulated_goodness = accumulated_goodness.reshape(original_bs, self.num_labels)
        prediction = torch.argmax(accumulated_goodness, dim=1)
        return prediction, accumulated_goodness
        
        
            