import torch
import torch.nn as nn
from typing import List, Optional
import einops

from utils import print_array_type


class TorchSingleInputEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(TorchSingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),  # nn.Linear
            nn.LayerNorm(out_channel),  # nn.LayerNorm
            nn.ReLU(inplace=True),  # nn.ReLU
            nn.Linear(out_channel, out_channel),  # nn.Linear
            nn.LayerNorm(out_channel),  # nn.LayerNorm
            nn.ReLU(inplace=True),  # nn.ReLU
            nn.Linear(out_channel, out_channel),  # nn.Linear
            nn.LayerNorm(out_channel),  # nn.LayerNorm
        )  # nn.Sequential

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Shape: Float[Tensor, "2 2"]
        output = self.embed(
            x
        )  # torch.Tensor -> [batch_size, embed_dim] # Shape: Float[Tensor, "2 2"]
        return output


class TorchMultipleInputEmbedding(nn.Module):

    def __init__(self, in_channels: List[int], out_channel: int) -> None:
        super(TorchMultipleInputEmbedding, self).__init__()  # nn.Module
        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channel, out_channel),
                    nn.LayerNorm(out_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_channel, out_channel),
                )
                for in_channel in in_channels  # List[int]
            ]
        )  # nn.ModuleList
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),  # nn.LayerNorm
        )  # nn.Sequential
        # self.apply(init_weights) # I am going to avoid this for now Make github issue tracker and postpone this
        # WRITE INIT WEIGHTS to laways be 1

    def forward(
        self,
        continuous_inputs: List[torch.Tensor],  # Shape: Float[Tensor, "2 2 2
        categorical_inputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:

        # print("TORCH continuous_inputs", continuous_inputs) # List[torch.Tensor]
        # TORCH continuous_inputs [tensor([[-3.4846,  0.1921],
        # [-0.0388,  0.7021]]), tensor([[ 0.1999,  1.0384],
        # [-2.6815, -2.0240]])]
        # Shape: [Tensor(2, 2), Tensor(2, 2)]
        # breakpoint()
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](
                continuous_inputs[i]
            )  # Float[Tensor, "2 2 2"]

        # print("TORCH continous inputs :" , continuous_inputs)
        output = torch.stack(continuous_inputs).sum(dim=0)  # Float[Tensor, "2 2]

        # print("TORCH output", output)
        # TORCH output tensor([[0.6121, 0.5279],
        # [0.8997, 0.7784]], grad_fn=<SumBackward1>)
        # Shape tensor([2, 2])
        # print("TORCH output :" , output)
        # Einops tested output

        # print("TORCH output forward shape", output.shape)
        # print("TORCH output forward first few values", output[:5])

        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)

        output = self.aggr_embed(output)  # Float[Tensor, "2 2"]

        return output
