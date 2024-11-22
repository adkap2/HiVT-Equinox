import torch
import torch.nn as nn
from typing import List, Optional
import einops

class TorchSingleInputEmbedding(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(TorchSingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x forward shape", x.shape)
        # print("x forward first few values", x[:5])
        return self.embed(x)
    


class TorchMultipleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(TorchMultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        # self.apply(init_weights) # I am going to avoid this for now Make github issue tracker and postpone this
        # WRITE INIT WEIGHTS to laways be 1


    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        
        output = torch.stack(continuous_inputs).sum(dim=0)
        # Einops tested output


        print("TORCH output forward shape", output.shape)
        print("TORCH output forward first few values", output[:5])
        
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)
