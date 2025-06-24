from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
import torch


@dataclass
class CustomModelOutput(ModelOutput):
    hidden_states: torch.FloatTensor=None
    repa_hidden_states: torch.FloatTensor=None
    latent_token: torch.FloatTensor=None
    pixel_value: torch.FloatTensor=None

    def __setattr__(self, name, value):
        self[name] = value



if __name__ == "__main__":
    a = CustomModelOutput(
        pixel_value=torch.Tensor([1, 2])
    )

    print(a.hidden_states)
    print(a['hidden_states'])
    # a.hidden_states = torch.Tensor([4, 5])
    a['hidden_states'] = torch.Tensor([11, 15])
    # a['hidden_states'] = None
    # a.hidden_states = None
    print(a.hidden_states)
    print(a['hidden_states'])
