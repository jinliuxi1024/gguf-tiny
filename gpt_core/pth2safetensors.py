import torch
from safetensors.torch import save_file

def convert_pth_to_safetensors(pth_file: str, output_file: str):
    model = torch.load(pth_file,map_location='cpu')
    save_file(model, output_file)


# pth_file = 'model.pth'
# output_file = 'model.safetensors'
# convert_pth_to_safetensors(pth_file, output_file)