import os
import torch
import sys
from transformers import AutoModel
from torch import jit
import timm

if __name__ == "__main__":
    # Example code taken from https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/pytorch-mobilenet-image
    # Modified to allow passing in an input argument representing model name
    if len(sys.argv) < 2:
        print("Usage: python gen_model.py <model name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    model_file_name = f"{model_name}.pt"
    model_file_name = model_file_name.replace("/", "-")

    with torch.no_grad():
        fake_input = torch.rand(1, 3, 224, 224)
        model = AutoModel.from_pretrained(model_name, pretrained=True)
        model.eval()
        out1 = model(fake_input).squeeze()

        traced_model = torch.jit.trace(model, fake_input)
        if not os.path.exists(model_file_name):
            traced_model.save(model_file_name)
        load_traced_model = jit.load(model_file_name)
        out2 = load_traced_model(fake_input).squeeze()

        print(out1[:5], out2[:5])
