import os
import torch
from torch import jit

with torch.no_grad():
    fake_input = torch.rand(1, 3, 224, 224)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    model.eval()
    out1 = model(fake_input).squeeze()

    sm = torch.jit.trace(model, fake_input)
    sm.save("squeezenet.pt")
    load_sm = jit.load("squeezenet.pt")
    out2 = load_sm(fake_input).squeeze()

    print(out1[:5], out2[:5])
    print(out1.shape, out2.shape)
