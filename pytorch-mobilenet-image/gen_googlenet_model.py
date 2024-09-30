import os
import torch
from torch import jit

with torch.no_grad():
    fake_input = torch.rand(1, 3, 224, 224)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model.eval()
    out1 = model(fake_input)

    sm = torch.jit.script(model)
    sm.save("googlenet.pt")
    load_sm = jit.load("googlenet.pt")
    out2 = load_sm(fake_input)

    print(out1[:5], out2[:5])
