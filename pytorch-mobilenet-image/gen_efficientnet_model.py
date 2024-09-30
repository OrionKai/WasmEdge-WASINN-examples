import os
import torch
from torch import jit

with torch.no_grad():
    fake_input = torch.rand(1, 3, 224, 224)
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    model.eval()
    out1 = model(fake_input)

    sm = torch.jit.trace(model, fake_input)
    sm.save("efficientnet.pt")
    load_sm = jit.load("efficientnet.pt")
    out2 = load_sm(fake_input)

    print(out1[:5], out2[:5])
