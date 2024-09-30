import os
import torch
from torch import jit

with torch.no_grad():
    fake_input = torch.rand(1, 3, 224, 224)
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    model.eval()
    out1 = model(fake_input).squeeze()

    sm = torch.jit.trace(model, fake_input)
    sm.save("resnet50.pt")
    load_sm = jit.load("resnet50.pt")
    out2 = load_sm(fake_input).squeeze()

    print(out1[:5], out2[:5])
