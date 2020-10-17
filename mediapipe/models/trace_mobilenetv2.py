# `torch` package known to work:
# python -m pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
import torch

# More about this model at https://pytorch.org/hub/pytorch_vision_mobilenet_v2

if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.5.0',
                           'mobilenet_v2',
                           pretrained=True)
    model.eval()
    input_tensor = torch.rand(1, 3, 224, 224)
    script_model = torch.jit.trace(model, input_tensor)
    script_model.save("mediapipe/models/mobilenetv2.pt")
