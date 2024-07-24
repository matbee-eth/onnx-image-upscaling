
from spandrel import ImageModelDescriptor, ModelLoader

# add extra architectures before `ModelLoader` is used
spandrel_extra_arches.install()

# load a model from disk
upscale_model = ModelLoader().load_from_file(r"path/to/model.pth")


# import torch
# from torchvision.transforms import functional as F
# from torchvision.transforms import InterpolationMode
# from torchvision.transforms.functional import InterpolationMode
# from torchvision.transforms import Resize
# from torchvision.transforms import CenterCrop
# import torchvision.transforms.functional as TF

# from onnxruntime import InferenceSession
# import onnxruntime as ort

# upscale_model = torch.load("./4x-UltraSharp.pth")

x = torch.rand(1, 3, 512, 512)

dynamic_axes = {
    "input": {0: "batch_size", 2: "width", 3: "height"},
    "output": {0: "batch_size", 2: "width_out", 3: "height_out"},
}

print(upscale_model)
    
torch.onnx.export(upscale_model,
                    x,
                    "./ultrasharp.onnx",
                    verbose=True,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=17,
                    export_params=True,
                    dynamic_axes=dynamic_axes,
                    )