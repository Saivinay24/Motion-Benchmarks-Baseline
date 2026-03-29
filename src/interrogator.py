import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class VLMInterrogator:
    """
    Project 2306 - AI of the Beholder: Explainability Module.
    Interrogates the VLM's attention mechanism to visualize if the model 
    focuses on skeletal naturalness or physical inconsistencies.
    """
    def __init__(self, model, target_layer_name='visual.transformer.resblocks.11'):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate_motion_heatmap(self, input_frame, target_class_idx):
        """
        Generates a Grad-CAM heatmap to identify 'Beholder' focus areas.
        """
        self.model.zero_grad()
        output = self.model(input_frame)
        
        # Target the 'naturalness' score or specific motion class
        loss = output[0, target_class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        cam = np.maximum(cam.detach().cpu().numpy(), 0)
        cam = cv2.resize(cam, (input_frame.shape[2], input_frame.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# Implementation Note: Use this to verify if the VLM focuses on 
# joint collapses (e.g., knees/elbows) during animation.