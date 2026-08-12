import os
import torch
import torch.nn as nn
import torchvision.models as models
from jetracer.urban.config import NUM_WAYPOINTS, ROUTE_COMMANDS, COMMAND_TO_INDEX

class ConditionedResNet18Waypoints(nn.Module):
    """
    Conditioned Trajectory Prediction Model:
    Inputs:
        - image: Tensor (B, 3, 224, 224)
        - route_cmd: LongTensor (B,) containing command index (0: LEFT, 1: STRAIGHT, 2: RIGHT)
    Outputs:
        - waypoints: Tensor (B, NUM_WAYPOINTS, 2) normalized to [-1, 1] range
    """
    def __init__(self, num_waypoints=NUM_WAYPOINTS, num_commands=len(ROUTE_COMMANDS), embedding_dim=32, pretrained=True):
        super(ConditionedResNet18Waypoints, self).__init__()
        self.num_waypoints = num_waypoints
        
        # 1. ResNet-18 Visual Feature Extractor
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Features output: (B, 512, 1, 1)
        
        # 2. Route Command Embedding Layer
        self.cmd_embedding = nn.Embedding(num_embeddings=num_commands, embedding_dim=embedding_dim)
        
        # 3. Trajectory Regressor Head
        combined_dim = 512 + embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_waypoints * 2)
        )

    def forward(self, image, route_cmd):
        # Extract visual features
        feat = self.backbone(image) # (B, 512, 1, 1)
        feat = torch.flatten(feat, 1) # (B, 512)
        
        # Extract command embedding
        cmd_embed = self.cmd_embedding(route_cmd) # (B, 32)
        
        # Fuse visual & command features
        fused = torch.cat([feat, cmd_embed], dim=1) # (B, 544)
        
        # Output trajectory waypoints
        out = self.fc(fused) # (B, num_waypoints * 2)
        waypoints = out.view(-1, self.num_waypoints, 2)
        return waypoints


def export_lane_model_to_onnx(model, onnx_save_path, device='cuda'):
    """
    Exports ConditionedResNet18Waypoints to ONNX (Opset 11 for Jetson Nano)
    """
    model.eval()
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    dummy_cmd = torch.tensor([1], dtype=torch.long, device=device) # Default STRAIGHT
    
    try:
        torch.onnx.export(
            model,
            (dummy_img, dummy_cmd),
            onnx_save_path,
            verbose=False,
            input_names=['image_input', 'command_input'],
            output_names=['waypoints_output'],
            opset_version=11,
            dynamo=False
        )
    except Exception:
        torch.onnx.export(
            model,
            (dummy_img, dummy_cmd),
            onnx_save_path,
            verbose=False,
            input_names=['image_input', 'command_input'],
            output_names=['waypoints_output'],
            opset_version=11
        )
    print(f"[+] Successfully exported Conditioned Lane Model to ONNX (Opset 11) -> '{onnx_save_path}'")
