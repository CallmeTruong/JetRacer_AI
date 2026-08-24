import os
import torch
import torch.nn as nn
import torchvision.models as models
from jetracer_ai.urban_traffic.constants import NUM_WAYPOINTS, ROUTE_COMMANDS, COMMAND_TO_INDEX



class ConditionedResNet18Waypoints(nn.Module):
    """
    Conditioned Trajectory Prediction Model (ResNet-18 Backbone).
    Inputs:
        - image: Tensor (B, 3, 224, 224)
        - route_cmd: LongTensor (B,)
    Outputs:
        - waypoints: Tensor (B, NUM_WAYPOINTS, 2) normalized [-1, 1]
    """
    def __init__(self, num_waypoints=NUM_WAYPOINTS, num_commands=len(ROUTE_COMMANDS), embedding_dim=32, pretrained=True):
        super(ConditionedResNet18Waypoints, self).__init__()
        self.num_waypoints = num_waypoints
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cmd_embedding = nn.Embedding(num_embeddings=num_commands, embedding_dim=embedding_dim)
        combined_dim = 512 + embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_waypoints * 2)
        )

    def forward(self, image, route_cmd):
        feat = self.backbone(image)
        feat = torch.flatten(feat, 1)
        cmd_embed = self.cmd_embedding(route_cmd)
        fused = torch.cat([feat, cmd_embed], dim=1)
        out = self.fc(fused)
        waypoints = out.view(-1, self.num_waypoints, 2)
        return waypoints


class ConditionedTrajectoryMobileNetV2(nn.Module):
    """
    Single-Task Trajectory Model (MobileNetV2 Backbone).
    Optimized for Jetson Nano: lightweight, fast inference.

    Inputs:
        - image: Tensor (B, 3, 224, 224)
        - route_cmd: LongTensor (B,) — command index (0:LEFT, 1:STRAIGHT, 2:RIGHT)
    Outputs:
        - waypoints: Tensor (B, NUM_WAYPOINTS, 2) normalized [-1, 1]
          (convert to pixel: px = (x+1)/2 * 224)
    """
    def __init__(self, num_waypoints=NUM_WAYPOINTS, num_commands=len(ROUTE_COMMANDS), embedding_dim=32, pretrained=True):
        super(ConditionedTrajectoryMobileNetV2, self).__init__()
        self.num_waypoints = num_waypoints

        # MobileNetV2 Backbone
        try:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        except Exception:
            mobilenet = models.mobilenet_v2(pretrained=pretrained)

        self.backbone = mobilenet.features          # (B, 1280, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # (B, 1280, 1, 1)

        # Route Command Embedding
        self.cmd_embedding = nn.Embedding(num_embeddings=num_commands, embedding_dim=embedding_dim)

        # Trajectory Waypoint Head (single output head)
        combined_dim = 1280 + embedding_dim
        self.trajectory_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_waypoints * 2)
        )

    def forward(self, image, route_cmd):
        feat = self.backbone(image)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)              # (B, 1280)
        cmd_embed = self.cmd_embedding(route_cmd)  # (B, 32)
        fused = torch.cat([feat, cmd_embed], dim=1) # (B, 1312)
        out = self.trajectory_head(fused)
        waypoints = out.view(-1, self.num_waypoints, 2)
        return waypoints


def export_trajectory_mobilenet_to_onnx(model, onnx_save_path, device='cuda'):
    """
    Exports ConditionedTrajectoryMobileNetV2 to ONNX (single output: waypoints).
    """
    model.eval()
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    dummy_cmd = torch.tensor([1], dtype=torch.long, device=device)

    try:
        torch.onnx.export(
            model,
            (dummy_img, dummy_cmd),
            onnx_save_path,
            input_names=['image', 'route_cmd'],
            output_names=['waypoints'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'route_cmd': {0: 'batch_size'},
                'waypoints': {0: 'batch_size'}
            },
            opset_version=11,
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            (dummy_img, dummy_cmd),
            onnx_save_path,
            input_names=['image', 'route_cmd'],
            output_names=['waypoints'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'route_cmd': {0: 'batch_size'},
                'waypoints': {0: 'batch_size'}
            },
            opset_version=11
        )
    print(f"[+] Exported Trajectory ONNX model → {onnx_save_path}")


def export_lane_model_to_onnx(model, onnx_save_path, device='cuda'):
    """
    Exports ConditionedResNet18Waypoints to ONNX (legacy ResNet18 single-task).
    """
    export_trajectory_mobilenet_to_onnx(model, onnx_save_path, device)
