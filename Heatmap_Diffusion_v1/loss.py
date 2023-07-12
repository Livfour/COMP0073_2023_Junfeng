import torch
import torch.nn as nn

class JointsMSELoss(nn.Module):
    def __init__(self, device="cuda"):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.joints_weight = torch.tensor(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=torch.float32,
            device=device
        ).reshape((-1, 1))

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt) * self.joints_weight[idx]

        return loss / num_joints
