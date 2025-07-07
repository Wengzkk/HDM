import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class HDMModel(nn.Module):
    def __init__(self, **kwargs):
        super(HDMModel, self).__init__()

        self.out_size_h = kwargs['out_size_h']
        self.out_size_w = kwargs['out_size_w']
        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.OOM = kwargs['OOM']

        # Initialize SD model (shared by both expert and apprentice domains)
        self.sd_model = UNetSD(in_channels=3, out_channels=1)

        # Initialize classifier for image-level anomaly detection
        self.classifier = Classifier(in_channels=1)

    def forward(self, x, mask=None):
        """
        Forward pass with optional mask-based perturbation.
        """
        # DAGM: Apply perturbation if mask is provided
        if mask is not None:
            x = self.apply_perturbation(x, mask)

        # Extract features using the SD model
        sd_output = self.sd_model(x)

        # Calculate image-level anomaly score using classifier
        anomaly_score = self.classifier(sd_output)

        return {'SD': sd_output, 'Score': anomaly_score}

    def apply_perturbation(self, x, mask):
        """
        DAGM module: apply noise and disturbances based on the input mask.
        """
        noise = torch.randn_like(x) * 0.1
        disturbance = torch.sin(x * 3.14) * 0.05

        # Apply mask-based perturbation
        perturbed_x = x * (1 - mask) + (noise + disturbance) * mask
        return perturbed_x

    def cal_loss(self, sd_output, gt_mask, anomaly_score, gt_label):
        """
        Compute loss based on the SD model's output, ground truth mask, and classifier output.
        """
        # Pixel-wise loss for anomaly map
        sd_loss = torch.mean((sd_output - gt_mask) ** 2)

        # Classification loss for image-level anomaly detection
        classification_loss = F.binary_cross_entropy_with_logits(anomaly_score, gt_label)

        # Total loss
        total_loss = sd_loss + classification_loss

        return total_loss

    @torch.no_grad()
    def cal_am(self, sd_output):
        """
        Calculate anomaly map based on SD model output.
        """
        anomaly_map = F.interpolate(sd_output, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
        anomaly_map = anomaly_map.squeeze(1).cpu().numpy()

        # Smooth the anomaly map using Gaussian filter
        am_np_list = []
        for i in range(anomaly_map.shape[0]):
            smoothed_map = gaussian_filter(anomaly_map[i], sigma=4)
            am_np_list.append(smoothed_map)

        return am_np_list

    def save(self, path, metric):
        torch.save({'sd_model': self.sd_model.state_dict(),
                    'classifier': self.classifier.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.sd_model.load_state_dict(checkpoint['sd_model'])
        self.classifier.load_state_dict(checkpoint['classifier'])

    def train_mode(self):
        self.sd_model.train()
        self.classifier.train()

    def eval_mode(self):
        self.sd_model.eval()
        self.classifier.eval()


class UNetSD(nn.Module):
    """
    U-Net structure for SD module.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetSD, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder blocks
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Decoder blocks
        for feature in reversed(features):
            self.decoder.append(self._block(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx, layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            skip_connection = skip_connections[idx]
            x = torch.cat((x, skip_connection), dim=1)
            x = layer(x)

        return self.final_conv(x)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Classifier(nn.Module):
    """
    Simple classifier for image-level anomaly detection.
    """
    def __init__(self, in_channels=1):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 16 * 16, 1)  # Assuming input size 256x256

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
