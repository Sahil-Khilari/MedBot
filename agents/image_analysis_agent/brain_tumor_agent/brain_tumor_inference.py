import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"BrainTumorAgent using device: {DEVICE}")


class BrainTumorClassifier:
    """DenseNet121 classifier for brain tumor type detection from MRI scans."""

    CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    def __init__(self, model_path: str, device=None):
        self.model_path = model_path
        self.device = device or DEVICE
        self._demo_mode = False
        self.model = self._build_model()
        self._load_model_weights()
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self, pretrained_backbone: bool = False) -> nn.Module:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(self.CLASS_NAMES))
        return model

    def _load_model_weights(self):
        if not os.path.exists(self.model_path):
            # Fall back to ImageNet pretrained backbone; classifier head is randomly initialized.
            # Predictions run but are NOT clinically accurate — for demo/testing only.
            logger.warning(
                f"[BrainTumorClassifier] Model file not found at '{self.model_path}'. "
                "Falling back to ImageNet pretrained backbone (demo mode — not clinically accurate)."
            )
            self.model = self._build_model(pretrained_backbone=True)
            self.model.to(self.device)
            self.model.eval()
            self._demo_mode = True
            return
        self._demo_mode = False
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"[BrainTumorClassifier] Loaded weights from '{self.model_path}'")
        except Exception as exc:
            logger.warning(
                f"[BrainTumorClassifier] Failed to load weights: {exc}. "
                "Classification will be unavailable."
            )
            self.model = None

    def predict(self, img_path: str):
        """Returns predicted class name string (suffixed '[demo]' in fallback mode), or None if unavailable."""
        if self.model is None:
            logger.warning("[BrainTumorClassifier] Model not loaded; skipping classification.")
            return None
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                _, pred = torch.max(out, 1)
            predicted_class = self.CLASS_NAMES[pred.cpu().item()]
            if self._demo_mode:
                predicted_class = f"{predicted_class} (demo — not clinically accurate)"
            logger.info(f"[BrainTumorClassifier] Predicted: {predicted_class}")
            return predicted_class
        except Exception as exc:
            logger.error(f"[BrainTumorClassifier] Prediction error: {exc}")
            return None


class UNet(nn.Module):
    """U-Net model for image segmentation."""

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path (encoder)
        self.conv1 = nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path (decoder)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))
        x5 = F.relu(self.conv5(self.pool(x4)))

        x6 = F.relu(self.upconv1(x5))
        x6 = torch.cat([x4, x6], dim=1)
        x6 = F.relu(self.conv6(x6))
        x7 = F.relu(self.upconv2(x6))
        x7 = torch.cat([x3, x7], dim=1)
        x7 = F.relu(self.conv7(x7))
        x8 = F.relu(self.upconv3(x7))
        x8 = torch.cat([x2, x8], dim=1)
        x8 = F.relu(self.conv8(x8))
        x9 = F.relu(self.upconv4(x8))
        x9 = torch.cat([x1, x9], dim=1)
        x9 = F.relu(self.conv9(x9))
        return self.conv10(x9)


class BrainTumorSegmentation:
    """U-Net segmentation to localise tumor region in brain MRI scans."""

    def __init__(self, model_path: str, device=None):
        self.model_path = model_path
        self.device = device or DEVICE
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(
                f"[BrainTumorSegmentation] Model file not found at '{self.model_path}'. "
                "Segmentation will be unavailable until the model is provided."
            )
            return None
        try:
            model = UNet(n_channels=3, n_classes=1).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Support both raw state_dict and {'state_dict': ...} checkpoint formats
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info(f"[BrainTumorSegmentation] Loaded weights from '{self.model_path}'")
            return model
        except Exception as exc:
            logger.warning(
                f"[BrainTumorSegmentation] Failed to load model: {exc}. "
                "Segmentation will be unavailable."
            )
            return None

    def _overlay_mask(self, img: np.ndarray, mask: np.ndarray, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mask_stacked = np.stack((mask,) * 3, axis=-1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(mask_stacked, alpha=0.4)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[BrainTumorSegmentation] Overlay saved to '{output_path}'")
            return True
        except Exception as exc:
            logger.error(f"[BrainTumorSegmentation] Overlay error: {exc}")
            return False

    def predict(self, image_path: str, output_path: str) -> bool:
        """Segment tumor region. Returns True on success, False on failure or unavailable model."""
        if self.model is None:
            logger.warning("[BrainTumorSegmentation] Model not loaded; skipping segmentation.")
            return False
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img_resized = cv2.resize(img, (256, 256))
            tensor = (
                torch.Tensor(img_resized)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to(self.device)
            )
            with torch.no_grad():
                mask = self.model(tensor).squeeze().cpu().numpy()
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            return self._overlay_mask(img, mask_resized, output_path)
        except Exception as exc:
            logger.error(f"[BrainTumorSegmentation] Segmentation error: {exc}")
            return False


class BrainTumorAgent:
    """
    Orchestrates classification then (conditional) segmentation for brain MRI analysis.
    Both sub-agents degrade gracefully when model files are absent.
    """

    def __init__(self, classification_model_path: str, segmentation_model_path: str, device=None):
        self.classifier = BrainTumorClassifier(
            model_path=classification_model_path, device=device
        )
        self.segmenter = BrainTumorSegmentation(
            model_path=segmentation_model_path, device=device
        )

    def analyze(self, image_path: str, segmentation_output_path: str) -> dict:
        """
        Run classify-then-segment pipeline.

        Returns:
            {
                'tumor_type': str | None,  # predicted class, or None if classifier unavailable
                'segmented': bool,          # True if overlay was saved successfully
                'output_path': str | None   # path to saved overlay, or None
            }
        """
        tumor_type = self.classifier.predict(image_path)

        if tumor_type is None:
            return {'tumor_type': None, 'segmented': False, 'output_path': None}

        if tumor_type == 'no_tumor':
            return {'tumor_type': 'no_tumor', 'segmented': False, 'output_path': None}

        segmented = self.segmenter.predict(image_path, segmentation_output_path)
        output_path = segmentation_output_path if segmented else None
        return {
            'tumor_type': tumor_type,
            'segmented': segmented,
            'output_path': output_path
        }
