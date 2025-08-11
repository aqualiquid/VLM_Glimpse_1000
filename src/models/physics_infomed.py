"""Physics-Informed Vision-Language Model for VLM_Glimpse_1000."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from .base_vlm import BaseVLM
from ..physics.beer_lambert import BeerLambertLayer
from ..physics.defect_physics import DefectPhysicsModule
from config.model_config import ModelConfig


class PhysicsConstraintLayer(nn.Module):
    """Physics constraint layer for X-ray attenuation modeling."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.physics_weight = config.physics.physics_weight
        
        # Beer-Lambert law implementation
        self.beer_lambert = BeerLambertLayer(config.physics)
        
        # Defect physics modeling
        self.defect_physics = DefectPhysicsModule(config.physics)
        
        # Learnable physics parameters
        self.thickness_predictor = nn.Sequential(
            nn.Linear(config.fusion.projection_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive thickness
        )
        
        self.material_classifier = nn.Sequential(
            nn.Linear(config.fusion.projection_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(config.physics.attenuation_coeffs))
        )
        
        # Physics-aware feature refinement
        self.physics_refiner = nn.Sequential(
            nn.Linear(config.fusion.projection_dim + 2, config.fusion.projection_dim),
            nn.LayerNorm(config.fusion.projection_dim),
            nn.ReLU(),
            nn.Linear(config.fusion.projection_dim, config.fusion.projection_dim)
        )
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        x_ray_intensity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply physics constraints to vision features.
        
        Args:
            vision_features: Vision features [B, N, D]
            x_ray_intensity: X-ray intensity values [B, N] (optional)
            
        Returns:
            Dictionary containing refined features and physics predictions
        """
        B, N, D = vision_features.shape
        
        # Predict material properties from features
        material_logits = self.material_classifier(vision_features)  # [B, N, num_materials]
        material_probs = F.softmax(material_logits, dim=-1)
        
        # Predict thickness
        thickness = self.thickness_predictor(vision_features).squeeze(-1)  # [B, N]
        
        # Calculate physics-based attenuation
        attenuation = self.beer_lambert(
            thickness=thickness,
            material_probs=material_probs,
            energy_keV=self.config.physics.energy_keV
        )
        
        # Physics-guided feature refinement
        physics_info = torch.stack([thickness, attenuation], dim=-1)  # [B, N, 2]
        enhanced_features = torch.cat([vision_features, physics_info], dim=-1)  # [B, N, D+2]
        refined_features = self.physics_refiner(enhanced_features)
        
        # Calculate physics consistency loss
        physics_loss = 0.0
        if x_ray_intensity is not None:
            # Compare predicted vs actual attenuation
            predicted_intensity = torch.exp(-attenuation)
            physics_loss = F.mse_loss(predicted_intensity, x_ray_intensity)
        
        return {
            "refined_features": refined_features,
            "material_logits": material_logits,
            "thickness": thickness,
            "attenuation": attenuation,
            "physics_loss": physics_loss
        }


class DefectLocalizationHead(nn.Module):
    """Defect localization head with physics awareness."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.fusion.projection_dim
        
        # Localization network
        self.localization_net = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # Defect probability map
            nn.Sigmoid()
        )
        
        # Physics-based defect scoring
        self.defect_scorer = nn.Sequential(
            nn.Linear(self.feature_dim + 3, 128),  # +3 for physics features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        physics_info: Dict[str, torch.Tensor],
        spatial_size: Tuple[int, int] = (14, 14)
    ) -> Dict[str, torch.Tensor]:
        """
        Generate defect localization maps.
        
        Args:
            features: Refined vision features [B, N, D]
            physics_info: Physics information dictionary
            spatial_size: Spatial dimensions for feature map
            
        Returns:
            Dictionary containing localization results
        """
        B, N, D = features.shape
        H, W = spatial_size
        
        # Reshape features to spatial format
        spatial_features = features.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # Generate defect probability map
        defect_map = self.localization_net(spatial_features)  # [B, 1, H, W]
        
        # Physics-enhanced defect scoring
        thickness = physics_info["thickness"].view(B, H, W, 1)
        attenuation = physics_info["attenuation"].view(B, H, W, 1)
        material_entropy = torch.sum(
            -physics_info["material_logits"] * torch.log(physics_info["material_logits"] + 1e-8),
            dim=-1, keepdim=True
        ).view(B, H, W, 1)
        
        # Combine features with physics
        physics_features = torch.cat([thickness, attenuation, material_entropy], dim=-1)  # [B, H, W, 3]
        combined_features = torch.cat([
            features.view(B, H, W, D),
            physics_features
        ], dim=-1)  # [B, H, W, D+3]
        
        # Score each spatial location
        defect_scores = self.defect_scorer(combined_features)  # [B, H, W, 1]
        defect_scores = defect_scores.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        return {
            "defect_map": defect_map,
            "defect_scores": defect_scores,
            "combined_defects": defect_map * defect_scores
        }


class PhysicsInformedVLM(BaseVLM):
    """Physics-Informed Vision-Language Model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Physics constraint layer
        self.physics_layer = PhysicsConstraintLayer(config)
        
        # Defect localization head
        self.defect_head = DefectLocalizationHead(config)
        
        # Enhanced classifier with physics features
        self.enhanced_classifier = nn.Sequential(
            nn.Linear(config.fusion.projection_dim * 2 + 64, config.hidden_dim),  # +64 for physics summary
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Physics summary network
        self.physics_summarizer = nn.Sequential(
            nn.Linear(len(config.physics.attenuation_coeffs) + 2, 128),  # materials + thickness + attenuation
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Multi-task loss weights
        self.classification_weight = 1.0
        self.physics_weight = config.physics.physics_weight
        self.localization_weight = 0.5
    
    def forward(
        self, 
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_ray_intensity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Physics-Informed VLM.
        
        Args:
            images: Input images [B, C, H, W]
            input_ids: Tokenized text [B, L]
            attention_mask: Text attention mask [B, L]
            x_ray_intensity: X-ray intensity values [B, N] (optional)
            
        Returns:
            Dictionary containing all outputs
        """
        # Base VLM forward pass
        base_outputs = super().forward(images, input_ids, attention_mask)
        
        # Apply physics constraints to vision features
        physics_outputs = self.physics_layer(
            base_outputs["vision_features"],
            x_ray_intensity
        )
        
        # Defect localization
        localization_outputs = self.defect_head(
            physics_outputs["refined_features"],
            physics_outputs
        )
        
        # Enhanced classification with physics
        # Summarize physics information
        physics_summary_input = torch.cat([
            physics_outputs["material_logits"].mean(dim=1),  # [B, num_materials]
            physics_outputs["thickness"].mean(dim=1, keepdim=True),  # [B, 1]
            physics_outputs["attenuation"].mean(dim=1, keepdim=True)  # [B, 1]
        ], dim=1)
        physics_summary = self.physics_summarizer(physics_summary_input)  # [B, 64]
        
        # Enhanced features for classification
        enhanced_fused = torch.cat([
            base_outputs["fused_features"],
            physics_summary
        ], dim=1)  # [B, 2D + 64]
        
        # Enhanced classification
        enhanced_logits = self.enhanced_classifier(enhanced_fused)
        
        return {
            **base_outputs,
            **physics_outputs,
            **localization_outputs,
            "enhanced_logits": enhanced_logits,
            "physics_summary": physics_summary
        }
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss including physics constraints.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Classification loss
        classification_loss = self.criterion(outputs["enhanced_logits"], batch["labels"])
        losses["classification"] = classification_loss
        
        # Physics consistency loss
        if outputs["physics_loss"] > 0:
            losses["physics"] = outputs["physics_loss"]
        else:
            losses["physics"] = torch.tensor(0.0, device=classification_loss.device)
        
        # Defect localization loss (if ground truth available)
        if "defect_masks" in batch:
            defect_target = F.interpolate(
                batch["defect_masks"].float(),
                size=outputs["defect_map"].shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            localization_loss = F.binary_cross_entropy(
                outputs["defect_map"].squeeze(1),
                defect_target.squeeze(1)
            )
            losses["localization"] = localization_loss
        else:
            losses["localization"] = torch.tensor(0.0, device=classification_loss.device)
        
        # Total loss
        total_loss = (
            self.classification_weight * losses["classification"] +
            self.physics_weight * losses["physics"] +
            self.localization_weight * losses["localization"]
        )
        losses["total"] = total_loss
        
        return losses
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with physics-informed loss."""
        outputs = self.forward(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            x_ray_intensity=batch.get("x_ray_intensity")
        )
        
        losses = self.compute_loss(outputs, batch)
        
        # Logging
        for loss_name, loss_value in losses.items():
            self.log(f"train_{loss_name}_loss", loss_value, on_step=True, on_epoch=True)
        
        return losses["total"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with physics-informed metrics."""
        outputs = self.forward(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            x_ray_intensity=batch.get("x_ray_intensity")
        )
        
        losses = self.compute_loss(outputs, batch)
        
        # Calculate accuracy
        preds = torch.argmax(outputs["enhanced_logits"], dim=1)
        acc = (preds == batch["labels"]).float().mean()
        
        # Physics metrics
        if "material_labels" in batch:
            material_preds = torch.argmax(outputs["material_logits"], dim=-1)
            material_acc = (material_preds == batch["material_labels"]).float().mean()
            self.log("val_material_acc", material_acc, on_epoch=True)
        
        # Logging
        for loss_name, loss_value in losses.items():
            self.log(f"val_{loss_name}_loss", loss_value, on_epoch=True)
        
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return losses["total"]