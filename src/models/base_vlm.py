"""Base Vision-Language Model for VLM_Glimpse_1000."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from timm import create_model
from typing import Dict, Optional, Tuple
import lightning as L
from einops import rearrange, repeat

from config.model_config import ModelConfig


class VisionEncoder(nn.Module):
    """Vision encoder using Swin Transformer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained vision model
        self.backbone = create_model(
            config.vision.model_name,
            pretrained=config.vision.pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling
        )
        
        # Feature dimensions
        self.feature_dim = config.vision.embed_dim
        
        # Projection layer
        self.projection = nn.Linear(self.feature_dim, config.fusion.projection_dim)
        
        # Freeze backbone if specified
        if config.vision.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Visual features [B, N, D] where N is number of patches
        """
        # Extract features from backbone
        features = self.backbone.forward_features(images)
        
        # Handle different output formats
        if isinstance(features, tuple):
            features = features[0]
        
        # Ensure we have the right shape [B, N, D]
        if len(features.shape) == 4:  # [B, C, H, W]
            B, C, H, W = features.shape
            features = rearrange(features, "b c h w -> b (h w) c")
        
        # Project to fusion dimension
        features = self.projection(features)
        
        return features


class LanguageEncoder(nn.Module):
    """Language encoder using BERT-based models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained language model
        self.backbone = AutoModel.from_pretrained(
            config.language.model_name,
            add_pooling_layer=False
        )
        
        # Feature dimensions
        self.feature_dim = config.language.hidden_size
        
        # Projection layer
        self.projection = nn.Linear(self.feature_dim, config.fusion.projection_dim)
        
        # Freeze backbone if specified
        if config.language.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through language encoder.
        
        Args:
            input_ids: Tokenized text [B, L]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Text features [B, L, D] where L is sequence length
        """
        # Get hidden states from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract last hidden states
        features = outputs.last_hidden_state  # [B, L, D]
        
        # Project to fusion dimension
        features = self.projection(features)
        
        return features


class CrossModalAttention(nn.Module):
    """Cross-modal attention for vision-language fusion."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.fusion_dim = config.fusion.projection_dim
        self.num_heads = 8
        self.head_dim = self.fusion_dim // self.num_heads
        
        # Multi-head attention layers
        self.vision_to_text_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            dropout=config.fusion.attention_dropout,
            batch_first=True
        )
        
        self.text_to_vision_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            dropout=config.fusion.attention_dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.vision_ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(config.fusion.attention_dropout),
            nn.Linear(self.fusion_dim * 4, self.fusion_dim)
        )
        
        self.text_ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(config.fusion.attention_dropout),
            nn.Linear(self.fusion_dim * 4, self.fusion_dim)
        )
        
        # Layer normalization
        self.vision_ln1 = nn.LayerNorm(self.fusion_dim)
        self.vision_ln2 = nn.LayerNorm(self.fusion_dim)
        self.text_ln1 = nn.LayerNorm(self.fusion_dim)
        self.text_ln2 = nn.LayerNorm(self.fusion_dim)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention forward pass.
        
        Args:
            vision_features: Vision features [B, N, D]
            text_features: Text features [B, L, D]
            text_mask: Text attention mask [B, L]
            
        Returns:
            Tuple of (attended_vision, attended_text)
        """
        # Vision attending to text
        vision_attended, _ = self.vision_to_text_attn(
            query=vision_features,
            key=text_features,
            value=text_features,
            key_padding_mask=~text_mask if text_mask is not None else None
        )
        vision_features = self.vision_ln1(vision_features + vision_attended)
        vision_features = self.vision_ln2(vision_features + self.vision_ffn(vision_features))
        
        # Text attending to vision
        text_attended, _ = self.text_to_vision_attn(
            query=text_features,
            key=vision_features,
            value=vision_features
        )
        text_features = self.text_ln1(text_features + text_attended)
        text_features = self.text_ln2(text_features + self.text_ffn(text_features))
        
        return vision_features, text_features


class BaseVLM(L.LightningModule):
    """Base Vision-Language Model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Encoders
        self.vision_encoder = VisionEncoder(config)
        self.language_encoder = LanguageEncoder(config)
        
        # Cross-modal fusion
        if config.fusion.fusion_type == "cross_attention":
            self.fusion = CrossModalAttention(config)
        else:
            raise NotImplementedError(f"Fusion type {config.fusion.fusion_type} not implemented")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion.projection_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(config.language.model_name)
    
    def forward(
        self, 
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLM.
        
        Args:
            images: Input images [B, C, H, W]
            input_ids: Tokenized text [B, L]
            attention_mask: Text attention mask [B, L]
            
        Returns:
            Dictionary containing logits and features
        """
        # Encode vision and language
        vision_features = self.vision_encoder(images)  # [B, N, D]
        text_features = self.language_encoder(input_ids, attention_mask)  # [B, L, D]
        
        # Cross-modal fusion
        fused_vision, fused_text = self.fusion(vision_features, text_features, attention_mask)
        
        # Global pooling
        vision_global = fused_vision.mean(dim=1)  # [B, D]
        text_global = fused_text.mean(dim=1)  # [B, D]
        
        # Concatenate for classification
        fused_features = torch.cat([vision_global, text_global], dim=1)  # [B, 2D]
        
        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        return {
            "logits": logits,
            "vision_features": fused_vision,
            "text_features": fused_text,
            "fused_features": fused_features
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        loss = self.criterion(outputs["logits"], batch["labels"])
        
        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self.forward(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        loss = self.criterion(outputs["logits"], batch["labels"])
        
        # Calculate accuracy
        preds = torch.argmax(outputs["logits"], dim=1)
        acc = (preds == batch["labels"]).float().mean()
        
        # Logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }