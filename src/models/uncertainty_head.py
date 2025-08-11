"""Uncertainty quantification heads for VLM_Glimpse_1000."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

from config.model_config import UncertaintyConfig


class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply dropout during both training and inference."""
        if training or self.training:
            return self.dropout(x)
        else:
            # Apply dropout even during inference for MC sampling
            return F.dropout(x, p=self.dropout_rate, training=True)


class EvidentialHead(nn.Module):
    """Evidential learning head for uncertainty quantification."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Evidence network
        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
            nn.Softplus()  # Ensure positive evidence
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for evidential learning.
        
        Args:
            features: Input features [B, D]
            
        Returns:
            Dictionary with evidential outputs
        """
        # Get evidence (positive values)
        evidence = self.evidence_net(features)  # [B, K]
        
        # Dirichlet parameters (alpha = evidence + 1)
        alpha = evidence + 1.0
        
        # Probability (expected value of Dirichlet)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        
        # Uncertainty measures
        S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
        uncertainty = self.num_classes / S  # Total uncertainty
        
        # Aleatoric and epistemic uncertainty
        expected_prob = alpha / S
        aleatoric = torch.sum(expected_prob * (1 - expected_prob) / (S + 1), dim=1, keepdim=True)
        epistemic = uncertainty - aleatoric
        
        return {
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "uncertainty": uncertainty,
            "aleatoric": aleatoric,
            "epistemic": epistemic
        }
    
    def evidential_loss(
        self, 
        alpha: torch.Tensor, 
        targets: torch.Tensor, 
        epoch: int,
        annealing_coeff: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evidential loss with KL regularization.
        
        Args:
            alpha: Dirichlet parameters [B, K]
            targets: Ground truth labels [B]
            epoch: Current training epoch
            annealing_coeff: Annealing coefficient for KL term
            
        Returns:
            Dictionary with loss components
        """
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Dirichlet strength
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Expected log-likelihood
        digamma_alpha = torch.digamma(alpha)
        digamma_S = torch.digamma(S)
        
        likelihood_loss = torch.sum(
            targets_onehot * (digamma_S - digamma_alpha), dim=1
        ).mean()
        
        # KL divergence regularization
        # KL(Dir(α) || Dir(1)) where Dir(1) is uniform prior
        kl_alpha = alpha - 1.0
        kl_loss = torch.sum(
            kl_alpha * (digamma_alpha - digamma_S), dim=1
        ).mean()
        
        # Annealing weight for KL term
        annealing_weight = min(1.0, epoch * annealing_coeff)
        
        # Total loss
        total_loss = likelihood_loss + annealing_weight * kl_loss
        
        return {
            "total": total_loss,
            "likelihood": likelihood_loss,
            "kl": kl_loss,
            "annealing_weight": torch.tensor(annealing_weight)
        }


class EnsembleHead(nn.Module):
    """Ensemble of prediction heads for uncertainty estimation."""
    
    def __init__(self, input_dim: int, num_classes: int, ensemble_size: int = 5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_classes = num_classes
        
        # Create ensemble of prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, num_classes)
            )
            for _ in range(ensemble_size)
        ])
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble heads.
        
        Args:
            features: Input features [B, D]
            
        Returns:
            Dictionary with ensemble outputs
        """
        # Get predictions from all heads
        logits_list = []
        probs_list = []
        
        for head in self.heads:
            logits = head(features)
            probs = F.softmax(logits, dim=1)
            logits_list.append(logits)
            probs_list.append(probs)
        
        # Stack predictions
        all_logits = torch.stack(logits_list, dim=1)  # [B, E, K]
        all_probs = torch.stack(probs_list, dim=1)   # [B, E, K]
        
        # Ensemble statistics
        mean_logits = torch.mean(all_logits, dim=1)  # [B, K]
        mean_probs = torch.mean(all_probs, dim=1)    # [B, K]
        
        # Uncertainty measures
        # Predictive entropy (total uncertainty)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-8), dim=1, keepdim=True
        )
        
        # Expected entropy (aleatoric uncertainty)
        individual_entropies = -torch.sum(
            all_probs * torch.log(all_probs + 1e-8), dim=2
        )  # [B, E]
        expected_entropy = torch.mean(individual_entropies, dim=1, keepdim=True)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        
        return {
            "all_logits": all_logits,
            "all_probs": all_probs,
            "mean_logits": mean_logits,
            "mean_probs": mean_probs,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_info": mutual_info
        }


class UncertaintyHead(nn.Module):
    """Multi-method uncertainty quantification head."""
    
    def __init__(self, config: UncertaintyConfig, input_dim: int, num_classes: int):
        super().__init__()
        self.config = config
        self.methods = config.methods
        self.num_classes = num_classes
        
        # Initialize uncertainty methods
        self.uncertainty_modules = nn.ModuleDict()
        
        if "monte_carlo" in self.methods:
            self.uncertainty_modules["monte_carlo"] = MonteCarloDropout(config.mc_dropout_rate)
            self.mc_classifier = nn.Linear(input_dim, num_classes)
        
        if "evidential" in self.methods:
            self.uncertainty_modules["evidential"] = EvidentialHead(input_dim, num_classes)
        
        if "ensemble" in self.methods:
            self.uncertainty_modules["ensemble"] = EnsembleHead(
                input_dim, num_classes, config.ensemble_size
            )
        
        # Temperature scaling for calibration
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self, 
        features: torch.Tensor,
        num_mc_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through uncertainty methods.
        
        Args:
            features: Input features [B, D]
            num_mc_samples: Number of MC samples (if None, use config)
            
        Returns:
            Dictionary with outputs from each uncertainty method
        """
        outputs = {}
        
        # Monte Carlo Dropout
        if "monte_carlo" in self.methods:
            if num_mc_samples is None:
                num_mc_samples = self.config.num_mc_samples
            
            mc_outputs = self._monte_carlo_forward(features, num_mc_samples)
            outputs["monte_carlo"] = mc_outputs
        
        # Evidential Learning
        if "evidential" in self.methods:
            evidential_outputs = self.uncertainty_modules["evidential"](features)
            outputs["evidential"] = evidential_outputs
        
        # Ensemble
        if "ensemble" in self.methods:
            ensemble_outputs = self.uncertainty_modules["ensemble"](features)
            outputs["ensemble"] = ensemble_outputs
        
        return outputs
    
    def _monte_carlo_forward(
        self, 
        features: torch.Tensor, 
        num_samples: int
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo forward pass."""
        mc_outputs = []
        
        for _ in range(num_samples):
            # Apply MC dropout
            dropped_features = self.uncertainty_modules["monte_carlo"](features, training=True)
            # Get prediction
            logits = self.mc_classifier(dropped_features)
            probs = F.softmax(logits, dim=1)
            mc_outputs.append(probs)
        
        # Stack all samples
        all_probs = torch.stack(mc_outputs, dim=1)  # [B, S, K]
        
        # Calculate statistics
        mean_probs = torch.mean(all_probs, dim=1)  # [B, K]
        var_probs = torch.var(all_probs, dim=1)    # [B, K]
        
        # Uncertainty measures
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-8), dim=1, keepdim=True
        )
        
        # Expected entropy
        individual_entropies = -torch.sum(
            all_probs * torch.log(all_probs + 1e-8), dim=2
        )  # [B, S]
        expected_entropy = torch.mean(individual_entropies, dim=1, keepdim=True)
        
        # Mutual information
        mutual_info = predictive_entropy - expected_entropy
        
        return {
            "all_probs": all_probs,
            "mean_probs": mean_probs,
            "var_probs": var_probs,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_info": mutual_info
        }
    
    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling for calibration."""
        if hasattr(self, "temperature"):
            return logits / self.temperature
        return logits
    
    def compute_uncertainty_metrics(
        self, 
        outputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute aggregated uncertainty metrics across methods."""
        metrics = {}
        
        # Collect uncertainties from all methods
        uncertainties = []
        
        for method, method_outputs in outputs.items():
            if "predictive_entropy" in method_outputs:
                uncertainties.append(method_outputs["predictive_entropy"])
            elif "uncertainty" in method_outputs:
                uncertainties.append(method_outputs["uncertainty"])
        
        if uncertainties:
            # Average uncertainty across methods
            avg_uncertainty = torch.mean(torch.stack(uncertainties, dim=0), dim=0)
            metrics["avg_uncertainty"] = avg_uncertainty
            
            # Uncertainty agreement (lower is better)
            if len(uncertainties) > 1:
                uncertainty_std = torch.std(torch.stack(uncertainties, dim=0), dim=0)
                metrics["uncertainty_disagreement"] = uncertainty_std
        
        return metrics