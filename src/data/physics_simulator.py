"""Physics-based X-ray simulation for VLM_Glimpse_1000."""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

from config.data_config import DataConfig
from ..physics.beer_lambert import BeerLambertSimulator


@dataclass
class MaterialProperties:
    """Material properties for X-ray simulation."""
    
    name: str
    density: float  # g/cm³
    atomic_number: float
    attenuation_coeff: float  # cm²/g at specific energy
    color: Tuple[int, int, int] = (128, 128, 128)  # RGB color for visualization


@dataclass
class DefectProperties:
    """Defect properties for simulation."""
    
    defect_type: str
    size_range: Tuple[float, float]  # mm
    density_ratio: float  # Relative to base material
    attenuation_ratio: float  # Relative to base material
    shape: str = "circular"  # "circular", "elliptical", "irregular"


class PhysicsSimulator:
    """Physics-based X-ray image simulator."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Initialize material properties
        self.materials = self._initialize_materials()
        
        # Initialize defect properties
        self.defects = self._initialize_defects()
        
        # Beer-Lambert simulator
        self.beer_lambert = BeerLambertSimulator(config.physics if hasattr(config, 'physics') else None)
        
        # Simulation parameters
        self.image_size = config.image_size
        self.pixel_size = 0.1  # mm per pixel
        self.beam_energy = 50.0  # keV
        self.beam_intensity = 1000.0  # photons
        
    def _initialize_materials(self) -> Dict[str, MaterialProperties]:
        """Initialize material properties database."""
        materials = {
            "aluminum": MaterialProperties(
                name="aluminum",
                density=2.70,
                atomic_number=13,
                attenuation_coeff=0.435,
                color=(192, 192, 192)
            ),
            "steel": MaterialProperties(
                name="steel",
                density=7.85,
                atomic_number=26,  # Iron approximation
                attenuation_coeff=0.234,
                color=(105, 105, 105)
            ),
            "copper": MaterialProperties(
                name="copper",
                density=8.96,
                atomic_number=29,
                attenuation_coeff=0.198,
                color=(184, 115, 51)
            ),
            "titanium": MaterialProperties(
                name="titanium",
                density=4.51,
                atomic_number=22,
                attenuation_coeff=0.312,
                color=(135, 135, 135)
            ),
            "lead": MaterialProperties(
                name="lead",
                density=11.34,
                atomic_number=82,
                attenuation_coeff=0.145,
                color=(64, 64, 64)
            ),
            "air": MaterialProperties(
                name="air",
                density=0.001225,
                atomic_number=7.5,  # Effective for air
                attenuation_coeff=0.0,
                color=(255, 255, 255)
            )
        }
        return materials
    
    def _initialize_defects(self) -> Dict[str, DefectProperties]:
        """Initialize defect properties database."""
        defects = {
            "crack": DefectProperties(
                defect_type="crack",
                size_range=(0.1, 5.0),
                density_ratio=0.0,  # Air-filled crack
                attenuation_ratio=0.0,
                shape="irregular"
            ),
            "void": DefectProperties(
                defect_type="void",
                size_range=(0.5, 10.0),
                density_ratio=0.0,  # Empty void
                attenuation_ratio=0.0,
                shape="circular"
            ),
            "inclusion": DefectProperties(
                defect_type="inclusion",
                size_range=(0.2, 3.0),
                density_ratio=1.5,  # Denser foreign material
                attenuation_ratio=1.3,
                shape="irregular"
            ),
            "porosity": DefectProperties(
                defect_type="porosity",
                size_range=(0.1, 1.0),
                density_ratio=0.3,  # Partially filled pores
                attenuation_ratio=0.3,
                shape="circular"
            ),
            "corrosion": DefectProperties(
                defect_type="corrosion",
                size_range=(1.0, 8.0),
                density_ratio=0.7,  # Reduced density due to corrosion
                attenuation_ratio=0.8,
                shape="irregular"
            )
        }
        return defects
    
    def generate_material_map(
        self, 
        material_name: str,
        thickness_range: Tuple[float, float] = (1.0, 5.0),
        add_variation: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate material and thickness maps.
        
        Args:
            material_name: Base material name
            thickness_range: Range of thickness values (mm)
            add_variation: Whether to add thickness variation
            
        Returns:
            Tuple of (material_map, thickness_map)
        """
        H, W = self.image_size
        
        # Create base material map
        material_map = np.full((H, W), list(self.materials.keys()).index(material_name))
        
        # Create thickness map
        base_thickness = np.random.uniform(thickness_range[0], thickness_range[1])
        thickness_map = np.full((H, W), base_thickness, dtype=np.float32)
        
        if add_variation:
            # Add smooth thickness variation
            variation = np.random.uniform(-0.2, 0.2, (H//4, W//4))
            variation = cv2.resize(variation, (W, H), interpolation=cv2.INTER_CUBIC)
            thickness_map += variation * base_thickness
            thickness_map = np.clip(thickness_map, thickness_range[0], thickness_range[1])
        
        return material_map, thickness_map
    
    def add_defects(
        self,
        material_map: np.ndarray,
        thickness_map: np.ndarray,
        defect_specs: List[Dict],
        material_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Add defects to material and thickness maps.
        
        Args:
            material_map: Material index map
            thickness_map: Thickness map
            defect_specs: List of defect specifications
            material_name: Base material name
            
        Returns:
            Updated maps and defect annotations
        """
        H, W = material_map.shape
        defect_annotations = []
        
        for spec in defect_specs:
            defect_type = spec["type"]
            if defect_type not in self.defects:
                continue
            
            defect_props = self.defects[defect_type]
            
            # Random defect parameters
            size = np.random.uniform(*defect_props.size_range)
            center_x = np.random.randint(int(size), W - int(size))
            center_y = np.random.randint(int(size), H - int(size))
            
            # Convert size to pixels
            size_pixels = int(size / self.pixel_size)
            
            # Create defect mask
            mask = self._create_defect_mask(
                (H, W), 
                (center_y, center_x), 
                size_pixels,
                defect_props.shape
            )
            
            # Apply defect to thickness map
            base_material = self.materials[material_name]
            defect_thickness = thickness_map * defect_props.density_ratio
            thickness_map = np.where(mask, defect_thickness, thickness_map)
            
            # Record defect annotation
            bbox = [
                max(0, center_x - size_pixels),
                max(0, center_y - size_pixels),
                min(W, center_x + size_pixels),
                min(H, center_y + size_pixels)
            ]
            
            defect_annotations.append({
                "type": defect_type,
                "center": [center_x, center_y],
                "size": size,
                "bbox": bbox,
                "mask": mask.astype(np.uint8)
            })
        
        return material_map, thickness_map, defect_annotations
    
    def _create_defect_mask(
        self,
        image_shape: Tuple[int, int],
        center: Tuple[int, int],
        size: int,
        shape: str
    ) -> np.ndarray:
        """Create defect mask based on shape."""
        H, W = image_shape
        mask = np.zeros((H, W), dtype=bool)
        
        y, x = np.ogrid[:H, :W]
        center_y, center_x = center
        
        if shape == "circular":
            # Circular defect
            mask = (x - center_x)**2 + (y - center_y)**2 <= size**2
        
        elif shape == "elliptical":
            # Elliptical defect
            aspect_ratio = np.random.uniform(0.3, 3.0)
            a = size
            b = size * aspect_ratio
            angle = np.random.uniform(0, 2*np.pi)
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = (x - center_x) * cos_a + (y - center_y) * sin_a
            y_rot = -(x - center_x) * sin_a + (y - center_y) * cos_a
            
            mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1
        
        elif shape == "irregular":
            # Irregular defect using random perturbation
            angles = np.linspace(0, 2*np.pi, 20)
            radii = size * (0.7 + 0.6 * np.random.random(len(angles)))
            
            # Create polygon points
            points = []
            for angle, radius in zip(angles, radii):
                px = center_x + radius * np.cos(angle)
                py = center_y + radius * np.sin(angle)
                points.append([int(px), int(py)])
            
            # Fill polygon
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask.astype(np.uint8), [points], 1)
            mask = mask.astype(bool)
        
        return mask
    
    def simulate_xray_image(
        self,
        material_map: np.ndarray,
        thickness_map: np.ndarray,
        add_noise: bool = True,
        add_beam_hardening: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate X-ray image using Beer-Lambert law.
        
        Args:
            material_map: Material index map
            thickness_map: Thickness map in mm
            add_noise: Whether to add noise
            add_beam_hardening: Whether to simulate beam hardening
            
        Returns:
            Simulated X-ray image and simulation info
        """
        H, W = material_map.shape
        
        # Initialize intensity map
        intensity_map = np.full((H, W), self.beam_intensity, dtype=np.float32)
        
        # Apply Beer-Lambert law for each material
        unique_materials = np.unique(material_map)
        
        for mat_idx in unique_materials:
            mat_name = list(self.materials.keys())[mat_idx]
            material = self.materials[mat_name]
            
            # Material mask
            mask = material_map == mat_idx
            
            # Calculate attenuation
            thickness = thickness_map[mask]
            linear_att_coeff = material.attenuation_coeff * material.density  # cm⁻¹
            
            # Convert thickness from mm to cm
            thickness_cm = thickness / 10.0
            
            # Apply Beer-Lambert law: I = I₀ * exp(-μ * t)
            attenuation = np.exp(-linear_att_coeff * thickness_cm)
            intensity_map[mask] *= attenuation
        
        # Beam hardening effect (simplified)
        if add_beam_hardening:
            # Simulate spectral changes - higher energy photons penetrate better
            hardening_factor = 0.1 * (1 - intensity_map / self.beam_intensity)
            intensity_map *= (1 + hardening_factor)
        
        # Add noise
        if add_noise:
            # Poisson noise (quantum noise)
            intensity_map = np.random.poisson(intensity_map).astype(np.float32)
            
            # Electronic noise
            electronic_noise = np.random.normal(0, 5, intensity_map.shape)
            intensity_map += electronic_noise
        
        # Convert to image (invert and normalize)
        xray_image = -np.log(np.clip(intensity_map / self.beam_intensity, 1e-6, 1.0))
        xray_image = ((xray_image - xray_image.min()) / 
                      (xray_image.max() - xray_image.min()) * 255).astype(np.uint8)
        
        # Simulation info
        sim_info = {
            "beam_energy": self.beam_energy,
            "beam_intensity": self.beam_intensity,
            "pixel_size": self.pixel_size,
            "materials_used": [list(self.materials.keys())[i] for i in unique_materials],
            "mean_thickness": float(np.mean(thickness_map)),
            "thickness_range": [float(np.min(thickness_map)), float(np.max(thickness_map))]
        }
        
        return xray_image, sim_info
    
    def generate_synthetic_sample(
        self,
        material_name: str = "aluminum",
        num_defects: int = None,
        defect_types: List[str] = None
    ) -> Dict:
        """
        Generate a complete synthetic X-ray sample.
        
        Args:
            material_name: Base material
            num_defects: Number of defects (random if None)
            defect_types: Types of defects (random if None)
            
        Returns:
            Dictionary with image, annotations, and metadata
        """
        # Random parameters if not specified
        if num_defects is None:
            num_defects = np.random.poisson(2)  # Average 2 defects
        
        if defect_types is None:
            available_defects = list(self.defects.keys())
            defect_types = np.random.choice(
                available_defects, 
                size=min(num_defects, len(available_defects)),
                replace=False
            ).tolist()
        
        # Generate base material and thickness maps
        material_map, thickness_map = self.generate_material_map(material_name)
        
        # Add defects
        defect_specs = [{"type": dt} for dt in defect_types[:num_defects]]
        material_map, thickness_map, defect_annotations = self.add_defects(
            material_map, thickness_map, defect_specs, material_name
        )
        
        # Simulate X-ray image
        xray_image, sim_info = self.simulate_xray_image(material_map, thickness_map)
        
        # Create defect mask
        defect_mask = np.zeros(material_map.shape, dtype=np.uint8)
        for defect in defect_annotations:
            defect_mask |= defect["mask"]
        
        # Determine label
        has_defect = len(defect_annotations) > 0
        label = 1 if has_defect else 0
        
        # Create description
        if has_defect:
            defect_list = [d["type"] for d in defect_annotations]
            unique_defects = list(set(defect_list))
            description = f"X-ray image of {material_name} showing {', '.join(unique_defects)}"
        else:
            description = f"X-ray image of {material_name} showing normal structure"
        
        return {
            "image": xray_image,
            "material_map": material_map,
            "thickness_map": thickness_map,
            "defect_mask": defect_mask,
            "defect_annotations": defect_annotations,
            "label": label,
            "material": material_name,
            "thickness": float(np.mean(thickness_map)),
            "defect_types": defect_types[:num_defects],
            "description": description,
            "simulation_info": sim_info
        }
    
    def generate_dataset(
        self,
        num_samples: int,
        save_path: Path,
        materials: List[str] = None,
        defect_probability: float = 0.7
    ) -> None:
        """
        Generate a synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            save_path: Path to save dataset
            materials: List of materials to use
            defect_probability: Probability of having defects
        """
        if materials is None:
            materials = ["aluminum", "steel", "copper", "titanium"]
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        annotations = []
        
        for i in range(num_samples):
            # Random material
            material = np.random.choice(materials)
            
            # Random defects
            has_defects = np.random.random() < defect_probability
            num_defects = np.random.poisson(2) if has_defects else 0
            
            # Generate sample
            sample = self.generate_synthetic_sample(
                material_name=material,
                num_defects=num_defects
            )
            
            # Save image
            image_filename = f"synthetic_{i:06d}.png"
            image_path = save_path / "images" / image_filename
            image_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(image_path), sample["image"])
            
            # Save defect mask if present
            if sample["defect_mask"].any():
                mask_filename = f"mask_{i:06d}.png"
                mask_path = save_path / "masks" / mask_filename
                mask_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(mask_path), sample["defect_mask"] * 255)
            else:
                mask_filename = None
            
            # Create annotation
            annotation = {
                "image_path": f"images/{image_filename}",
                "mask_path": f"masks/{mask_filename}" if mask_filename else None,
                "label": sample["label"],
                "material": sample["material"],
                "thickness": sample["thickness"],
                "defect_types": sample["defect_types"],
                "description": sample["description"],
                "simulation_info": sample["simulation_info"],
                "defect_annotations": [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in defect.items() if k != "mask"}
                    for defect in sample["defect_annotations"]
                ]
            }
            annotations.append(annotation)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        # Save annotations
        annotations_file = save_path / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Generated {num_samples} synthetic samples at {save_path}")
        print(f"Defect samples: {sum(1 for a in annotations if a['label'] == 1)}")
        print(f"Normal samples: {sum(1 for a in annotations if a['label'] == 0)}")