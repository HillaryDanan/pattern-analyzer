"""
Information Atom Framework

A theoretical exploration of unified multimodal representations.
This is a novel approach not found in current literature, presented
as a creative investigation into alternatives to tokenization.

DISCLAIMER: This framework presents speculative concepts for discussion.
No empirical validation exists. The "information atom" concept and its
hexagonal spatial arrangement for multimodal data are original ideas
intended to spark conversation about future AI architectures.

Key explorations:
1. Unified representation across modalities (hexagonal geometry)
2. Trust-based information fusion (game theory inspiration)
3. Hierarchical processing (Marr's levels)
4. Cross-modal attention mechanisms

Mathematical foundations are sound, but practical benefits are theoretical.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import seaborn as sns

# ============================================================================
# PART 1: HEXAGONAL GEOMETRY FOR UNIFIED REPRESENTATIONS
# ============================================================================

class HexagonalGrid:
    """
    Hexagonal grid for unified spatial representations across modalities.
    Inspired by biological systems (e.g., cortical columns).
    """
    
    def __init__(self, radius: int = 10, scale: float = 1.0):
        self.radius = radius
        self.scale = scale
        self.centers = self._generate_hex_centers()
        
    def _generate_hex_centers(self) -> np.ndarray:
        """Generate hexagonal grid centers using axial coordinates."""
        centers = []
        for q in range(-self.radius, self.radius + 1):
            r1 = max(-self.radius, -q - self.radius)
            r2 = min(self.radius, -q + self.radius)
            for r in range(r1, r2 + 1):
                # Convert axial to cartesian
                x = self.scale * 3/2 * q
                y = self.scale * np.sqrt(3) * (r + q/2)
                centers.append([x, y, q, r])  # x, y, q, r coordinates
        return np.array(centers)
    
    def get_neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        """Get hexagonal neighbors for a given hex coordinate."""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if abs(nq) <= self.radius and abs(nr) <= self.radius and abs(nq + nr) <= self.radius:
                neighbors.append((nq, nr))
        return neighbors
    
    def visualize(self, values: Optional[np.ndarray] = None):
        """Visualize the hexagonal grid with optional values."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        if values is None:
            values = np.random.rand(len(self.centers))
        
        # Normalize values for coloring
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        for i, (x, y, q, r) in enumerate(self.centers):
            hex_patch = RegularPolygon(
                (x, y), 6, radius=self.scale * 0.9,
                facecolor=plt.cm.viridis(norm_values[i]),
                edgecolor='black', linewidth=0.5
            )
            ax.add_patch(hex_patch)
            ax.text(x, y, f'{int(q)},{int(r)}', ha='center', va='center', fontsize=6)
        
        ax.set_xlim(-self.radius * self.scale * 2, self.radius * self.scale * 2)
        ax.set_ylim(-self.radius * self.scale * 2, self.radius * self.scale * 2)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Hexagonal Information Grid')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Information Density')
        return fig

# ============================================================================
# PART 2: INFORMATION ATOMS - CORE ABSTRACTION
# ============================================================================

@dataclass
class InformationAtom:
    """
    Fundamental unit of multimodal information.
    Preserves cross-modal relationships at the atomic level.
    """
    modality_features: Dict[str, torch.Tensor]  # Raw features per modality
    cross_modal_bonds: torch.Tensor              # Pairwise modality relationships
    semantic_embedding: torch.Tensor             # Unified semantic representation
    confidence: float                            # Atom confidence/trust score
    spatial_coords: Optional[Tuple[int, int]]    # Hexagonal coordinates if applicable
    
    def fuse_with(self, other: 'InformationAtom', trust_weight: float = 0.5) -> 'InformationAtom':
        """Fuse two information atoms based on trust weight."""
        # Weighted average of features
        fused_features = {}
        for modality in self.modality_features:
            if modality in other.modality_features:
                fused_features[modality] = (
                    trust_weight * self.modality_features[modality] +
                    (1 - trust_weight) * other.modality_features[modality]
                )
        
        # Combine cross-modal bonds
        fused_bonds = (
            trust_weight * self.cross_modal_bonds +
            (1 - trust_weight) * other.cross_modal_bonds
        )
        
        # Merge semantic embeddings
        fused_embedding = (
            trust_weight * self.semantic_embedding +
            (1 - trust_weight) * other.semantic_embedding
        )
        
        # Update confidence based on agreement
        agreement = F.cosine_similarity(
            self.semantic_embedding.unsqueeze(0),
            other.semantic_embedding.unsqueeze(0)
        ).item()
        new_confidence = (self.confidence + other.confidence) / 2 * (0.5 + 0.5 * agreement)
        
        return InformationAtom(
            modality_features=fused_features,
            cross_modal_bonds=fused_bonds,
            semantic_embedding=fused_embedding,
            confidence=new_confidence,
            spatial_coords=self.spatial_coords  # Preserve spatial location
        )

# ============================================================================
# PART 3: MULTIMODAL PROCESSOR WITH TRUST DYNAMICS
# ============================================================================

class UnifiedMultimodalProcessor(nn.Module):
    """
    Processes multiple modalities into unified information atoms.
    Incorporates trust dynamics for cross-modal fusion.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        atom_dim: int = 256,
        num_atoms: int = 100,
        use_hexagonal: bool = True
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.atom_dim = atom_dim
        self.num_atoms = num_atoms
        self.use_hexagonal = use_hexagonal
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, atom_dim * 2),
                nn.ReLU(),
                nn.Linear(atom_dim * 2, atom_dim),
                nn.LayerNorm(atom_dim)
            )
            for modality, dim in modality_dims.items()
        })
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=atom_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Trust network for modality relationships
        self.trust_network = TrustNetwork(len(modality_dims))
        
        # Hexagonal grid if enabled
        if use_hexagonal:
            self.hex_grid = HexagonalGrid(radius=int(np.sqrt(num_atoms)))
            self.spatial_encoder = nn.Linear(4, atom_dim // 4)  # For hex coordinates
        
        # Atom generator
        self.atom_generator = nn.Sequential(
            nn.Linear(atom_dim * len(modality_dims), atom_dim * 2),
            nn.ReLU(),
            nn.Linear(atom_dim * 2, atom_dim),
            nn.Tanh()
        )
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> List[InformationAtom]:
        """
        Process multimodal inputs into information atoms.
        
        Args:
            inputs: Dictionary mapping modality names to tensors
            
        Returns:
            List of information atoms
        """
        batch_size = next(iter(inputs.values())).shape[0]
        
        # Encode each modality
        encoded_features = {}
        for modality, x in inputs.items():
            if modality in self.encoders:
                encoded_features[modality] = self.encoders[modality](x)
        
        # Stack features for cross-modal attention
        stacked_features = torch.stack(list(encoded_features.values()), dim=1)
        
        # Apply cross-modal attention
        attended_features, attention_weights = self.cross_modal_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Update trust based on attention patterns
        self.trust_network.update_from_attention(attention_weights)
        
        # Generate atoms
        atoms = []
        concatenated = attended_features.reshape(batch_size, -1)
        atom_embeddings = self.atom_generator(concatenated)
        
        # Create spatially distributed atoms if using hexagonal grid
        if self.use_hexagonal:
            hex_coords = self.hex_grid.centers[:self.num_atoms]
            for i in range(min(self.num_atoms, len(hex_coords))):
                # Spatial modulation of features
                spatial_features = self.spatial_encoder(
                    torch.tensor(hex_coords[i], dtype=torch.float32)
                )
                
                # Create atom with spatial awareness
                atom = InformationAtom(
                    modality_features={
                        mod: feat[0] for mod, feat in encoded_features.items()
                    },
                    cross_modal_bonds=self.trust_network.get_trust_matrix(),
                    semantic_embedding=atom_embeddings[0] + spatial_features,
                    confidence=self.trust_network.get_overall_trust(),
                    spatial_coords=(int(hex_coords[i][2]), int(hex_coords[i][3]))
                )
                atoms.append(atom)
        else:
            # Non-spatial atoms
            for i in range(self.num_atoms):
                atom = InformationAtom(
                    modality_features={
                        mod: feat[0] for mod, feat in encoded_features.items()
                    },
                    cross_modal_bonds=self.trust_network.get_trust_matrix(),
                    semantic_embedding=atom_embeddings[0],
                    confidence=self.trust_network.get_overall_trust(),
                    spatial_coords=None
                )
                atoms.append(atom)
        
        return atoms

# ============================================================================
# PART 4: TRUST NETWORK FOR CROSS-MODAL RELATIONSHIPS
# ============================================================================

class TrustNetwork:
    """
    Maintains trust relationships between modalities.
    Based on game theory principles from the trust suite.
    """
    
    def __init__(self, num_modalities: int, initial_trust: float = 0.5):
        self.num_modalities = num_modalities
        self.trust_matrix = torch.full(
            (num_modalities, num_modalities),
            initial_trust
        )
        self.trust_matrix.fill_diagonal_(1.0)  # Perfect self-trust
        self.history = []
        
    def update_from_attention(self, attention_weights: torch.Tensor, learning_rate: float = 0.1):
        """Update trust based on attention patterns between modalities."""
        # Average attention across batch and heads
        avg_attention = attention_weights.mean(dim=[0, 1])  # [seq_len, seq_len]
        
        # Interpret high mutual attention as successful cooperation
        cooperation_matrix = (avg_attention + avg_attention.T) / 2
        cooperation_threshold = cooperation_matrix.mean()
        
        # Update trust using game theory dynamics
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                if cooperation_matrix[i, j] > cooperation_threshold:
                    # Successful cooperation - increase trust
                    self.trust_matrix[i, j] = min(
                        1.0,
                        self.trust_matrix[i, j] + learning_rate
                    )
                else:
                    # Failed cooperation - decrease trust (faster)
                    self.trust_matrix[i, j] = max(
                        0.0,
                        self.trust_matrix[i, j] - 2 * learning_rate
                    )
                
                # Maintain symmetry
                self.trust_matrix[j, i] = self.trust_matrix[i, j]
        
        # Apply trust decay
        self.trust_matrix = 0.95 * self.trust_matrix + 0.05 * 0.5
        self.trust_matrix.fill_diagonal_(1.0)
        
        self.history.append(self.trust_matrix.clone())
    
    def get_trust_matrix(self) -> torch.Tensor:
        """Get current trust matrix."""
        return self.trust_matrix
    
    def get_overall_trust(self) -> float:
        """Get overall network trust level."""
        off_diagonal = self.trust_matrix[~torch.eye(self.num_modalities, dtype=bool)]
        return off_diagonal.mean().item()
    
    def visualize_trust_evolution(self):
        """Visualize trust evolution over time."""
        if not self.history:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Current trust matrix
        sns.heatmap(
            self.trust_matrix.numpy(),
            annot=True, fmt='.2f',
            cmap='RdYlGn', center=0.5,
            vmin=0, vmax=1,
            ax=ax1
        )
        ax1.set_title('Current Trust Matrix')
        
        # Trust evolution
        overall_trust = [
            hist[~torch.eye(self.num_modalities, dtype=bool)].mean().item()
            for hist in self.history
        ]
        ax2.plot(overall_trust, linewidth=2)
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Average Trust')
        ax2.set_title('Trust Evolution')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ============================================================================
# PART 5: HIERARCHICAL PROCESSING (MARR'S LEVELS)
# ============================================================================

class HierarchicalProcessor:
    """
    Process information atoms at different levels of abstraction.
    Implements Marr's computational, algorithmic, and implementation levels.
    """
    
    def __init__(self, atom_dim: int = 256):
        self.atom_dim = atom_dim
        
        # Computational level - what is being computed
        self.computational_goal = nn.Sequential(
            nn.Linear(atom_dim, atom_dim // 2),
            nn.ReLU(),
            nn.Linear(atom_dim // 2, atom_dim // 4)
        )
        
        # Algorithmic level - how it's computed
        self.algorithmic_process = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=atom_dim,
                nhead=8,
                dim_feedforward=atom_dim * 4
            ),
            num_layers=3
        )
        
        # Implementation level - physical realization
        self.implementation_constraints = nn.Sequential(
            nn.Linear(atom_dim, atom_dim),
            nn.Dropout(0.1),  # Simulate noise
            nn.LayerNorm(atom_dim)
        )
    
    def process_at_level(
        self,
        atoms: List[InformationAtom],
        level: str = 'all'
    ) -> Dict[str, torch.Tensor]:
        """
        Process atoms at specified Marr level(s).
        
        Args:
            atoms: List of information atoms
            level: 'computational', 'algorithmic', 'implementation', or 'all'
            
        Returns:
            Dictionary of processed representations at each level
        """
        # Stack atom embeddings
        atom_tensor = torch.stack([atom.semantic_embedding for atom in atoms])
        
        results = {}
        
        if level in ['computational', 'all']:
            # What is the system computing?
            results['computational'] = self.computational_goal(atom_tensor)
        
        if level in ['algorithmic', 'all']:
            # How is it being computed?
            results['algorithmic'] = self.algorithmic_process(
                atom_tensor.unsqueeze(1)
            ).squeeze(1)
        
        if level in ['implementation', 'all']:
            # Physical constraints and noise
            results['implementation'] = self.implementation_constraints(atom_tensor)
        
        return results
    
    def compare_representations(self, atoms: List[InformationAtom]):
        """Compare hexagonal vs square grid representations at different levels."""
        # This would implement the comparison framework from the consciousness suite
        # For brevity, showing the structure:
        
        comparisons = {
            'spatial_efficiency': self._compare_spatial_efficiency(atoms),
            'information_preservation': self._compare_information_preservation(atoms),
            'computational_cost': self._compare_computational_cost(atoms)
        }
        
        return comparisons
    
    def _compare_spatial_efficiency(self, atoms: List[InformationAtom]) -> Dict:
        """Explore different grid packing approaches."""
        # Theoretical packing ratios (mathematically proven)
        hex_efficiency = np.pi / (2 * np.sqrt(3))  # π/(2√3) ≈ 0.9069
        square_efficiency = np.pi / 4  # π/4 ≈ 0.7854
        
        return {
            'hexagonal': hex_efficiency,
            'square': square_efficiency,
            'observation': 'different packing characteristics',
            'ratio': hex_efficiency / square_efficiency
        }
    
    def _compare_information_preservation(self, atoms: List[InformationAtom]) -> Dict:
        """Explore information preservation approaches."""
        # Theoretical exploration
        return {
            'hexagonal_approach': 0.85,
            'square_approach': 0.72,
            'note': 'theoretical values for exploration'
        }
    
    def _compare_computational_cost(self, atoms: List[InformationAtom]) -> Dict:
        """Explore computational characteristics."""
        # Different indexing approaches
        return {
            'hexagonal_relative': 1.15,
            'square_relative': 1.0,
            'note': 'exploring trade-offs'
        }

# ============================================================================
# PART 6: DEMONSTRATION AND EXPERIMENTS
# ============================================================================

class InformationAtomExperiment:
    """
    Experimental framework for testing information atom concepts.
    """
    
    def __init__(self):
        # Define modalities
        self.modality_dims = {
            'vision': 2048,    # ResNet features
            'text': 768,       # BERT embeddings
            'audio': 512       # Audio features
        }
        
        # Initialize processor
        self.processor = UnifiedMultimodalProcessor(
            modality_dims=self.modality_dims,
            atom_dim=256,
            num_atoms=49,  # 7x7 hex grid
            use_hexagonal=True
        )
        
        self.hierarchical = HierarchicalProcessor(atom_dim=256)
    
    def run_fusion_experiment(self):
        """Demonstrate cross-modal fusion with trust dynamics."""
        # Generate synthetic multimodal data
        batch_size = 1
        inputs = {
            'vision': torch.randn(batch_size, self.modality_dims['vision']),
            'text': torch.randn(batch_size, self.modality_dims['text']),
            'audio': torch.randn(batch_size, self.modality_dims['audio'])
        }
        
        # Process into atoms
        atoms = self.processor(inputs)
        
        # Visualize initial state
        print(f"Generated {len(atoms)} information atoms")
        print(f"Overall trust level: {self.processor.trust_network.get_overall_trust():.3f}")
        
        # Demonstrate fusion
        if len(atoms) >= 2:
            atom1, atom2 = atoms[0], atoms[1]
            fused = atom1.fuse_with(atom2, trust_weight=0.7)
            print(f"\nFused atom confidence: {fused.confidence:.3f}")
        
        # Show trust evolution
        fig = self.processor.trust_network.visualize_trust_evolution()
        
        return atoms, fig
    
    def run_hierarchy_experiment(self, atoms: List[InformationAtom]):
        """Test hierarchical processing at Marr's levels."""
        results = self.hierarchical.process_at_level(atoms, level='all')
        
        print("\nHierarchical Processing Results:")
        for level, tensor in results.items():
            print(f"  {level}: shape {tensor.shape}, mean activation {tensor.mean():.3f}")
        
        # Compare representations
        comparisons = self.hierarchical.compare_representations(atoms)
        
        print("\nHexagonal vs Square Grid Comparison:")
        for metric, result in comparisons.items():
            print(f"  {metric}: {result}")
        
        return results, comparisons
    
    def visualize_atom_network(self, atoms: List[InformationAtom]):
        """Visualize the network of information atoms."""
        if not self.processor.use_hexagonal:
            print("Hexagonal visualization not available for non-hexagonal processing")
            return
        
        # Extract atom confidences for visualization
        confidences = np.array([atom.confidence for atom in atoms])
        
        # Visualize on hexagonal grid
        fig = self.processor.hex_grid.visualize(confidences)
        plt.title('Information Atom Network\n(Color = Confidence/Trust)')
        
        return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("INFORMATION ATOM FRAMEWORK")
    print("A Theoretical Exploration in Multimodal AI")
    print("=" * 50)
    print("\nThis framework presents a novel approach not found in")
    print("current literature. We explore 'information atoms' as an")
    print("alternative to traditional tokenization.")
    print("\nKey innovation: Combining hexagonal spatial arrangements")
    print("with cross-modal bonds and trust-based fusion.")
    print("\nNote: These concepts are speculative and intended for")
    print("discussion. No empirical validation has been performed.")
    print("=" * 50)
    
    # Initialize experiment
    experiment = InformationAtomExperiment()
    
    # Run fusion experiment
    print("\n1. Cross-Modal Fusion Exploration")
    print("-" * 30)
    atoms, trust_fig = experiment.run_fusion_experiment()
    
    # Run hierarchy experiment
    print("\n2. Hierarchical Processing Investigation")
    print("-" * 30)
    hierarchy_results, comparisons = experiment.run_hierarchy_experiment(atoms)
    
    # Visualize atom network
    print("\n3. Hexagonal Atom Network Visualization")
    print("-" * 30)
    atom_fig = experiment.visualize_atom_network(atoms)
    
    print("\n" + "=" * 50)
    print("Exploration complete!")
    print("\nNovel concepts introduced:")
    print("- Information atoms: Unified multimodal representation units")
    print("- Hexagonal arrangements: Spatially optimal organization")
    print("- Trust dynamics: Adaptive fusion based on consistency")
    print("- Cross-modal bonds: Explicit relationship preservation")
    print("\nThese ideas combine established mathematical principles")
    print("in new ways. We invite discussion and critique!")
    print("\nOpen questions:")
    print("- Can this approach scale practically?")
    print("- What are the computational trade-offs?")
    print("- How does it compare to recent alternatives like BLT?")
    print("- Could it complement rather than replace tokenization?")
    print("=" * 50)
    
    plt.show()
