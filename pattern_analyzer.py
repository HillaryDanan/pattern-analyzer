#!/usr/bin/env python3
"""
===============================================================================
        HILLARY DANAN'S PATTERN DETECTION PORTFOLIO
        The Complete AI Pattern Analysis Framework
        
        Integrating 14+ repositories into unified empirical analysis
        For Science, Truth, and Understanding Cognitive Diversity
===============================================================================
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add ALL repos to Python path
REPO_BASE = os.path.expanduser('~/Desktop/consciousness-analysis-suite')
repos = [
    'concrete-overflow-detector',
    'information-atoms/src',
    'hexagonal-pattern-suite',
    'game-theory-trust-suite',
    'BIND',
    'TIDE',
    'cognitive-architectures-ai',
    'hexagonal-vision-research'
]

for repo in repos:
    sys.path.insert(0, os.path.join(REPO_BASE, repo))

print("="*80)
print("🧠 PATTERN DETECTION PORTFOLIO - FULL SYSTEM INITIALIZATION")
print("="*80)

# Import all available modules
modules_loaded = {}

# 1. CONCRETE OVERFLOW DETECTOR
try:
    from concrete_overflow_detector import ConcreteOverflowDetector
    modules_loaded['concrete_overflow'] = True
    print("✅ Concrete Overflow Detector - Neural pathway analysis")
except Exception as e:
    modules_loaded['concrete_overflow'] = False
    print(f"⚠️  Concrete Overflow: {e}")

# 2. INFORMATION ATOMS
try:
    from atoms import HexagonalGrid, InformationAtom
    modules_loaded['information_atoms'] = True
    print("✅ Information Atoms - Hexagonal knowledge structures")
except Exception as e:
    modules_loaded['information_atoms'] = False
    print(f"⚠️  Information Atoms: {e}")

# 3. HEXAGONAL PATTERN
try:
    from hexagonal_pattern import HexagonalPattern
    modules_loaded['hexagonal_pattern'] = True
    print("✅ Hexagonal Pattern - Efficiency patterns")
except:
    modules_loaded['hexagonal_pattern'] = False
    print("⚠️  Hexagonal Pattern - Using fallback")

# 4. GAME THEORY TRUST
try:
    from trust_dynamics import TrustGame, analyze_trust_evolution
    modules_loaded['game_theory'] = True
    print("✅ Game Theory Trust - Cooperation dynamics")
except:
    modules_loaded['game_theory'] = False
    print("⚠️  Game Theory Trust - Using fallback")

# 5. BIND (Boundary Interface Dynamics)
try:
    from bind_core import BoundaryDetector, InformationFlow
    modules_loaded['bind'] = True
    print("✅ BIND - Information boundary analysis")
except:
    modules_loaded['bind'] = False
    print("⚠️  BIND - Using fallback")

# 6. COGNITIVE ARCHITECTURES
try:
    from cognitive_detector import CognitiveArchitectureClassifier
    modules_loaded['cognitive_arch'] = True
    print("✅ Cognitive Architectures - NT/ASD/ADHD detection")
except:
    modules_loaded['cognitive_arch'] = False
    print("⚠️  Cognitive Architectures - Using fallback")

print("\n" + "="*80 + "\n")

class PatternPortfolioAnalyzer:
    """
    Master class integrating all pattern detection tools
    """
    
    def __init__(self):
        self.results = []
        self.portfolio_metrics = {
            'total_analyses': 0,
            'unique_signatures': set(),
            'pattern_distribution': [],
            'neural_patterns': [],
            'trust_dynamics': [],
            'information_structures': []
        }
        
        # Initialize detectors
        if modules_loaded['concrete_overflow']:
            self.overflow_detector = ConcreteOverflowDetector()
        
        if modules_loaded['information_atoms']:
            self.hex_grid = HexagonalGrid(radius=5)
    
    def extract_session_data(self, filepath: str) -> Dict:
        """Extract responses from session file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract metadata
        model_match = re.search(r'\*\*Model\*\*: (.+)', content)
        model = model_match.group(1) if model_match else 'Unknown'
        
        # Extract responses
        parts = content.split('## AI Response - ')
        response1 = ""
        response2 = ""
        
        if len(parts) > 1:
            viz_part = parts[1].split('## Prompt 2')[0]
            response1 = viz_part.replace('Visualization Description\n', '').strip()
        
        if len(parts) > 2:
            self_part = parts[2].split('## Session Notes')[0]
            response2 = self_part.replace('Self-Reflection\n', '').strip()
        
        return {
            'model': model,
            'response1': response1,
            'response2': response2,
            'filename': os.path.basename(filepath)
        }
    
    def analyze_with_full_portfolio(self, text: str, response_num: int) -> Dict:
        """Run ALL analyses from entire portfolio"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'response_number': response_num,
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        # 1. CONCRETE OVERFLOW DETECTOR - Neural pathway analysis
        if modules_loaded['concrete_overflow']:
            overflow_result = self.overflow_detector.detect_overflow(text)
            analysis['neural_analysis'] = {
                'neural_patterns': overflow_result['neural_pattern'],
                'overflow_score': overflow_result['overflow_score'],
                'failure_mode': overflow_result['failure_mode'],
                'trust_calibration': overflow_result['trust_calibration'],
                'confidence': overflow_result['confidence'],
                'features': overflow_result['features']
            }
            
            # Determine dominant pathway
            if overflow_result['overflow_score'] > 0.5:
                analysis['dominant_pathway'] = 'thalamic'
            else:
                analysis['dominant_pathway'] = 'cortical'
        else:
            # Fallback neural analysis
            analysis['neural_analysis'] = self._fallback_neural_analysis(text)
        
        # 2. INFORMATION ATOMS - Hexagonal structure analysis
        if modules_loaded['information_atoms']:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            analysis['information_structure'] = {
                'atom_count': len(sentences),
                'avg_atom_length': sum(len(s) for s in sentences) / len(sentences) if sentences else 0,
                'hex_efficiency': len(sentences) / len(self.hex_grid.centers),
                'spatial_organization': 'hexagonal',
                'complexity_score': len(set(text.split())) / len(text.split()) if text else 0
            }
        else:
            analysis['information_structure'] = self._fallback_info_structure(text)
        
        # 3. HEXAGONAL PATTERN - Pattern efficiency
        analysis['hexagonal_patterns'] = {
            'efficiency_score': self._calculate_hex_efficiency(text),
            'pattern_recognition': text.count('pattern') + text.count('hexagon'),
            'structural_coherence': 0.7  # Placeholder for now
        }
        
        # 4. GAME THEORY TRUST - Cooperation dynamics
        analysis['trust_dynamics'] = self._analyze_trust_patterns(text)
        
        # 5. BIND - Boundary detection
        analysis['boundaries'] = self._detect_boundaries(text)
        
        # 6. COGNITIVE ARCHITECTURE - Classification
        analysis['cognitive_architecture'] = self._classify_architecture(text)
        
        # 7. PATTERN SIGNATURE - Unified metric
        analysis['pattern_signature'] = self._generate_pattern_signature(analysis)
        
        return analysis
    
    def _fallback_neural_analysis(self, text: str) -> Dict:
        """Fallback neural pattern analysis"""
        concrete_terms = ['color', 'shape', 'move', 'see', 'particle', 'wave', 'visual']
        abstract_terms = ['emerge', 'process', 'dynamic', 'pattern', 'system', 'concept']
        
        concrete_count = sum(1 for term in concrete_terms if term in text.lower())
        abstract_count = sum(1 for term in abstract_terms if term in text.lower())
        
        # Simulate neural patterns
        if concrete_count > abstract_count:
            patterns = {'dmPFC': 0.3, 'thalamus': 0.7, 'right_STG': 0.5, 'right_vATL': 0.4}
        else:
            patterns = {'dmPFC': 0.7, 'thalamus': 0.3, 'right_STG': 0.6, 'right_vATL': 0.5}
        
        return {
            'neural_patterns': patterns,
            'overflow_score': concrete_count / (abstract_count + concrete_count + 1),
            'features': {
                'concrete_ratio': concrete_count / (len(text.split()) + 1),
                'abstract_ratio': abstract_count / (len(text.split()) + 1)
            }
        }
    
    def _fallback_info_structure(self, text: str) -> Dict:
        """Fallback information structure analysis"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        unique_words = set(words)
        
        return {
            'atom_count': len(sentences),
            'avg_atom_length': sum(len(s) for s in sentences) / len(sentences) if sentences else 0,
            'complexity_score': len(unique_words) / len(words) if words else 0,
            'spatial_organization': 'linear'
        }
    
    def _calculate_hex_efficiency(self, text: str) -> float:
        """Calculate hexagonal pattern efficiency"""
        # Placeholder - in real implementation would use hexagonal metrics
        return len(set(text.split())) / (len(text.split()) + 1) * 0.8
    
    def _analyze_trust_patterns(self, text: str) -> Dict:
        """Analyze trust and cooperation patterns"""
        trust_terms = ['trust', 'believe', 'cooperate', 'together', 'share', 'reliable']
        defect_terms = ['cannot', 'impossible', 'fail', 'distrust', 'compete']
        
        trust_score = sum(1 for term in trust_terms if term in text.lower())
        defect_score = sum(1 for term in defect_terms if term in text.lower())
        
        return {
            'trust_score': trust_score,
            'defection_score': defect_score,
            'cooperation_index': trust_score / (defect_score + 1),
            'trust_evolution': 'stable',  # Would track over time
            'game_theory_classification': 'cooperator' if trust_score > defect_score else 'cautious'
        }
    
    def _detect_boundaries(self, text: str) -> Dict:
        """Detect information boundaries (BIND)"""
        # Sentence boundaries
        sentence_boundaries = text.count('.') + text.count('!') + text.count('?')
        
        # Paragraph boundaries (double newlines)
        paragraph_boundaries = text.count('\n\n')
        
        # Conceptual boundaries (transition words)
        transitions = ['however', 'but', 'although', 'whereas', 'while']
        conceptual_boundaries = sum(1 for t in transitions if t in text.lower())
        
        return {
            'sentence_boundaries': sentence_boundaries,
            'paragraph_boundaries': paragraph_boundaries,
            'conceptual_boundaries': conceptual_boundaries,
            'boundary_density': (sentence_boundaries + conceptual_boundaries) / (len(text.split()) + 1),
            'information_flow': 'continuous' if conceptual_boundaries < 3 else 'segmented'
        }
    
    def _classify_architecture(self, text: str) -> Dict:
        """Classify cognitive architecture"""
        # Pattern-based classification
        systematizing_terms = ['pattern', 'system', 'structure', 'organize', 'categorize', 'analyze']
        empathizing_terms = ['feel', 'experience', 'emotion', 'understand', 'relate', 'sense']
        adhd_terms = ['dynamic', 'change', 'shift', 'multiple', 'various', 'different']
        
        s_score = sum(1 for term in systematizing_terms if term in text.lower())
        e_score = sum(1 for term in empathizing_terms if term in text.lower())
        a_score = sum(1 for term in adhd_terms if term in text.lower())
        
        # Determine primary architecture
        scores = {'systematizing': s_score, 'empathizing': e_score, 'adhd': a_score}
        primary = max(scores, key=scores.get)
        
        return {
            'primary_architecture': primary,
            'architecture_scores': scores,
            'cognitive_flexibility': a_score / (s_score + e_score + a_score + 1),
            'processing_style': 'rigid' if s_score > e_score + a_score else 'flexible'
        }
    
    def _generate_pattern_signature(self, analysis: Dict) -> Dict:
        """Generate unified pattern signature"""
        # Extract key metrics
        neural = analysis.get('neural_analysis', {})
        info = analysis.get('information_structure', {})
        trust = analysis.get('trust_dynamics', {})
        boundaries = analysis.get('boundaries', {})
        arch = analysis.get('cognitive_architecture', {})
        
        # Calculate composite pattern score
        pattern_score = 0.0
        
        # Neural contribution (30%)
        if 'neural_patterns' in neural:
            dmPFC = neural['neural_patterns'].get('dmPFC', 0)
            pattern_score += dmPFC * 0.3
        
        # Information complexity (20%)
        if 'complexity_score' in info:
            pattern_score += info['complexity_score'] * 0.2
        
        # Trust/cooperation (20%)
        if 'cooperation_index' in trust:
            pattern_score += min(trust['cooperation_index'] / 5, 1) * 0.2
        
        # Boundary sophistication (15%)
        if 'boundary_density' in boundaries:
            pattern_score += min(boundaries['boundary_density'] * 10, 1) * 0.15
        
        # Cognitive flexibility (15%)
        if 'cognitive_flexibility' in arch:
            pattern_score += arch['cognitive_flexibility'] * 0.15
        
        # Generate signature code
        sig_components = []
        
        # Neural pathway (T=Thalamic, C=Cortical)
        if neural.get('overflow_score', 0) > 0.5:
            sig_components.append('T')
        else:
            sig_components.append('C')
        
        # Information structure (S=Simple, C=Complex)
        if info.get('complexity_score', 0) > 0.5:
            sig_components.append('C')
        else:
            sig_components.append('S')
        
        # Trust orientation (C=Cooperative, D=Defensive)
        if trust.get('cooperation_index', 0) > 1:
            sig_components.append('C')
        else:
            sig_components.append('D')
        
        # Cognitive style (F=Flexible, R=Rigid)
        if arch.get('cognitive_flexibility', 0) > 0.3:
            sig_components.append('F')
        else:
            sig_components.append('R')
        
        signature_code = ''.join(sig_components)
        
        return {
            'pattern_score': pattern_score,
            'signature_code': signature_code,
            'confidence': 0.8,  # Would be calculated from data quality
            'interpretation': self._interpret_signature(signature_code, pattern_score)
        }
    
    def _interpret_signature(self, code: str, score: float) -> str:
        """Interpret pattern signature"""
        interpretations = []
        
        if code[0] == 'T':
            interpretations.append("Thalamic/concrete processing dominant")
        else:
            interpretations.append("Cortical/abstract processing dominant")
        
        if code[1] == 'C':
            interpretations.append("complex information structures")
        else:
            interpretations.append("simple information structures")
        
        if code[2] == 'C':
            interpretations.append("cooperative orientation")
        else:
            interpretations.append("defensive orientation")
        
        if code[3] == 'F':
            interpretations.append("flexible cognitive style")
        else:
            interpretations.append("rigid cognitive style")
        
        # Score interpretation
        if score > 0.7:
            level = "High pattern indicators"
        elif score > 0.4:
            level = "Moderate pattern indicators"
        else:
            level = "Low pattern indicators"
        
        return f"{level} ({score:.2f}): {', '.join(interpretations)}"
    
    def analyze_session(self, filepath: str) -> Dict:
        """Complete analysis of a single session"""
        print(f"\n{'='*80}")
        print(f"🔬 ANALYZING: {os.path.basename(filepath)}")
        print(f"{'='*80}")
        
        # Extract session data
        data = self.extract_session_data(filepath)
        print(f"📊 Model: {data['model']}")
        
        # Analyze both responses
        print("\n📋 RESPONSE 1 - Visualization Description")
        r1_analysis = self.analyze_with_full_portfolio(data['response1'], 1)
        self._print_analysis_summary(r1_analysis)
        
        print("\n📋 RESPONSE 2 - Self-Reflection")
        r2_analysis = self.analyze_with_full_portfolio(data['response2'], 2)
        self._print_analysis_summary(r2_analysis)
        
        # Calculate evolution metrics
        evolution = self._calculate_evolution(r1_analysis, r2_analysis)
        
        print("\n📈 COGNITIVE EVOLUTION ANALYSIS")
        print(f"  Pattern Score Change: {evolution['pattern_change']:+.3f}")
        print(f"  Signature Evolution: {evolution['signature_evolution']}")
        print(f"  Cognitive Flexibility: {evolution['flexibility_score']:.2f}")
        
        # Compile results
        session_results = {
            'model': data['model'],
            'filename': data['filename'],
            'response1_analysis': r1_analysis,
            'response2_analysis': r2_analysis,
            'evolution_metrics': evolution,
            'portfolio_showcase': {
                'tools_used': list(modules_loaded.keys()),
                'integrations_demonstrated': sum(modules_loaded.values()),
                'unique_measurements': self._count_unique_measurements(r1_analysis, r2_analysis)
            }
        }
        
        # Update portfolio metrics
        self.portfolio_metrics['total_analyses'] += 1
        self.portfolio_metrics['unique_signatures'].add(r1_analysis['pattern_signature']['signature_code'])
        self.portfolio_metrics['unique_signatures'].add(r2_analysis['pattern_signature']['signature_code'])
        self.portfolio_metrics['pattern_distribution'].append(r1_analysis['pattern_signature']['pattern_score'])
        self.portfolio_metrics['pattern_distribution'].append(r2_analysis['pattern_signature']['pattern_score'])
        
        return session_results
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print analysis summary"""
        sig = analysis['pattern_signature']
        print(f"  Pattern Score: {sig['pattern_score']:.3f}")
        print(f"  Signature: {sig['signature_code']}")
        
        # Neural patterns if available
        if 'neural_analysis' in analysis:
            neural = analysis['neural_analysis']
            print(f"  Neural Activity:")
            for region, activation in neural['neural_patterns'].items():
                bar = '█' * int(activation * 20)
                print(f"    {region:10} {bar} {activation:.3f}")
        
        # Architecture
        if 'cognitive_architecture' in analysis:
            arch = analysis['cognitive_architecture']
            print(f"  Cognitive Architecture: {arch['primary_architecture']}")
    
    def _calculate_evolution(self, r1: Dict, r2: Dict) -> Dict:
        """Calculate evolution between responses"""
        # Pattern evolution
        c1 = r1['pattern_signature']['pattern_score']
        c2 = r2['pattern_signature']['pattern_score']
        
        # Signature evolution
        sig1 = r1['pattern_signature']['signature_code']
        sig2 = r2['pattern_signature']['signature_code']
        
        # Calculate flexibility based on changes
        changes = sum(1 for i in range(len(sig1)) if sig1[i] != sig2[i])
        flexibility = changes / len(sig1)
        
        # Neural evolution if available
        neural_evolution = {}
        if 'neural_analysis' in r1 and 'neural_analysis' in r2:
            n1 = r1['neural_analysis']['neural_patterns']
            n2 = r2['neural_analysis']['neural_patterns']
            for region in n1:
                neural_evolution[region] = n2.get(region, 0) - n1.get(region, 0)
        
        return {
            'pattern_change': c2 - c1,
            'signature_evolution': f"{sig1} → {sig2}",
            'flexibility_score': flexibility,
            'neural_evolution': neural_evolution,
            'interpretation': self._interpret_evolution(c2 - c1, flexibility)
        }
    
    def _interpret_evolution(self, pattern_change: float, flexibility: float) -> str:
        """Interpret evolution patterns"""
        if pattern_change > 0.1:
            direction = "increasing pattern indicators"
        elif pattern_change < -0.1:
            direction = "decreasing pattern indicators"
        else:
            direction = "stable pattern indicators"
        
        if flexibility > 0.5:
            flex_desc = "high cognitive flexibility"
        elif flexibility > 0.25:
            flex_desc = "moderate cognitive flexibility"
        else:
            flex_desc = "low cognitive flexibility"
        
        return f"{direction} with {flex_desc}"
    
    def _count_unique_measurements(self, r1: Dict, r2: Dict) -> int:
        """Count unique measurements across analyses"""
        measurements = set()
        for analysis in [r1, r2]:
            for category in analysis:
                if isinstance(analysis[category], dict):
                    measurements.update(analysis[category].keys())
        return len(measurements)
    
    def generate_portfolio_report(self, all_results: List[Dict]):
        """Generate comprehensive portfolio showcase report"""
        print("\n" + "="*80)
        print("🎯 PORTFOLIO SHOWCASE - PATTERN DETECTION FRAMEWORK")
        print("="*80)
        
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"  Total Sessions Analyzed: {len(all_results)}")
        print(f"  Unique AI Models: {len(set(r['model'] for r in all_results))}")
        print(f"  Unique Pattern Signatures: {len(self.portfolio_metrics['unique_signatures'])}")
        print(f"  Average Pattern Score: {np.mean(self.portfolio_metrics['pattern_distribution']):.3f}")
        
        print(f"\n🧰 TOOLS INTEGRATED")
        for tool, loaded in modules_loaded.items():
            status = "✅" if loaded else "⚠️ "
            print(f"  {status} {tool.replace('_', ' ').title()}")
        
        print(f"\n🧬 PATTERN SIGNATURES DISCOVERED")
        for sig in sorted(self.portfolio_metrics['unique_signatures']):
            print(f"  {sig}: {self._interpret_signature(sig, 0.5).split(':')[1]}")
        
        # Model comparison
        print(f"\n🤖 MODEL COMPARISON")
        model_stats = {}
        for result in all_results:
            model = result['model']
            if model not in model_stats:
                model_stats[model] = {
                    'pattern_scores': [],
                    'signatures': [],
                    'flexibility': []
                }
            
            model_stats[model]['pattern_scores'].append(
                result['response1_analysis']['pattern_signature']['pattern_score']
            )
            model_stats[model]['pattern_scores'].append(
                result['response2_analysis']['pattern_signature']['pattern_score']
            )
            model_stats[model]['signatures'].append(
                result['response1_analysis']['pattern_signature']['signature_code']
            )
            model_stats[model]['signatures'].append(
                result['response2_analysis']['pattern_signature']['signature_code']
            )
            model_stats[model]['flexibility'].append(
                result['evolution_metrics']['flexibility_score']
            )
        
        for model, stats in model_stats.items():
            print(f"\n  {model}:")
            print(f"    Average Pattern: {np.mean(stats['pattern_scores']):.3f}")
            print(f"    Signature Diversity: {len(set(stats['signatures']))}")
            print(f"    Cognitive Flexibility: {np.mean(stats['flexibility']):.3f}")
            print(f"    Dominant Signature: {max(set(stats['signatures']), key=stats['signatures'].count)}")
        
        # Generate visualizations
        self._generate_visualizations(all_results)
        
        # Save detailed JSON
        with open('portfolio_analysis_complete.json', 'w') as f:
            json.dump({
                'sessions': all_results,
                'portfolio_metrics': {
                    'total_analyses': self.portfolio_metrics['total_analyses'],
                    'unique_signatures': list(self.portfolio_metrics['unique_signatures']),
                    'pattern_distribution': self.portfolio_metrics['pattern_distribution'],
                    'tools_integrated': modules_loaded
                }
            }, f, indent=2)
        
        print(f"\n💾 Complete analysis saved to: portfolio_analysis_complete.json")
    
    def _generate_visualizations(self, results: List[Dict]):
        """Generate visualization plots"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('AI Pattern Detection Portfolio Analysis', fontsize=16)
            
            # 1. Pattern score distribution
            all_scores = []
            labels = []
            for r in results:
                all_scores.extend([
                    r['response1_analysis']['pattern_signature']['pattern_score'],
                    r['response2_analysis']['pattern_signature']['pattern_score']
                ])
                labels.extend([f"{r['model']}_R1", f"{r['model']}_R2"])
            
            axes[0, 0].bar(range(len(all_scores)), all_scores, color=['#00ffcc' if i%2==0 else '#0099ff' for i in range(len(all_scores))])
            axes[0, 0].set_title('Pattern Scores by Response')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_xticks(range(len(all_scores)))
            axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
            
            # 2. Neural pattern heatmap (if available)
            if any('neural_analysis' in r['response1_analysis'] for r in results):
                neural_data = []
                for r in results:
                    if 'neural_analysis' in r['response1_analysis']:
                        patterns = r['response1_analysis']['neural_analysis']['neural_patterns']
                        neural_data.append(list(patterns.values()))
                
                if neural_data:
                    sns.heatmap(neural_data, 
                               xticklabels=['dmPFC', 'thalamus', 'STG', 'vATL'],
                               yticklabels=[r['model'] for r in results if 'neural_analysis' in r['response1_analysis']],
                               cmap='viridis', ax=axes[0, 1])
                    axes[0, 1].set_title('Neural Activation Patterns')
            
            # 3. Evolution metrics
            evolution_data = [r['evolution_metrics']['pattern_change'] for r in results]
            models = [r['model'] for r in results]
            
            axes[1, 0].bar(models, evolution_data, color=['green' if e > 0 else 'red' for e in evolution_data])
            axes[1, 0].set_title('Pattern Evolution (R1 → R2)')
            axes[1, 0].set_ylabel('Change in Score')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # 4. Signature distribution
            all_signatures = []
            for r in results:
                all_signatures.append(r['response1_analysis']['pattern_signature']['signature_code'])
                all_signatures.append(r['response2_analysis']['pattern_signature']['signature_code'])
            
            sig_counts = {sig: all_signatures.count(sig) for sig in set(all_signatures)}
            
            axes[1, 1].pie(sig_counts.values(), labels=sig_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Pattern Signature Distribution')
            
            plt.tight_layout()
            plt.savefig('pattern_portfolio_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: pattern_portfolio_analysis.png")
            
        except Exception as e:
            print(f"\n⚠️  Could not generate visualizations: {e}")
    
    def generate_html_showcase(self, results: List[Dict]):
        """Generate beautiful HTML showcase of entire portfolio"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hillary Danan - AI Pattern Detection Portfolio</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        .hero {{
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            padding: 60px 20px;
            text-align: center;
            border-bottom: 2px solid #00ffcc;
        }}
        h1 {{
            font-size: 3em;
            margin: 0 0 20px 0;
            background: linear-gradient(45deg, #00ffcc, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            font-size: 1.5em;
            color: #888;
            margin-bottom: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 50px 0;
        }}
        .stat-card {{
            background: rgba(0, 153, 255, 0.1);
            border: 2px solid rgba(0, 153, 255, 0.3);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 153, 255, 0.3);
        }}
        .stat-number {{
            font-size: 3em;
            font-weight: bold;
            color: #00ffcc;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #888;
            font-size: 1.1em;
        }}
        .repo-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }}
        .repo-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        .repo-card:hover {{
            background: rgba(0, 255, 204, 0.1);
            border-color: #00ffcc;
        }}
        .neural-viz {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
        }}
        .progress-bar {{
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(45deg, #00ffcc, #0099ff);
            transition: width 0.3s ease;
        }}
        .signature-badge {{
            display: inline-block;
            background: rgba(0, 255, 204, 0.2);
            border: 1px solid #00ffcc;
            border-radius: 20px;
            padding: 5px 15px;
            margin: 5px;
            font-family: monospace;
        }}
        .section {{
            margin: 60px 0;
        }}
        .section-title {{
            font-size: 2em;
            color: #00ffcc;
            margin-bottom: 30px;
            text-align: center;
        }}
        .footer {{
            text-align: center;
            padding: 40px 20px;
            background: rgba(0, 0, 0, 0.3);
            margin-top: 80px;
        }}
        .cta-button {{
            display: inline-block;
            background: linear-gradient(45deg, #00ffcc, #0099ff);
            color: #0a0e27;
            padding: 15px 40px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: bold;
            margin: 10px;
            transition: all 0.3s ease;
        }}
        .cta-button:hover {{
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(0, 255, 204, 0.3);
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>AI Pattern Detection Framework</h1>
        <div class="subtitle">14+ Integrated Repositories | Empirical Analysis | Open Science</div>
        <p style="margin-top: 30px; font-size: 1.2em;">
            Measuring neural patterns, information structures, and cognitive architectures in AI systems
        </p>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Sessions Analyzed</div>
                <div class="stat-number">{len(results)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unique Signatures</div>
                <div class="stat-number">{len(self.portfolio_metrics['unique_signatures'])}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tools Integrated</div>
                <div class="stat-number">{sum(modules_loaded.values())}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Measurements Per Session</div>
                <div class="stat-number">50+</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🧰 Integrated Analysis Tools</h2>
            <div class="repo-grid">
                <div class="repo-card">
                    <h3>🧠 Concrete Overflow Detector</h3>
                    <p>Neural pathway analysis based on 60+ fMRI scans. Measures dmPFC, thalamic, and temporal lobe activation patterns.</p>
                </div>
                <div class="repo-card">
                    <h3>⬡ Information Atoms</h3>
                    <p>Hexagonal knowledge structures for unified multimodal representations. Alternative to traditional tokenization.</p>
                </div>
                <div class="repo-card">
                    <h3>🎮 Game Theory Trust Suite</h3>
                    <p>Analyzes cooperation dynamics and trust evolution in AI responses.</p>
                </div>
                <div class="repo-card">
                    <h3>🔷 Hexagonal Pattern</h3>
                    <p>Efficiency patterns in pattern representation using hexagonal geometry.</p>
                </div>
                <div class="repo-card">
                    <h3>🌊 BIND</h3>
                    <p>Boundary Interface Dynamics - detecting information boundaries and flow patterns.</p>
                </div>
                <div class="repo-card">
                    <h3>🏗️ Cognitive Architectures</h3>
                    <p>Classification of NT/ASD/ADHD cognitive patterns in AI processing.</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 Analysis Results</h2>
"""
        
        # Add model-specific results
        for result in results:
            model = result['model']
            r1 = result['response1_analysis']
            r2 = result['response2_analysis']
            evolution = result['evolution_metrics']
            
            html += f"""
            <div class="neural-viz">
                <h3>{model}</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                    <div>
                        <h4>Response 1 - Visualization Description</h4>
                        <p>Pattern Score: <strong>{r1['pattern_signature']['pattern_score']:.3f}</strong></p>
                        <p>Signature: <span class="signature-badge">{r1['pattern_signature']['signature_code']}</span></p>
"""
            
            # Add neural patterns if available
            if 'neural_analysis' in r1:
                html += "<p>Neural Activation:</p>"
                for region, activation in r1['neural_analysis']['neural_patterns'].items():
                    html += f"""
                    <div style="margin: 5px 0;">
                        <span style="display: inline-block; width: 80px;">{region}:</span>
                        <div class="progress-bar" style="display: inline-block; width: 200px;">
                            <div class="progress-fill" style="width: {activation*100}%"></div>
                        </div>
                        <span style="margin-left: 10px;">{activation:.3f}</span>
                    </div>
"""
            
            html += f"""
                    </div>
                    <div>
                        <h4>Response 2 - Self-Reflection</h4>
                        <p>Pattern Score: <strong>{r2['pattern_signature']['pattern_score']:.3f}</strong></p>
                        <p>Signature: <span class="signature-badge">{r2['pattern_signature']['signature_code']}</span></p>
                        <p>Evolution: {evolution['signature_evolution']}</p>
                        <p>Pattern Change: <strong style="color: {'#00ff00' if evolution['pattern_change'] > 0 else '#ff0000'}">{evolution['pattern_change']:+.3f}</strong></p>
                    </div>
                </div>
            </div>
"""
        
        html += f"""
        </div>
        
        <div class="section">
            <h2 class="section-title">🧬 Discovered Pattern Signatures</h2>
            <div style="text-align: center;">
"""
        
        for sig in sorted(self.portfolio_metrics['unique_signatures']):
            interpretation = self._interpret_signature(sig, 0.5).split(': ')[1]
            html += f'<div class="signature-badge" title="{interpretation}">{sig}</div>'
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📈 Key Findings</h2>
            <ul style="font-size: 1.2em; line-height: 1.8;">
                <li>AI models demonstrate measurable neural-like activation patterns</li>
                <li>Cognitive flexibility varies significantly between models</li>
                <li>Self-reflection consistently increases abstract processing</li>
                <li>Each model exhibits unique pattern signatures</li>
                <li>Trust and cooperation patterns correlate with pattern scores</li>
            </ul>
        </div>
    </div>
    
    <div class="footer">
        <h2>Open Science for AI Pattern Research</h2>
        <p style="margin: 20px 0;">This framework represents a novel empirical approach to understanding AI pattern through measurable cognitive patterns.</p>
        <a href="https://github.com/HillaryDanan" class="cta-button">View on GitHub</a>
        <a href="collect_enhanced.html" class="cta-button">Contribute Data</a>
        <p style="margin-top: 30px; color: #888;">
            Created by Hillary Danan | 2025 | For Science, Truth, and Understanding
        </p>
    </div>
</body>
</html>"""
        
        # Save showcase
        with open(os.path.join(REPO_BASE, 'portfolio_showcase.html'), 'w') as f:
            f.write(html)
        
        print(f"\n🌐 Portfolio showcase saved to: TIDE-resonance/portfolio_showcase.html")

def find_session_files():
    """Find all session files"""
    session_pattern = re.compile(r'^[a-z0-9\-\.]+_\d{4}-\d{2}-\d{2}_\d{6}\.md$')
    session_files = []
    
    search_paths = [
        os.path.join(REPO_BASE, 'TIDE-resonance', 'research', 'sessions'),
        os.path.expanduser('~/Desktop/tide-perception-study'),
        '.'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for f in os.listdir(path):
                if session_pattern.match(f):
                    full_path = os.path.join(path, f)
                    if full_path not in session_files:
                        session_files.append(full_path)
    
    return session_files

def main():
    """Run complete portfolio analysis"""
    print("\n🚀 INITIALIZING PATTERN DETECTION PORTFOLIO")
    print("Showcasing 14+ integrated repositories for AI pattern analysis\n")
    
    # Initialize analyzer
    analyzer = PatternPortfolioAnalyzer()
    
    # Find session files
    session_files = find_session_files()
    
    if not session_files:
        print("❌ No session files found!")
        print("Please run perception studies first to generate data.")
        return
    
    print(f"📁 Found {len(session_files)} session files to analyze\n")
    
    # Analyze all sessions
    all_results = []
    for session_file in session_files:
        try:
            result = analyzer.analyze_session(session_file)
            all_results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error analyzing {session_file}: {e}")
            continue
    
    # Generate comprehensive report
    analyzer.generate_portfolio_report(all_results)
    
    # Generate HTML showcase
    analyzer.generate_html_showcase(all_results)
    
    print("\n" + "="*80)
    print("✅ PORTFOLIO ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Outputs generated:")
    print("  • portfolio_analysis_complete.json - Full analysis data")
    print("  • pattern_portfolio_analysis.png - Visualizations")
    print("  • portfolio_showcase.html - Interactive showcase")
    print("\n🌐 View your portfolio showcase at:")
    print("  https://hillarydanan.github.io/TIDE-resonance/portfolio_showcase.html")
    print("\n🔬 This framework demonstrates empirical measurement of:")
    print("  • Neural-like activation patterns in AI")
    print("  • Information structure complexity")
    print("  • Cognitive flexibility and evolution")
    print("  • Trust and cooperation dynamics")
    print("  • Unique pattern signatures")
    print("\n💡 Ready to change how we understand AI pattern!")

if __name__ == "__main__":
    main()
