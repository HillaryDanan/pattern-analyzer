# Pattern Analyzer Examples

⚠️ **IMPORTANT**: All data in this directory is **SAMPLE/DEMO DATA** for demonstration purposes only. The actual tools perform real scientific analysis on real data.

This directory contains examples demonstrating the integrated pipeline between TIDE-analysis and Pattern Analyzer.

## 🚨 Disclaimer

- **All numbers are fictional** - Created to show output format
- **All patterns are examples** - To demonstrate what real analysis produces  
- **All insights are samples** - Showing the types of findings possible
- **Real tools use real data** - No fake claims, pure data-driven science

## 🚀 Quick Start

### Run the Integrated Pipeline Demo

```bash
python integrated_pipeline_demo.py
```

This will:
1. Simulate TIDE analysis on a model
2. Pass results to Pattern Analyzer
3. Generate comprehensive reports
4. Save outputs to a timestamped directory

### View Sample Outputs

Check the `sample_outputs/` directory to see example results:
- `tide_output.json` - Raw TIDE analysis data
- `analysis_summary.txt` - Human-readable summary
- `detailed_metrics.csv` - Comprehensive metrics table

## 📊 Understanding the Pipeline

```
AI Model → TIDE Analysis → Pattern Analyzer → Insights
    ↓           ↓                ↓              ↓
  Input    Token Maps    14+ Math Tools   Actionable
           & Resonance    & Analysis       Reports
```

## 🔧 Using with Real Data

To use with actual models instead of simulated data:

```python
from tide_analysis import TIDEAnalyzer
from pattern_analyzer import PatternEngine

# Analyze a real model
tide = TIDEAnalyzer()
tide_results = tide.analyze('path/to/model')

# Process with Pattern Analyzer
engine = PatternEngine(tide_results)
insights = engine.comprehensive_analysis()

# Generate reports
engine.generate_reports('output_directory/')
```

## 📈 Key Metrics Explained

- **Complexity Score**: Overall architectural complexity (0-1)
- **Pattern Richness**: Diversity of discovered patterns (0-1)
- **Emergence Score**: Likelihood of emergent behaviors (0-1)
- **Resonance Events**: Synchronized activation patterns
- **Fractal Dimension**: Self-similarity across scales

## 🎯 What to Look For

When analyzing your own models:
1. High resonance in specific layers (optimization targets)
2. Emergence scores > 0.8 (complex behaviors)
3. Clustering patterns (functional modules)
4. Cross-layer interactions (distributed processing)

## 📚 Learn More

- [TIDE-analysis Documentation](https://github.com/HillaryDanan/TIDE-analysis)
- [Pattern Analyzer Documentation](https://github.com/HillaryDanan/pattern-analyzer)
- [Research Papers](https://github.com/HillaryDanan/AI-Architecture-Papers)