"""ECLIPSE XAI — Explainability module package init."""
from .shap_explainer import SHAPExplainer, explain_prediction
from .attention_viz import render_attention_heatmap

__all__ = ["SHAPExplainer", "explain_prediction", "render_attention_heatmap"]
