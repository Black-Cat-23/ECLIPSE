# ECLIPSE 🔭
### *Exoplanet Classification & Light-curve Intelligence Pipeline for Space Exploration*

> **ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement PS-07**  
> *AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eclipse-isro/eclipse/blob/main/notebooks/02_model_training.ipynb)

---

## What Makes ECLIPSE Different

Every other team will build a **binary** planet vs. non-planet classifier. ECLIPSE directly solves what the PS-07 problem statement actually asks:

| Requirement | ECLIPSE | Typical Team |
|-------------|---------|-------------|
| 4-class classification (TRANSIT/EB/BLEND/OTHER) | ✅ | ❌ Binary |
| Transit parameter estimation P, τ, δ with uncertainty | ✅ | ❌ |
| SNR / significance per event | ✅ | ❌ |
| Raw TESS PDCSAP data from MAST | ✅ | ❌ Kaggle CSV |

---

## Architecture: ECLIPSE-PRIME

```
Raw PDCSAP Flux ──► Stream A: Temporal Anomaly Transformer ──┐
                         (ExoVeil-inspired, handles single      │
                          transits, no period needed)           │
                                                               ▼
TLS Period Search ──► Phase Fold ──► Stream B: Periodic ──► Cross-Attention
                           CNN + MHA Classifier                Fusion (256-d)
                           + Stellar Params + Centroid              │
                                                                    ▼
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                               4-class         P̂ ± σ_P         SNR ∈ ℝ+
                                               softmax         τ̂ ± σ_τ
                                            {TRANSIT,EB,       δ̂ ± σ_δ
                                             BLEND,OTHER}   (GaussNLL loss)
                                                    │
                                           MAPIE Conformal Wrapper
                                           90%/95% coverage guaranteed
```

**Novel contributions (nothing published combines all five):**
1. First architecture unifying ExoVeil-style raw-flux anomaly detection with periodic phase-fold classification via **cross-attention fusion**
2. First model simultaneously solving **4-class classification + parameter estimation + SNR** in one network
3. First use of **batman transit model as a soft physics constraint** in the loss function
4. Handles **single-transit events** (TESS 27-day sectors) that all prior vetting models fail on
5. **MAPIE conformal calibration** for statistically guaranteed uncertainty coverage

---

## Evaluation Criteria Mapping

| Criterion | Weight | ECLIPSE Feature |
|-----------|--------|----------------|
| Accuracy of event detection & classification | 35% | ECLIPSE-PRIME 4-class + TLS detection |
| Accuracy of transit parameters P, τ, δ | 25% | Multi-task regression + MCMC refinement |
| Methods / Approach / Novelty | 20% | Dual-stream + physics loss + conformal |
| Visualization & clarity | 20% | Interactive React dashboard + PDF reports |

---

## Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/eclipse-isro/eclipse
cd eclipse
pip install -r requirements.txt
cp .env.example .env
```

### 2. Download TESS Sector 1 Data
```bash
python -c "
from src.ingestion.tess_fetcher import TESSFetcher
from src.ingestion.catalog_loader import CatalogLoader

loader = CatalogLoader()
tic_ids = loader.get_sector_tic_ids(sector=1, limit=1000)
fetcher = TESSFetcher(sector=1)
fetcher.batch_download(tic_ids)
"
```

### 3. Build Training Dataset
```bash
python -c "
from src.preprocessing.dataset_builder import build_tce_catalog
build_tce_catalog(sector=1)
"
```

### 4. Train ECLIPSE-PRIME
```bash
python -m src.training.train --config configs/default.yaml
```

### 5. Run Inference on a Star
```bash
python -c "
from src.inference.pipeline import ECLIPSEInferencePipeline
pipe = ECLIPSEInferencePipeline(sector=1, model_path='checkpoints/best.pt')
result = pipe.run(tic_id=261136679)
print(result)
"
```

### 6. Launch Dashboard
```bash
# Terminal 1 — API
uvicorn api.main:app --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### Docker (all-in-one)
```bash
docker-compose up --build
```

---

## Project Structure
```
eclipse/
├── src/
│   ├── ingestion/        # TESS/Kepler data fetching
│   ├── preprocessing/    # Denoise → TLS → phase fold → features
│   ├── models/           # ECLIPSE-PRIME, streams, heads, losses
│   ├── training/         # Training loop, metrics, data loaders
│   ├── inference/        # End-to-end pipeline + batch processor
│   ├── xai/              # SHAP, attention rollout, Captum IG
│   └── evaluation/       # Benchmarks, injection-recovery
├── api/                  # FastAPI backend + WebSocket streaming
├── frontend/             # React + Vite + TailwindCSS dashboard
├── notebooks/            # 4 Colab-ready notebooks
└── tests/                # Unit + integration test suite
```

---

## Colab Training (T4 / A100)

Open `notebooks/02_model_training.ipynb` in Colab. The notebook:
- Installs all dependencies
- Downloads Sector 1 labeled data
- Trains ECLIPSE-PRIME with AMP + gradient checkpointing (fits T4 15GB)
- Saves checkpoint and logs metrics

---

## Loss Function
```
L_total = 1.0 × Focal_CE(4-class)
        + 0.5 × GaussianNLL(period)
        + 0.5 × GaussianNLL(duration)
        + 0.5 × GaussianNLL(depth)
        + 0.3 × MSE(SNR)
        + 0.2 × PhysicsConstraint(batman_consistency)
```

---

## Citation / References

- ExoVeil: Priyanshu (2026) arXiv:2606.02778
- Transit Least Squares: Hippke & Heller (2019) A&A 623, A39
- AstroNet: Shallue & Vanderburg (2018) AJ 155, 94
- batman: Kreidberg (2015) PASP 127, 1161
- MAPIE: Taquet et al. (2022) JMLR
- TESS: Ricker et al. (2015) JATIS 1, 014003

---

*Built for ISRO BAH 2026 by Team ECLIPSE.*
