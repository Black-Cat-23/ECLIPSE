---
title: ECLIPSE Exoplanet Discovery
emoji: 🪐
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ECLIPSE 🪐 
**Exoplanet Classification & Light-curve Intelligence Pipeline**

> *ISRO BAH 2026 Hackathon Submission*

![ECLIPSE Dashboard UI](https://img.shields.io/badge/UI-React%20%2B%20Vite-blue?style=for-the-badge&logo=react)
![ECLIPSE Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Machine Learning](https://img.shields.io/badge/ML-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)

ECLIPSE is a state-of-the-art, end-to-end pipeline designed to discover, validate, and analyze exoplanet candidates from raw photometric light curves (TESS / Kepler). Using a custom deep learning architecture and integrated Explainable AI (XAI), ECLIPSE not only finds planets—it proves *why* they exist.

## 🌟 Key Features

* **Real-time Sector Processing:** Ingests massive catalogs of TIC IDs and processes them through a highly optimized inference pipeline in milliseconds.
* **Explainable AI (XAI) Profiles:** Generates deep-learning attention heatmaps, SHAP feature importance charts, and conformal prediction sets to build trust in black-box models.
* **Habitability Assessment:** Automatically derives planetary radius, equilibrium temperature, and categorizes candidates into the Conservative/Optimistic Habitable Zone based on stellar parameters.
* **NASA-Grade PDF Reporting:** One-click generation of fully comprehensive scientific reports containing phase-folded light curves, astrometric centroid validation, and transit parameters.

## 🚀 Tech Stack

### Frontend (User Interface)
* **React 18** + **Vite** for blazing fast HMR and rendering.
* **Framer Motion** for butter-smooth cinematic UI transitions.
* **TailwindCSS** + **Lucide Icons** for a premium, futuristic space aesthetic.
* **React Query** for intelligent data caching and background fetching.

### Backend (Inference Pipeline)
* **FastAPI** for high-performance, asynchronous REST APIs and WebSockets.
* **PyTorch** for the underlying deep learning transit classification model.
* **ReportLab** + **Matplotlib** for generating dynamic, on-the-fly scientific PDF reports.
* **SQLite** for instant persistence of processed candidates.

## 🛠️ Local Development & Deployment

### 1. Start the Backend
ECLIPSE relies on a fast, asynchronous Python backend.
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\Activate on Windows
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### 2. Start the Frontend
The gorgeous user interface is built with Vite.
```bash
cd frontend
npm install
npm run dev
```

### 3. Production Deployment (Render)
ECLIPSE is designed to be easily hosted on platforms like Render:
1. Deploy the backend as a **Web Service** (Python 3 environment).
2. Deploy the `frontend/` directory as a **Static Site** (Node environment).
3. Set `VITE_API_URL` on the Static Site to point to your deployed backend URL.

## 🧠 The Science

ECLIPSE analyzes the **Phase-Folded Light Curve** of a target star. By applying a Box Least Squares (BLS) or Transit Least Squares (TLS) algorithm, we identify periodic dips in stellar flux. The depth of the dip tells us the **Planet Radius**, and the period allows us to calculate the **Equilibrium Temperature** via Kepler's Third Law, ultimately determining if the exoplanet resides in the habitable zone.

---
*Built with ❤️ for the ISRO BAH 2026 Hackathon*
