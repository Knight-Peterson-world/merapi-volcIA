#!/bin/bash
# start.sh — script de démarrage pour Render
# Render injecte $PORT automatiquement.

set -e

# Variables d'environnement nécessaires pour éviter les imports lourds
export USE_TF=0
export USE_TORCH=1
# Désactiver la génération IA en production (SD/LoRA trop lourd)
export RENDER_DEPLOYMENT=1

# Lancer Streamlit sur le port fourni par Render
exec streamlit run app/streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
