#!/bin/bash
streamlit run app.py \
  --server.port=8501 \
  --server.fileWatcherType=none \
  --server.headless=true \
  --browser.gatherUsageStats=false
