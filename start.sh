#!/bin/bash
streamlit run streamlit_app.py \
  --server.port=8501 \
  --logger.level=debug \
  --server.fileWatcherType=none \
  --server.headless=true \
  --browser.gatherUsageStats=false
  
