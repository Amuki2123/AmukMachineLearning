#!/bin/bash
streamlit run your_app.py --server.port=8501 --logger.level=debug \
  --server.port=8501 \
  --server.fileWatcherType=none \
  --server.headless=true \
  --browser.gatherUsageStats=false
  
