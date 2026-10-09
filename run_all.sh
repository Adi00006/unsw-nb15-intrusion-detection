#!/usr/bin/env bash
# Full pipeline on the real UNSW-NB15 files placed in ./data
set -euo pipefail
python -m src.eda
python -m src.train --task binary
python -m src.train --task multiclass
python -m src.train --task multiclass --smote --tag smote
echo "Done. Launch the demo with: streamlit run app/streamlit_app.py"
