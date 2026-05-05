# CINDERellA Interactive Results

Streamlit dashboard for exploring CINDERellA module-level and gene-level Bayesian-network outputs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the compact snapshot under `data/`. To regenerate that snapshot from the original Eden analysis folders:

```bash
python scripts/prepare_data.py
```

## Included Data

The data snapshot intentionally excludes large MATLAB files, input matrices, PDFs, and PNGs. It includes node maps, edge-frequency files, selection/correlation summaries, driver summaries, and therapeutic target annotations needed for interactive Plotly views.

