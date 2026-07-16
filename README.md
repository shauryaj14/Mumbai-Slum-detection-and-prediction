# Mumbai-Slum-detection-and-prediction
# Mapping and Predicting Informal Settlement Expansion in Mumbai

A satellite-imagery and machine-learning pipeline that detects informal settlements
in Mumbai from Sentinel-2 imagery and predicts where they are likely to expand next.
Built as a CREST Award project.

**Live portfolio:** https://shauryaj14.github.io/Mumbai-Slum-detection-and-prediction/

## What this does

1. Pulls annual Sentinel-2 satellite composites of Mumbai via Google Earth Engine
2. Detects informal settlement extent using a SegFormer deep learning model
   adapted to 10-band multispectral imagery
3. Aggregates predictions onto a 160m monitoring grid across 6 years (2016-2021)
4. Trains a Random Forest classifier to predict which areas are likely to newly
   expand, using settlement density and spatial adjacency features

## Repository structure

- `scripts/` - full pipeline: data acquisition, model training, grid construction,
  feature engineering, and expansion prediction
- `code.ipynb` - main notebook for training and evaluating the detection model
- `docs/` - full written reports (methodology, results, conclusion, CREST report)
- `project-site/` - interactive results website
- `index.html` - this portfolio page

## Key results

- Detection model: F1 = 0.676, IoU = 0.510, validated consistently across two
  independent ground-truth years
- Expansion prediction: 68% recall at the selected operating threshold
- Found that settlement adjacency, not infrastructure proximity, drives expansion

## Data sources

- Sentinel-2 (ESA/Copernicus) via Google Earth Engine
- Slum Rehabilitation Authority (SRA), Maharashtra - ground truth boundaries
- OpenStreetMap - geospatial context features

## Limitations

Only two years of verified ground truth exist; other years rely on model
predictions. This system is a monitoring and prioritization aid, not an
authoritative predictive tool. See `docs/methodology.docx` for full detail.
