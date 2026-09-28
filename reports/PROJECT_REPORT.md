# PPE / Helmet Detection System V2

## Abstract
This project implements an AI-assisted workplace safety monitoring system. A YOLO-based PPE detector identifies workers, helmets, missing helmets and safety vests. A rule engine associates PPE detections with workers and records potential violations.

## V2 improvement
The first version required `models/best.pt` to be manually supplied. V2 uses Hugging Face Hub to download a compatible PPE model automatically on first execution.

## Architecture
Input → YOLO PPE Detection → Person/PPE Association → Compliance Rules → Evidence Snapshot → CSV Log → Streamlit Dashboard

## Technologies
Python, Ultralytics YOLO, OpenCV, Pandas, PyYAML, Streamlit, Hugging Face Hub.

## Detection logic
- No-helmet: explicit `No-Helmet` detection without a simultaneous helmet detection.
- Missing vest: person detected without a vest inside/near the person box.
- A cooldown prevents the same continuing event from creating a new CSV row every frame.

## Limitations
The missing-vest rule is an absence-based heuristic. Camera angle and occlusion can produce false positives. For a production system, train/fine-tune a model on the target site and use tracking and calibrated PPE association.

## Future enhancements
- Multi-camera RTSP support
- Worker tracking IDs
- SQLite/PostgreSQL
- Email/SMS alerts
- WebSocket live dashboard
- Zone-based PPE policies
- Daily PDF reports
- GPU/TensorRT deployment
