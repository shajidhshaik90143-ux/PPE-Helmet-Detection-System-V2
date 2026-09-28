# PPE / Helmet Detection System V2

A stronger ready-to-run PPE safety monitoring project using Ultralytics YOLO and a publicly available YOLOv11 PPE model.

## What is fixed in V2?
V1 stopped with:
`FileNotFoundError: Model not found: models\best.pt`

V2 automatically downloads a compatible PPE model the first time you run it, so you do NOT need to manually find or copy `best.pt`.

The default model is:
`melihuzunoglu/ppe-detection`

The model documentation describes these labels:
- Human
- Helmet
- No-Helmet
- Vest

For no-vest, V2 uses a person-level rule: if a person is detected and no vest is detected inside/near that person's bounding box, the person is flagged for missing vest. This is an approximation and should be validated for your camera environment.

## Main features
- Automatic PPE model download
- Webcam detection
- Video file detection
- Image detection
- Helmet / no-helmet detection
- Safety vest detection
- Missing-vest rule
- Violation cooldown
- Evidence snapshots
- CSV incident log
- Real-time FPS
- Compliance counters
- Streamlit dashboard
- Upload image/video from dashboard
- Training template
- Configurable confidence / IoU
- Clean modular architecture

## Python
Your existing Python 3.10 environment is suitable for this project.

## Install
```powershell
cd "C:\Users\admin\Documents\VS CODE\PPE Helmet Detection System"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## First run
This automatically downloads the PPE model:
```powershell
python setup_model.py
```

Or simply:
```powershell
python app.py --source 0
```

## Webcam
```powershell
python app.py --source 0
```

Press `Q` or `ESC` to stop.

## Video
```powershell
python app.py --source "data/input/site.mp4" --save-output
```

Output:
```text
data/output/detection_output.mp4
```

## Image
```powershell
python app.py --source "data/input/worker.jpg"
```

The annotated image is saved under:
```text
data/output/
```

## Dashboard
```powershell
streamlit run dashboard.py
```

Open:
```text
http://localhost:8501
```

The dashboard can:
- show incident counts
- show violation history
- display evidence images
- upload an image for detection
- upload a video for detection

## Project structure
```text
PPE_Helmet_Detection_System_V2/
├── app.py
├── dashboard.py
├── setup_model.py
├── train.py
├── requirements.txt
├── README.md
├── config/
│   └── settings.yaml
├── models/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── model_manager.py
│   ├── detector.py
│   ├── compliance.py
│   ├── logger.py
│   └── utils.py
├── data/
│   ├── input/
│   ├── output/
│   └── violations/
├── reports/
│   └── PROJECT_REPORT.md
└── tools/
    └── health_check.py
```

## Important safety note
This is a computer-vision project for education/research and should not be treated as the sole safety control in a real workplace. Model accuracy depends on camera angle, lighting, distance, occlusion, worker clothing, and training data.

## Resume description
Built a real-time PPE compliance monitoring system using Python, OpenCV, Ultralytics YOLO, and Streamlit. The system detects workers, helmets, missing helmets and safety vests, applies worker-level compliance rules, captures violation evidence, maintains CSV incident logs, and provides an interactive monitoring dashboard.
