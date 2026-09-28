from pathlib import Path
import tempfile

import cv2
import pandas as pd
import streamlit as st
import yaml

from src.model_manager import ensure_model
from src.detector import PPEModel
from src.compliance import extract, is_person, analyze_person


st.set_page_config(
    page_title="PPE Safety Dashboard V2",
    page_icon="🦺",
    layout="wide",
)

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

st.title("🦺 PPE / Helmet Detection Dashboard V2")
st.caption("AI-assisted workplace PPE compliance monitoring")

# Ensure model for dashboard upload inference.
with st.spinner("Checking PPE model..."):
    model_path = ensure_model(
        cfg["model"]["repo_id"],
        cfg["model"]["filename"],
        cfg["model"]["local_path"],
    )

csv_path = Path(cfg["output"]["log_csv"])
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(
        columns=["timestamp", "violation_type", "confidence", "source", "snapshot"]
    )

total = len(df)
no_helmet = int((df["violation_type"] == "NO_HELMET").sum()) if total else 0
no_vest = int((df["violation_type"] == "NO_VEST").sum()) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Incidents", total)
c2.metric("No Helmet", no_helmet)
c3.metric("No Vest", no_vest)
c4.metric("Model", "YOLOv11 PPE")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 History", "🖼 Image Detection", "🎥 Video Detection"])

with tab1:
    st.subheader("Violation History")
    if total:
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.subheader("Violation Distribution")
        st.bar_chart(df["violation_type"].value_counts())

        st.subheader("Evidence")
        for _, row in df.tail(10).iloc[::-1].iterrows():
            snap = Path(str(row["snapshot"]))
            if snap.exists():
                st.image(
                    str(snap),
                    caption=f'{row["timestamp"]} | {row["violation_type"]} | confidence {row["confidence"]}',
                    width=420,
                )
    else:
        st.info("No violations logged yet. Start webcam detection or upload an image/video.")

with tab2:
    st.subheader("Analyze an image")
    uploaded = st.file_uploader(
        "Upload JPG/PNG/WebP",
        type=["jpg", "jpeg", "png", "webp"],
        key="image",
    )

    if uploaded:
        data = uploaded.getvalue()
        tmp = Path(tempfile.gettempdir()) / f"ppe_{uploaded.name}"
        tmp.write_bytes(data)

        frame = cv2.imread(str(tmp))
        model = PPEModel(
            model_path,
            cfg["inference"]["confidence"],
            cfg["inference"]["iou"],
            cfg["inference"]["image_size"],
        )
        result = model.predict(frame)
        detections = extract(result)
        persons = [d for d in detections if is_person(d.name)]

        violations = []
        for d in detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"{d.name} {d.confidence:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2,
            )

        for person in persons:
            status = analyze_person(
                person,
                detections,
                cfg["rules"]["association_margin"],
                cfg["rules"]["missing_vest_enabled"],
            )
            if status["violation"]:
                violations.append(status)

        if violations:
            st.error(f"⚠️ {len(violations)} worker compliance violation(s) detected")
        else:
            st.success("✅ No violation detected by the configured rules")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)

with tab3:
    st.subheader("Video analysis")
    uploaded_video = st.file_uploader(
        "Upload MP4/AVI/MOV",
        type=["mp4", "avi", "mov"],
        key="video",
    )

    if uploaded_video:
        st.info(
            "For long videos, use `python app.py --source <video> --save-output` "
            "for faster batch processing."
        )
        video_path = Path(tempfile.gettempdir()) / f"ppe_{uploaded_video.name}"
        video_path.write_bytes(uploaded_video.getvalue())

        if st.button("Process Video", type="primary"):
            cap = cv2.VideoCapture(str(video_path))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_dir = Path(cfg["output"]["annotated_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / "dashboard_detection.mp4"

            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            model = PPEModel(
                model_path,
                cfg["inference"]["confidence"],
                cfg["inference"]["iou"],
                cfg["inference"]["image_size"],
            )

            progress = st.progress(0)
            preview = st.empty()
            index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(frame)
                detections = extract(result)

                for d in detections:
                    x1, y1, x2, y2 = d.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(
                        frame,
                        f"{d.name} {d.confidence:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 0),
                        2,
                    )

                writer.write(frame)
                index += 1

                if frames:
                    progress.progress(min(index / frames, 1.0))

                if index % 10 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
            writer.release()

            st.success(f"Finished. Output saved to {output}")
            st.video(str(output))
