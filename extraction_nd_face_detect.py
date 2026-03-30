import cv2
import os
import logging
from insightface.app import FaceAnalysis
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class PersonalDatasetBuilder:
    def __init__(self):
        self.logger = logging.getLogger("DatasetBuilder")

        try:
            self.scrfd_model = FaceAnalysis(
                name='buffalo_s',
                providers=['CPUExecutionProvider']
            )
            self.scrfd_model.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("SCRFD loaded successfully")
        except Exception as e:
            self.logger.error(f"SCRFD init failed: {e}")

    def process_video(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return

        video_name = Path(video_path).stem

        last_processed_second = -1
        frame_count = 0
        saved_faces = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            current_second = int(timestamp_ms / 500)  # 2 FPS sampling

            if current_second == last_processed_second:
                continue
            last_processed_second = current_second

            frame_count += 1

            # upscale for better detection (your logic)
            scale_factor = 1.5
            frame = cv2.resize(
                frame, None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC
            )

            faces = self.scrfd_model.get(frame)

            if len(faces) == 0:
                continue

            for i, face in enumerate(faces):
                det_score = getattr(face, 'det_score', 0)

                if det_score < 0.5:
                    continue

                x1, y1, x2, y2 = face.bbox.astype(int)

                # add margin (your logic)
                margin_w = int((x2 - x1) * 0.4)
                margin_h = int((y2 - y1) * 0.4)

                x1_new = max(0, x1 - margin_w)
                y1_new = max(0, y1 - margin_h)
                x2_new = min(frame.shape[1], x2 + margin_w)
                y2_new = min(frame.shape[0], y2 + margin_h)

                face_crop = frame[y1_new:y2_new, x1_new:x2_new]

                if face_crop.size == 0:
                    continue

                face_crop = cv2.resize(face_crop, (224, 224))

                save_path = os.path.join(
                    output_dir,
                    f"{video_name}_frame{frame_count}_face{i}.jpg"
                )

                cv2.imwrite(save_path, face_crop)
                saved_faces += 1

        video.release()

        self.logger.info(f"Processed video: {video_name}")
        self.logger.info(f"Frames processed: {frame_count}")
        self.logger.info(f"Faces saved: {saved_faces}")

    def process_all_videos(self, input_folder, output_folder):
        input_folder = Path(input_folder)

        video_files = list(input_folder.glob("*.*"))

        if not video_files:
            self.logger.warning("No videos found")
            return

        self.logger.info(f"Found {len(video_files)} videos")

        for video_path in video_files:
            self.logger.info(f"Processing: {video_path}")
            self.process_video(str(video_path), output_folder)


if __name__ == "__main__":
    builder = PersonalDatasetBuilder()

    INPUT_DIR = "self_vid_data_raw"
    OUTPUT_DIR = "dataset/personal_raw_faces"

    builder.process_all_videos(INPUT_DIR, OUTPUT_DIR)