import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

class FootballDataset:
    def __init__(self, base_output_path="datasets/football_dataset"):
        """
        Folder structure:
        -football_dataset
            -images
                -train
                -test
            -labels
                -train
                -test
        """

        self.base_output_path = Path(base_output_path)
        self.images_base_path = self.base_output_path / "images"
        self.labels_base_path = self.base_output_path / "labels"


        self.train_images_path = self.images_base_path / "train"
        self.test_images_path = self.images_base_path / "test"
        self.train_labels_path = self.labels_base_path / "train"
        self.test_labels_path = self.labels_base_path / "test"


        for path in [
            self.images_base_path,
            self.labels_base_path,
            self.train_images_path,
            self.test_images_path,
            self.train_labels_path,
            self.test_labels_path
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _get_valid_paths(self, root):
        """Get only paths have both video and annotation
        Could use path without annotation for test"""
        video_paths = {str(path).replace(".mp4", "") for path in root.glob("**/*.mp4")}
        anno_paths = {str(path).replace(".json", "") for path in root.glob("**/*.json")}

        return list(video_paths & anno_paths)

    def _group_annotation_by_frame_id(self, annotations):
        grouped_annotations = defaultdict(list)

        for anno in annotations:
            if anno["category_id"] in [3, 4]:
                grouped_annotations[anno["image_id"]].append(anno)

        return grouped_annotations

    def _process_data(self, root, images_path, labels_path):
        frame_count = 0
        for path in self._get_valid_paths(root):
            video_path = f"{path}.mp4"
            anno_path = f"{path}.json"

            cap = cv2.VideoCapture(video_path)

            with open(anno_path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data["annotations"]
                image_width = json_data["images"][0]["width"]
                image_height = json_data["images"][0]["height"]

            grouped_annotations = self._group_annotation_by_frame_id(annotations)
            frame_id = 1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                objects = grouped_annotations[frame_id]

                label_filename = labels_path / f"{frame_count}.txt"
                frame_filename = images_path / f"{frame_count}.jpg"

                #save image
                cv2.imwrite(str(frame_filename), frame)

                #save label
                with open(label_filename, 'w') as f:
                    for obj in objects:
                        xmin, ymin, width, height = obj["bbox"]
                        category_id = obj["category_id"]


                        class_id = 0 if category_id == 4 else 1

                        # Convert bbox to YOLO format
                        xcenter = (xmin + width / 2) / image_width
                        ycenter = (ymin + height / 2) / image_height
                        norm_width = width / image_width
                        norm_height = height / image_height

                        # Write YOLO format annotation
                        f.write(f"{class_id} {xcenter:.6f} {ycenter:.6f} {norm_width:.6f} {norm_height:.6f}\n")

                frame_count += 1
                frame_id += 1

            cap.release()

        return frame_count



    def prepare_data(self, train_root="datasets/football_train", test_root="datasets/football_test"):
        """

        """
        train_frames = self._process_data(
            Path(train_root),
            self.train_images_path,
            self.train_labels_path
        )

        test_frames = self._process_data(
            Path(test_root),
            self.test_images_path,
            self.test_labels_path
        )

        return {
            "train": train_frames,
            "test": test_frames
        }

def main():
    converter = FootballDataset(base_output_path="datasets/football_dataset")
    start_time = time.time()
    processed_frames = converter.prepare_data(
        train_root="datasets/football_train",
        test_root = "datasets/football_test"
    )

    end_time = time.time()
    print("Processed Frames:")
    print(f"Train: {processed_frames['train']} frames")
    print(f"Test: {processed_frames['test']} frames")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()