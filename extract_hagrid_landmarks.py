"""
Extracts 21-point hand landmarks from a folder of HaGRID "ok" gesture
images, using the EXACT SAME palm-detection + hand-landmark + normalization
pipeline app.py uses on live camera frames. This guarantees the resulting
training rows are apples-to-apples with what your live app actually sees,
rather than risking any drift from reimplementing the math separately.

Writes to a SEPARATE csv (not your real keypoint.csv) so you can sanity
check it with check_ok_data.py before merging it in.

Only keeps images where EXACTLY ONE hand is detected -- images with zero
or multiple detected hands are skipped and counted, to avoid mislabeling
a stray second hand (or a failed detection) as a clean "ok" example.

Usage:
    python extract_hagrid_landmarks.py --images-dir path/to/hagrid/ok --label OK
"""

import argparse
import csv
from math import degrees
from pathlib import Path

import cv2 as cv
import numpy as np

from model import PalmDetection, HandLandmark
from utils.utils import rotate_and_crop_rectangle
from app import pre_process_landmark


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images-dir', type=str, required=True,
        help='Folder of HaGRID images for one gesture (e.g. the "ok" subfolder).',
    )
    parser.add_argument(
        '--label', type=str, default='OK',
        help='Label name to write in the output CSV (default: %(default)s). '
             'Must match a name in keypoint_classifier_label.csv exactly.',
    )
    parser.add_argument(
        '--output', type=str, default='keypoint_hagrid_extracted.csv',
        help='Output CSV path (default: %(default)s). NOT your real '
             'keypoint.csv -- review/merge manually after sanity-checking.',
    )
    parser.add_argument(
        '--min-detection-confidence', type=float, default=0.6,
        help='Palm detection confidence threshold (default: %(default)s), '
             'matching app.py\'s default.',
    )
    parser.add_argument(
        '--label-csv', type=str,
        default='model/keypoint_classifier/keypoint_classifier_label.csv',
        help='Path to the label CSV, used to look up the numeric class '
             'index for --label (default: %(default)s).',
    )
    return parser.parse_args()


def get_label_index(label_csv_path, label_name):
    with open(label_csv_path, encoding='utf-8-sig') as f:
        labels = [row[0] for row in csv.reader(f)]
    if label_name not in labels:
        raise ValueError(
            f"Label '{label_name}' not found in {label_csv_path}. "
            f"Available labels: {labels}"
        )
    return labels.index(label_name)


def extract_single_image(image, palm_detection, hand_landmark):
    """Runs the same pipeline as app.py's main loop on one static image.
    Returns a pre_processed_landmark array, or None if the image doesn't
    have exactly one detected hand.
    """
    cap_height, cap_width = image.shape[:2]
    wh_ratio = cap_width / cap_height

    hands = palm_detection(image)
    if len(hands) != 1:
        return None  # skip: zero or multiple hands detected

    rects = []
    for hand in hands:
        sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y = hand
        cx = int(sqn_rr_center_x * cap_width)
        cy = int(sqn_rr_center_y * cap_height)
        xmin = int((sqn_rr_center_x - (sqn_rr_size / 2)) * cap_width)
        xmax = int((sqn_rr_center_x + (sqn_rr_size / 2)) * cap_width)
        ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2)) * cap_height)
        ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2)) * cap_height)
        xmin, xmax = max(0, xmin), min(cap_width, xmax)
        ymin, ymax = max(0, ymin), min(cap_height, ymax)
        degree = degrees(rotation)
        rects.append([cx, cy, (xmax - xmin), (ymax - ymin), degree])

    rects = np.asarray(rects, dtype=np.float32)

    cropted_rotated_hands_images = rotate_and_crop_rectangle(
        image=image,
        rects_tmp=rects,
        operation_when_cropping_out_of_range='padding',
    )
    if len(cropted_rotated_hands_images) == 0:
        return None

    hand_landmarks, _ = hand_landmark(
        images=cropted_rotated_hands_images,
        rects=rects,
    )
    if len(hand_landmarks) != 1:
        return None

    landmark = hand_landmarks[0]
    return pre_process_landmark(landmark)


def main():
    args = get_args()
    label_index = get_label_index(args.label_csv, args.label)
    print(f"Using label '{args.label}' -> class index {label_index}")

    images_dir = Path(args.images_dir)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    print(f"Found {len(image_paths)} images in {images_dir}")

    palm_detection = PalmDetection(score_threshold=args.min_detection_confidence)
    hand_landmark = HandLandmark()

    written = 0
    skipped_no_hand = 0
    skipped_multi_hand = 0
    skipped_error = 0

    with open(args.output, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        for i, path in enumerate(image_paths):
            image = cv.imread(str(path))
            if image is None:
                skipped_error += 1
                continue

            try:
                pre_processed_landmark = extract_single_image(
                    image, palm_detection, hand_landmark,
                )
            except Exception as e:
                skipped_error += 1
                print(f"  [{i}] error on {path.name}: {e}")
                continue

            if pre_processed_landmark is None:
                # Ambiguous whether it was 0 or >1 hands without re-checking;
                # good enough for a summary count either way.
                skipped_no_hand += 1
                continue

            writer.writerow([label_index, *pre_processed_landmark])
            written += 1

            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{len(image_paths)} "
                      f"(written={written}, skipped={skipped_no_hand + skipped_error})")

    print()
    print(f"Done. Wrote {written} rows to {args.output}")
    print(f"Skipped (no/multiple hand detected): {skipped_no_hand}")
    print(f"Skipped (errors): {skipped_error}")
    print()
    print("Next: sanity-check this file before merging into your real "
          "keypoint.csv, e.g.:")
    print(f"  python check_ok_data.py  # (point CSV_PATH at {args.output} first)")


if __name__ == '__main__':
    main()