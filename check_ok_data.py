"""
Checks logged class-3 (OK sign) training rows in keypoint.csv for whether
they actually represent a real thumb-index pinch, based on the same
pre_process_landmark normalization the live app uses.

Run from the repo root:
    python check_ok_data.py
"""
import csv
import math

CSV_PATH = "model/keypoint_classifier/keypoint.csv"
TARGET_CLASS = "3"
PINCH_THRESHOLD = 0.15  # distance above this = probably not actually pinched

with open(CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    rows = [row for row in reader if row and row[0] == TARGET_CLASS]

if not rows:
    print(f"No rows found for class {TARGET_CLASS}.")
    raise SystemExit

distances = []
for row in rows:
    vals = list(map(float, row[1:]))
    thumb_tip = (vals[8], vals[9])    # landmark point 4
    index_tip = (vals[16], vals[17])  # landmark point 8
    d = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    distances.append(d)

distances.sort()
n = len(distances)
bad = sum(1 for d in distances if d > PINCH_THRESHOLD)

print(f"Total class-{TARGET_CLASS} rows: {n}")
print(f"Min distance:  {distances[0]:.4f}")
print(f"Max distance:  {distances[-1]:.4f}")
print(f"Mean distance: {sum(distances)/n:.4f}")
print(f"Median distance: {distances[n//2]:.4f}")
print(f"Rows with thumb-index distance > {PINCH_THRESHOLD} "
      f"(likely NOT a real pinch): {bad} ({bad/n:.1%})")
print()
print("10 worst (largest distance) samples:")
for d in distances[-10:]:
    print(f"  {d:.4f}")