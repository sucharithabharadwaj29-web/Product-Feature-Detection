#!/usr/bin/env python3
"""Task A product feature identification with square-referenced circular-hole matching.

Features:
  - Outer silhouette: grayscale Otsu threshold + morphology + largest contour
  - Square opening: Canny edges + morphology + convex quadrilateral validation
  - Circular openings: Hough candidates + annular contrast + square-relative template matching

V3 is a reproducible OpenCV implementation, not the hidden implementation
of the earlier image-generation annotations. It is tuned as a starting point
for the supplied metallic-part photographs and must be validated on more views.

Examples:
  python feature_detection_v2.py product.jpg --debug
  python feature_detection_v2.py product.jpg --roi 250,300,420,360 --debug
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

# ------------------------- Default parameters -------------------------
# These are exposed as constants so they can be tuned for the camera setup.
OUTER_BLUR = (5, 5)
OUTER_BLUR_SIGMA = 1.2
OUTER_CLOSE_KERNEL_FRACTION = 0.041
OUTER_APPROX_EPSILON = 0.003  # fraction of outer contour perimeter

EDGE_BLUR = (5, 5)
EDGE_BLUR_SIGMA = 1.0
CANNY_LOW = 30
CANNY_HIGH = 90
SQUARE_CLOSE_KERNEL = 5
SQUARE_APPROX_EPSILON = 0.02
SQUARE_MIN_AREA_FRACTION = 0.025
SQUARE_MAX_AREA_FRACTION = 0.55
SQUARE_MAX_ASPECT_RATIO = 1.45

HOUGH_DP = 1.0
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 14
HOUGH_MIN_RADIUS_FRACTION = 0.045
HOUGH_MAX_RADIUS_FRACTION = 0.250
HOUGH_MIN_DISTANCE_FRACTION = 0.050
MIN_RING_EDGE_SUPPORT = 0.35
EXPECTED_CIRCLES = 3

# OpenCV uses BGR colour order.
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def choose_roi(image, roi_text=None, max_width=1000, max_height=760):
    """Select ROI on a screen-fitted preview and map it to original pixels."""
    h, w = image.shape[:2]
    if roi_text:
        x, y, rw, rh = map(int, roi_text.split(','))
    else:
        scale = min(max_width / w, max_height / h, 1.0)
        preview = cv2.resize(
            image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        cv2.namedWindow('Select complete product ROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select complete product ROI', preview.shape[1], preview.shape[0])
        cv2.imshow('Select complete product ROI', preview)
        print('Draw a rectangle around the whole component, with a little background margin.')
        print('Press ENTER/SPACE to confirm; press C to cancel.')
        rx, ry, rw_preview, rh_preview = cv2.selectROI(
            'Select complete product ROI', preview, showCrosshair=True, fromCenter=False
        )
        cv2.destroyWindow('Select complete product ROI')
        if rw_preview <= 0 or rh_preview <= 0:
            raise ValueError('ROI selection cancelled.')
        x = int(round(rx / scale))
        y = int(round(ry / scale))
        rw = int(round(rw_preview / scale))
        rh = int(round(rh_preview / scale))

    if rw <= 0 or rh <= 0:
        raise ValueError('ROI width and height must be positive.')
    x = max(0, x)
    y = max(0, y)
    rw = min(rw, w - x)
    rh = min(rh, h - y)
    if rw < 60 or rh < 60:
        raise ValueError('Selected ROI is too small. Select the complete product.')
    return x, y, rw, rh


def find_outer_boundary(roi):
    """Find dark metal silhouette against lighter background using Otsu."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, OUTER_BLUR, OUTER_BLUR_SIGMA)

    # Otsu chooses a threshold from this ROI's intensity histogram.
    otsu_threshold, mask = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Close tiny breaks in the dark silhouette. Keep this kernel modest.
    close_kernel = max(5, int(round(min(roi.shape[:2]) * OUTER_CLOSE_KERNEL_FRACTION)))
    if close_kernel % 2 == 0:
        close_kernel += 1
    k = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('No outer contour found. Check ROI and lighting.')

    # The ROI should contain one component; select the largest external region.
    outer = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(outer)
    if area < 0.01 * roi.shape[0] * roi.shape[1]:
        raise RuntimeError('Detected outer contour is too small; reselect ROI.')

    perimeter = cv2.arcLength(outer, True)
    smooth_outer = cv2.approxPolyDP(
        outer, OUTER_APPROX_EPSILON * perimeter, True
    )
    return gray, mask, outer, smooth_outer, float(otsu_threshold)


def find_square_cutout(gray, outer_contour):
    """Find the strongest approximately square quadrilateral inside the part."""
    blurred = cv2.GaussianBlur(gray, EDGE_BLUR, EDGE_BLUR_SIGMA)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    k = np.ones((SQUARE_CLOSE_KERNEL, SQUARE_CLOSE_KERNEL), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    outer_area = max(1.0, cv2.contourArea(outer_contour))
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (SQUARE_MIN_AREA_FRACTION * outer_area <= area <=
                SQUARE_MAX_AREA_FRACTION * outer_area):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(
            contour, SQUARE_APPROX_EPSILON * perimeter, True
        )
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > SQUARE_MAX_ASPECT_RATIO:
            continue

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        if cv2.pointPolygonTest(outer_contour, (cx, cy), False) < 0:
            continue

        # Prefer a near-square quadrilateral with a larger, stable contour.
        score = area / max(aspect, 1.0)
        candidates.append((score, approx, contour, aspect))

    if not candidates:
        return None, edges, closed_edges
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], edges, closed_edges


def ring_edge_support(edge_map, x, y, radius, samples=72):
    """Fraction of sampled circumference locations supported by nearby edges."""
    h, w = edge_map.shape
    hits = 0
    for angle in np.linspace(0, 2 * np.pi, samples, endpoint=False):
        px = int(round(x + radius * np.cos(angle)))
        py = int(round(y + radius * np.sin(angle)))
        x1, x2 = max(0, px - 1), min(w, px + 2)
        y1, y2 = max(0, py - 1), min(h, py + 2)
        if x1 < x2 and y1 < y2 and np.any(edge_map[y1:y2, x1:x2] > 0):
            hits += 1
    return hits / float(samples)



# Square-relative hole layout for THIS product family.
# Coordinates are normalized in the square's perspective-rectified frame.
# The four cyclic corner starts are all tested, so in-plane rotation does not
# require the user to orient the part in a fixed direction.
#
# Order: large hole, small hole A, small hole B.
HOLE_TEMPLATE = np.array([
    [-0.615, 1.085],
    [ 0.120, 1.220],
    [ 1.340, 1.240],
], dtype=np.float32)

# Expected radii divided by mean square side length.
# The small holes are allowed a wider range because the metallic rims and
# reflections affect the apparent Hough radius.
HOLE_RADIUS_RATIOS = np.array([0.180, 0.105, 0.090], dtype=np.float32)
HOLE_RADIUS_TOLERANCES = np.array([0.075, 0.070, 0.065], dtype=np.float32)

MIN_ANNULUS_CONTRAST = 12.0
MIN_TEMPLATE_DISTANCE = 0.22
MIN_CIRCLE_EDGE_SUPPORT = 0.22


def order_quad_corners(contour):
    """Return four quadrilateral corners in cyclic angular order."""
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError("Square contour must have exactly four corners")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def square_frame_transforms(square_contour):
    """
    Create four candidate homographies mapping the square to a unit square.

    The cyclic start corner is ambiguous under arbitrary in-plane rotation.
    Testing all four starts resolves that ambiguity using the known hole layout.
    """
    corners = order_quad_corners(square_contour)
    destination = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )
    transforms = []
    for shift in range(4):
        ordered = np.roll(corners, -shift, axis=0).astype(np.float32)
        H = cv2.getPerspectiveTransform(ordered, destination)
        transforms.append((shift, H, ordered))
    side_lengths = [
        np.linalg.norm(corners[(i + 1) % 4] - corners[i])
        for i in range(4)
    ]
    mean_side = float(np.mean(side_lengths))
    return transforms, mean_side


def annular_contrast(gray, x, y, radius):
    """
    Compare the dark hole interior with its bright metallic rim.

    Returns center mean, rim mean, outer-band mean, and rim-minus-center
    contrast. A dark seam without a closed bright rim tends to score poorly.
    """
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    center_region = distance <= max(2.0, 0.43 * radius)
    rim_region = (distance >= 0.76 * radius) & (distance <= 1.10 * radius)
    outer_region = (distance >= 1.18 * radius) & (distance <= 1.48 * radius)

    center_pixels = gray[center_region]
    rim_pixels = gray[rim_region]
    outer_pixels = gray[outer_region]

    if min(len(center_pixels), len(rim_pixels), len(outer_pixels)) == 0:
        return None

    center_mean = float(np.mean(center_pixels))
    rim_mean = float(np.mean(rim_pixels))
    outer_mean = float(np.mean(outer_pixels))
    contrast = rim_mean - center_mean

    return center_mean, rim_mean, outer_mean, contrast


def find_circular_holes(gray, outer_contour, square_contour=None):
    """
    Find the three known circular features using:
      1. HoughCircles to propose candidate centers/radii.
      2. Dark-center / bright-annulus contrast validation.
      3. A perspective-normalized feature layout referenced to the square.
      4. Radius-to-square-side consistency.
      5. One-to-one assignment so a single candidate cannot represent two holes.

    This is intentionally product-specific: the square-to-hole arrangement is
    stable for the supplied component. For a different part, recalibrate
    HOLE_TEMPLATE and HOLE_RADIUS_RATIOS using labelled reference images.
    """
    if square_contour is None:
        print("[WARNING] Square not detected; square-referenced hole matching skipped.")
        return [], cv2.Canny(gray, 40, 100)

    h, w = gray.shape
    transforms, square_side = square_frame_transforms(square_contour)

    min_radius = max(7, int(round(square_side * 0.045)))
    max_radius = max(min_radius + 2, int(round(square_side * 0.25)))
    min_dist = max(10, int(round(square_side * 0.05)))

    blurred = cv2.medianBlur(gray, 5)
    raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dist,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    edge_map = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 1.0), 40, 100
    )

    if raw is None:
        return [], edge_map

    candidates = []
    outer_poly = outer_contour.reshape(-1, 2)

    for x, y, radius in np.round(raw[0]).astype(int):
        if radius < min_radius or radius > max_radius:
            continue

        # Candidate center must lie inside the product's external contour.
        if cv2.pointPolygonTest(
            outer_poly, (float(x), float(y)), False
        ) < 0:
            continue

        # Reject candidates whose centers lie inside the square opening.
        if cv2.pointPolygonTest(
            square_contour, (float(x), float(y)), False
        ) >= 0:
            continue

        annulus = annular_contrast(gray, x, y, radius)
        if annulus is None:
            continue

        center_mean, rim_mean, outer_mean, contrast = annulus
        if contrast < MIN_ANNULUS_CONTRAST:
            continue

        support = ring_edge_support(edge_map, x, y, radius, samples=96)
        if support < MIN_CIRCLE_EDGE_SUPPORT:
            continue

        candidates.append({
            "center": (int(x), int(y)),
            "radius": int(radius),
            "center_mean": center_mean,
            "rim_mean": rim_mean,
            "outer_mean": outer_mean,
            "contrast": contrast,
            "support": float(support),
        })

    if not candidates:
        return [], edge_map

    # Deduplicate near-identical Hough proposals while retaining the proposal
    # with the strongest combined annulus contrast and edge support.
    candidates.sort(
        key=lambda c: (c["contrast"] / 80.0 + c["support"]),
        reverse=True
    )
    unique = []
    duplicate_distance = max(6, int(round(square_side * 0.035)))

    for candidate in candidates:
        x, y = candidate["center"]
        is_duplicate = False
        for previous in unique:
            px, py = previous["center"]
            if np.hypot(x - px, y - py) < duplicate_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(candidate)

    # Evaluate all four possible square-frame orientations.
    best_assignment = None
    best_total_cost = float("inf")
    best_shift = None

    for shift, H, _ in transforms:
        centers = np.array(
            [c["center"] for c in unique], dtype=np.float32
        ).reshape(1, -1, 2)
        normalized = cv2.perspectiveTransform(centers, H)[0]

        option_lists = []

        for feature_index, expected_xy in enumerate(HOLE_TEMPLATE):
            expected_radius = HOLE_RADIUS_RATIOS[feature_index] * square_side
            radius_tolerance = HOLE_RADIUS_TOLERANCES[feature_index] * square_side
            options = []

            for candidate_index, candidate in enumerate(unique):
                nx, ny = normalized[candidate_index]
                distance = float(np.linalg.norm(
                    np.array([nx, ny], dtype=np.float32) - expected_xy
                ))

                radius_error = abs(candidate["radius"] - expected_radius)
                if distance > MIN_TEMPLATE_DISTANCE:
                    continue
                if radius_error > radius_tolerance:
                    continue

                # Lower is better. Position and expected size dominate;
                # annulus contrast and edge support provide tie-breaking.
                contrast_bonus = min(candidate["contrast"], 70.0) / 140.0
                support_bonus = candidate["support"] * 0.20

                cost = (
                    4.0 * distance
                    + 1.25 * (radius_error / max(square_side, 1.0))
                    - contrast_bonus
                    - support_bonus
                )
                options.append((cost, candidate_index))

            options.sort(key=lambda item: item[0])
            option_lists.append(options[:10])

        if any(len(options) == 0 for options in option_lists):
            continue

        # The three template positions are well separated. Test the small
        # Cartesian product of the best options and require distinct circles.
        import itertools
        for combination in itertools.product(*option_lists):
            indices = [item[1] for item in combination]
            if len(set(indices)) != 3:
                continue
            total_cost = sum(item[0] for item in combination)
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_assignment = [unique[i] for i in indices]
                best_shift = shift

    if best_assignment is None:
        print("[WARNING] Could not confidently match all three circular features.")
        return [], edge_map

    print(f"[INFO] Square-reference orientation shift: {best_shift}")
    print(f"[INFO] Circular feature assignment cost: {best_total_cost:.3f}")

    # Return in semantic order: large hole, small A, small B.
    for label, candidate in zip(
        ["large", "small_1", "small_2"], best_assignment
    ):
        print(
            f"[CIRCLE] {label}: center={candidate['center']}, "
            f"radius={candidate['radius']}, "
            f"annulus_contrast={candidate['contrast']:.1f}, "
            f"edge_support={candidate['support']:.2f}"
        )

    return best_assignment, edge_map


def draw_legend(image):
    """Add a compact colour legend to an annotated image copy."""
    h, w = image.shape[:2]
    font_scale = max(0.45, min(0.8, w / 1100.0))
    thickness = max(1, int(round(font_scale * 2)))
    row_h = int(36 * font_scale / 0.65)
    labels = [
        ('Outer boundary', GREEN),
        ('Square cutout', CYAN),
        ('Large circular hole', MAGENTA),
        ('Small circular holes', YELLOW),
    ]
    panel_w = int(300 * font_scale / 0.65)
    panel_h = 16 + len(labels) * row_h
    x0, y0 = 14, 14
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.84, image, 0.16, 0, image)
    for i, (label, colour) in enumerate(labels):
        yy = y0 + 18 + i * row_h
        cv2.rectangle(image, (x0 + 12, yy - 9), (x0 + 38, yy + 9), colour, 3)
        cv2.putText(image, label, (x0 + 50, yy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, WHITE, thickness, cv2.LINE_AA)


def map_contour(contour, x_offset, y_offset):
    mapped = contour.copy()
    mapped[:, 0, 0] += x_offset
    mapped[:, 0, 1] += y_offset
    return mapped


def run(image_path, roi_text=None, debug=False, show=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    x0, y0, rw, rh = choose_roi(image, roi_text)
    roi = image[y0:y0 + rh, x0:x0 + rw].copy()

    gray, outer_mask, outer_raw, outer_smooth, otsu_t = find_outer_boundary(roi)
    square, edges, square_edges_closed = find_square_cutout(gray, outer_raw)
    circles, circle_edges = find_circular_holes(gray, outer_raw, square)

    # Draw on the ORIGINAL full-resolution image using ROI coordinate offsets.
    full_output = image.copy()
    cv2.drawContours(full_output, [map_contour(outer_smooth, x0, y0)], -1, GREEN, 3)

    if square is not None:
        cv2.drawContours(full_output, [map_contour(square, x0, y0)], -1, CYAN, 3)

    for index, circle in enumerate(circles):
        cx, cy = circle['center']
        colour = MAGENTA if index == 0 else YELLOW
        cv2.circle(full_output, (cx + x0, cy + y0), circle['radius'], colour, 3, cv2.LINE_AA)
        cv2.circle(full_output, (cx + x0, cy + y0), 2, WHITE, -1, cv2.LINE_AA)

    draw_legend(full_output)

    # Also produce a close-up crop for easy inspection.
    crop_output = full_output[y0:y0 + rh, x0:x0 + rw].copy()
    draw_legend(crop_output)

    base = Path(image_path)
    full_path = base.with_name(base.stem + '_annotated_v3_full.png')
    crop_path = base.with_name(base.stem + '_annotated_v3_crop.png')
    cv2.imwrite(str(full_path), full_output)
    cv2.imwrite(str(crop_path), crop_output)

    print('\n--- TASK A V3 RESULTS ---')
    print(f'ROI: x={x0}, y={y0}, width={rw}, height={rh}')
    print(f'Otsu threshold: {otsu_t:.1f}')
    print('Outer boundary: FOUND')
    print('Square cutout:', 'FOUND' if square is not None else 'NOT FOUND')
    print(f'Circular features detected: {len(circles)}')
    for i, c in enumerate(circles):
        label = 'large' if i == 0 else f'small {i}'
        print(f"  {label}: center={c['center']}, radius={c['radius']}, ring_support={c['support']:.2f}")
    print('Full image:', full_path)
    print('Close-up:', crop_path)

    if debug:
        debug_dir = base.with_name(base.stem + '_debug_v3')
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / '01_selected_roi.png'), roi)
        cv2.imwrite(str(debug_dir / '02_outer_mask.png'), outer_mask)
        cv2.imwrite(str(debug_dir / '03_outer_contour.png'), cv2.drawContours(roi.copy(), [outer_smooth], -1, GREEN, 2))
        cv2.imwrite(str(debug_dir / '04_square_canny.png'), edges)
        cv2.imwrite(str(debug_dir / '05_square_closed_edges.png'), square_edges_closed)
        cv2.imwrite(str(debug_dir / '06_circle_edges.png'), circle_edges)
        cv2.imwrite(str(debug_dir / '07_annotated_roi.png'), crop_output)
        print('V3 debug folder:', debug_dir)

    if show:
        # Fit display window to a manageable size; saved files remain full resolution.
        preview_scale = min(1100 / full_output.shape[1], 800 / full_output.shape[0], 1.0)
        preview = cv2.resize(full_output, None, fx=preview_scale, fy=preview_scale,
                             interpolation=cv2.INTER_AREA if preview_scale < 1 else cv2.INTER_LINEAR)
        cv2.namedWindow('Feature detection - full image', cv2.WINDOW_NORMAL)
        cv2.imshow('Feature detection - full image', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return full_output


def main():
    parser = argparse.ArgumentParser(description='Identify product features (Task A only).')
    parser.add_argument('image', help='Input photograph path')
    parser.add_argument('--roi', help='Optional ROI: x,y,width,height in original image pixels')
    parser.add_argument('--debug', action='store_true', help='Save intermediate debug images')
    parser.add_argument('--no-display', action='store_true', help='Do not open an OpenCV display window')
    args = parser.parse_args()
    run(args.image, args.roi, args.debug, not args.no_display)


if __name__ == '__main__':
    main()
