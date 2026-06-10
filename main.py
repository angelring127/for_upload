import os
import argparse
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
from collections import Counter

TRAILING_SEPARATOR_CHARS = " \t\r\n\u00a0-–—|"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
ANIME_MODEL_PATH = os.path.join(BASE_DIR, "models", "lbpcascade_animeface.xml")
DEFAULT_OUTPUT_SIZE = (750, 1080)

_yunet_detector = None
_yunet_input_size = None
_haar_cascades = None
_anime_cascade = None

def is_monochrome(image, threshold=0.95):
    """이미지が単色かどうかを確認"""
    if len(image.shape) == 3:  # カラー画像
        # 各チャンネルの標準偏差を計算
        std_dev = np.std(image, axis=(0, 1))
        # 標準偏差が非常に小さい場合は単色と見なす
        return np.all(std_dev < threshold)
    return True

def find_common_suffix(filenames, min_count=5):
    """ファイル名の末尾に共通して現れる文字列を見つける"""
    stems = [os.path.splitext(filename)[0] for filename in filenames]
    if len(stems) < min_count:
        return None

    suffix_counts = Counter()
    for index, stem in enumerate(stems):
        for other_stem in stems[index + 1:]:
            suffix_length = 0
            max_length = min(len(stem), len(other_stem))
            while (
                suffix_length < max_length
                and stem[-suffix_length - 1] == other_stem[-suffix_length - 1]
            ):
                suffix_length += 1

            if suffix_length >= 8:
                suffix_counts[stem[len(stem) - suffix_length:]] += 1

    candidates = []
    for suffix, pair_count in suffix_counts.items():
        matched_count = sum(stem.endswith(suffix) for stem in stems)
        if matched_count >= min_count:
            candidates.append((matched_count, len(suffix), pair_count, suffix))

    if candidates:
        return max(candidates)[3]
    return None

def remove_common_suffix(filename, common_suffix):
    """共通suffixが末尾にある場合だけ削除する"""
    stem, ext = os.path.splitext(filename)
    if common_suffix and stem.endswith(common_suffix):
        stem = stem[:-len(common_suffix)].rstrip(TRAILING_SEPARATOR_CHARS)
    return f"{stem}{ext}"

def get_unique_path(directory, filename, original_path=None):
    """既存ファイルを上書きしないリネーム先を作る"""
    target_path = os.path.join(directory, filename)
    if target_path == original_path or not os.path.exists(target_path):
        return target_path

    stem, ext = os.path.splitext(filename)
    index = 1
    while True:
        target_path = os.path.join(directory, f"{stem} ({index}){ext}")
        if target_path == original_path or not os.path.exists(target_path):
            return target_path
        index += 1

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def parse_output_size(value):
    try:
        width, height = value.lower().split("x", 1)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError
        return (width, height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("output size must look like 750x1080") from exc

def get_haar_cascades():
    global _haar_cascades
    if _haar_cascades is None:
        cascade_names = [
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_default.xml",
            "haarcascade_profileface.xml",
        ]
        _haar_cascades = []
        for name in cascade_names:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if not cascade.empty():
                _haar_cascades.append(cascade)
    return _haar_cascades

def get_anime_cascade():
    global _anime_cascade
    if _anime_cascade is None and os.path.exists(ANIME_MODEL_PATH):
        cascade = cv2.CascadeClassifier(ANIME_MODEL_PATH)
        if not cascade.empty():
            _anime_cascade = cascade
    return _anime_cascade

def detect_faces_yunet(frame):
    global _yunet_detector, _yunet_input_size

    if not os.path.exists(YUNET_MODEL_PATH):
        return []

    height, width = frame.shape[:2]
    input_size = (width, height)

    try:
        if _yunet_detector is None:
            _yunet_detector = cv2.FaceDetectorYN_create(
                YUNET_MODEL_PATH,
                "",
                input_size,
                0.6,
                0.3,
                5000,
            )
            _yunet_input_size = input_size
        elif _yunet_input_size != input_size:
            _yunet_detector.setInputSize(input_size)
            _yunet_input_size = input_size

        _, detections = _yunet_detector.detect(frame)
    except cv2.error:
        return []

    if detections is None:
        return []

    faces = []
    for detection in detections:
        x, y, w, h = detection[:4]
        confidence = float(detection[-1])
        x = int(clamp(round(x), 0, width - 1))
        y = int(clamp(round(y), 0, height - 1))
        w = int(clamp(round(w), 1, width - x))
        h = int(clamp(round(h), 1, height - y))
        faces.append({"box": (x, y, w, h), "confidence": confidence, "source": "yunet"})
    return faces

def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = gray.shape[:2]
    min_face_size = max(24, min(frame_width, frame_height) // 18)
    faces = []
    seen_boxes = set()

    for cascade in get_haar_cascades():
        detected = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        for x, y, w, h in detected:
            box = (int(x), int(y), int(w), int(h))
            if box in seen_boxes:
                continue
            seen_boxes.add(box)
            faces.append({"box": box, "confidence": 0.55, "source": "haar"})

    return faces

def detect_faces_anime(frame):
    cascade = get_anime_cascade()
    if cascade is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    frame_height, frame_width = gray.shape[:2]
    min_face_size = max(24, min(frame_width, frame_height) // 20)
    detected = cascade.detectMultiScale(
        gray,
        scaleFactor=1.06,
        minNeighbors=4,
        minSize=(min_face_size, min_face_size),
    )

    return [
        {"box": (int(x), int(y), int(w), int(h)), "confidence": 0.62, "source": "anime"}
        for x, y, w, h in detected
    ]

def box_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_width = max(0, min(ax2, bx2) - max(ax, bx))
    inter_height = max(0, min(ay2, by2) - max(ay, by))
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area else 0.0

def face_detection_rank(face):
    source_bonus = {"yunet": 0.2, "anime": 0.14, "haar": 0.0}.get(face.get("source"), 0.0)
    x, y, w, h = face["box"]
    return face.get("confidence", 0.5) + source_bonus + (w * h) / 1_000_000

def merge_duplicate_faces(faces):
    merged = []
    for face in sorted(faces, key=face_detection_rank, reverse=True):
        if all(box_iou(face["box"], kept["box"]) < 0.35 for kept in merged):
            merged.append(face)
    return merged

def detect_face_in_frame(frame):
    """Detect real, 3D, and 2D/anime faces in a frame."""
    faces = []
    faces.extend(detect_faces_yunet(frame))
    faces.extend(detect_faces_haar(frame))
    faces.extend(detect_faces_anime(frame))
    return merge_duplicate_faces(faces)

def frame_positions(total_frames, scan_count):
    if total_frames <= 1:
        return [0]

    start = int(total_frames * 0.08)
    end = max(start + 1, int(total_frames * 0.92))
    count = max(1, min(scan_count, total_frames))
    positions = np.linspace(start, end, num=count)
    return sorted({int(clamp(round(pos), 0, total_frames - 1)) for pos in positions})

def frame_quality_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
    contrast_score = min(contrast / 64.0, 1.0)
    sharpness_score = min(sharpness / 450.0, 1.0)
    score = brightness_score * 0.25 + contrast_score * 0.25 + sharpness_score * 0.5

    if is_monochrome(frame, threshold=2.0):
        score -= 1.5
    if brightness < 16 or brightness > 242:
        score -= 0.8

    return score

def score_faces(frame, faces):
    if not faces:
        return 0.0

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
    best_score = 0.0

    for face in faces:
        x, y, w, h = face["box"]
        area_ratio = (w * h) / frame_area
        cx = x + w / 2
        cy = y + h / 2
        dx = abs(cx - frame_width / 2) / (frame_width / 2)
        dy = abs(cy - frame_height / 2) / (frame_height / 2)
        center_score = 1.0 - min((dx + dy) / 2, 1.0)
        size_score = min(area_ratio * 80.0, 1.0)
        face_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        detail_score = min(float(cv2.Laplacian(face_gray, cv2.CV_64F).var()) / 500.0, 1.0)
        confidence = face.get("confidence", 0.5)
        source_bonus = {"yunet": 0.4, "anime": 0.32, "haar": 0.0}.get(face.get("source"), 0.0)
        best_score = max(
            best_score,
            size_score * 5.0 + confidence * 2.0 + center_score + detail_score + source_bonus,
        )

    multi_face_bonus = min(len(faces), 3) * 0.35
    return best_score + multi_face_bonus

def find_best_frame(cap, total_frames, scan_count):
    selected_frame = None
    selected_faces = []
    selected_score = None

    for frame_pos in frame_positions(total_frames, scan_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detect_face_in_frame(frame)
        score = frame_quality_score(frame)
        if faces:
            score += 10.0 + score_faces(frame, faces)

        if selected_score is None or score > selected_score:
            selected_frame = frame
            selected_faces = faces
            selected_score = score

    if selected_frame is not None:
        return selected_frame, selected_faces

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames // 2, 0))
    ret, frame = cap.read()
    if not ret:
        raise Exception("フレームを読み込めません")
    return frame, detect_face_in_frame(frame)

def representative_face_score(frame, face):
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
    x, y, w, h = face["box"]
    area_ratio = (w * h) / frame_area
    cx = x + w / 2
    cy = y + h / 2
    center_distance = (
        abs(cx - frame_width / 2) / (frame_width / 2)
        + abs(cy - frame_height / 2) / (frame_height / 2)
    ) / 2
    center_score = 1.0 - min(center_distance, 1.0)
    face_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    detail_score = min(float(cv2.Laplacian(face_gray, cv2.CV_64F).var()) / 500.0, 1.0)
    confidence = face.get("confidence", 0.5)
    source_bonus = {"yunet": 0.25, "anime": 0.2, "haar": 0.0}.get(face.get("source"), 0.0)
    return area_ratio * 100.0 + confidence * 2.0 + center_score + detail_score + source_bonus

def select_representative_face(frame, faces):
    return max(faces, key=lambda face: representative_face_score(frame, face))

def face_safe_rect(face, image_size):
    width, height = image_size
    x1, y1, face_width, face_height = face["box"]
    x2 = x1 + face_width
    y2 = y1 + face_height

    x1 -= face_width * 0.5
    x2 += face_width * 0.5
    y1 -= face_height * 0.7
    y2 += face_height * 1.35

    return (
        int(clamp(round(x1), 0, width)),
        int(clamp(round(y1), 0, height)),
        int(clamp(round(x2), 0, width)),
        int(clamp(round(y2), 0, height)),
    )

def max_crop_size(image_size, output_size):
    width, height = image_size
    output_width, output_height = output_size
    target_ratio = output_width / output_height

    if width / height > target_ratio:
        return int(round(height * target_ratio)), height
    return width, int(round(width / target_ratio))

def ratio_size_for_rect(rect, target_ratio):
    x1, y1, x2, y2 = rect
    required_width = max(1, x2 - x1)
    required_height = max(1, y2 - y1)

    if required_width / required_height > target_ratio:
        crop_width = required_width
        crop_height = int(round(crop_width / target_ratio))
    else:
        crop_height = required_height
        crop_width = int(round(crop_height * target_ratio))

    return crop_width, crop_height

def clamp_crop_to_safe_rect(left, top, crop_width, crop_height, image_size, safe_rect):
    width, height = image_size
    x1, y1, x2, y2 = safe_rect
    left_low = max(0, x2 - crop_width)
    left_high = min(x1, width - crop_width)
    top_low = max(0, y2 - crop_height)
    top_high = min(y1, height - crop_height)

    if left_low <= left_high:
        left = clamp(left, left_low, left_high)
    else:
        left = clamp(left, 0, width - crop_width)

    if top_low <= top_high:
        top = clamp(top, top_low, top_high)
    else:
        top = clamp(top, 0, height - crop_height)

    return int(round(left)), int(round(top))

def compute_face_crop(image_size, output_size, face):
    width, height = image_size
    output_width, output_height = output_size
    target_ratio = output_width / output_height
    safe_rect = face_safe_rect(face, image_size)
    safe_crop_width, safe_crop_height = ratio_size_for_rect(safe_rect, target_ratio)
    max_width, max_height = max_crop_size(image_size, output_size)
    _, _, face_width, face_height = face["box"]

    desired_width = max(
        safe_crop_width,
        int(round(face_width / 0.34)),
        int(round((face_height / 0.28) * target_ratio)),
    )
    desired_height = int(round(desired_width / target_ratio))

    if desired_height < safe_crop_height:
        desired_height = safe_crop_height
        desired_width = int(round(desired_height * target_ratio))

    crop_width = int(clamp(desired_width, min(safe_crop_width, max_width), max_width))
    crop_height = int(round(crop_width / target_ratio))
    if crop_height > max_height:
        crop_height = max_height
        crop_width = int(round(crop_height * target_ratio))

    crop_width = int(clamp(crop_width, 1, width))
    crop_height = int(clamp(crop_height, 1, height))

    x, y, w, h = face["box"]
    face_center_x = x + w / 2
    face_center_y = y + h / 2
    left = face_center_x - crop_width / 2
    top = face_center_y - crop_height * 0.40
    left, top = clamp_crop_to_safe_rect(left, top, crop_width, crop_height, image_size, safe_rect)
    return (left, top, left + crop_width, top + crop_height)

def focus_point_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = np.percentile(magnitude, 88)
    weights = np.where(magnitude >= threshold, magnitude, 0)

    if float(weights.sum()) <= 0:
        height, width = gray.shape[:2]
        return width / 2, height / 2

    ys, xs = np.indices(weights.shape)
    total = float(weights.sum())
    return float((xs * weights).sum() / total), float((ys * weights).sum() / total)

def compute_focus_crop(image_size, output_size, focus_point):
    width, height = image_size
    crop_width, crop_height = max_crop_size(image_size, output_size)
    focus_x, focus_y = focus_point
    left = int(round(clamp(focus_x - crop_width / 2, 0, width - crop_width)))
    top = int(round(clamp(focus_y - crop_height / 2, 0, height - crop_height)))
    return (left, top, left + crop_width, top + crop_height)

def compose_thumbnail(frame, faces, output_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    if faces:
        face = select_representative_face(frame, faces)
        crop_rect = compute_face_crop(image.size, output_size, face)
    else:
        crop_rect = compute_focus_crop(image.size, output_size, focus_point_from_frame(frame))

    return image.crop(crop_rect).resize(output_size, Image.Resampling.LANCZOS)

def save_jpeg_under_size(image, output_path, target_size_kb):
    quality = 95
    final_size_kb = None

    while True:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        final_size_kb = len(buffer.getvalue()) / 1024

        if final_size_kb <= target_size_kb or quality <= 50:
            break
        quality -= 5

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path, "JPEG", quality=quality, optimize=True)
    if final_size_kb and final_size_kb > target_size_kb:
        print(f"警告: {target_size_kb}KB以下にできませんでした ({final_size_kb:.1f}KB, quality={quality})")

def create_thumbnail(
    video_path,
    output_path,
    target_size_kb=80,
    output_size=DEFAULT_OUTPUT_SIZE,
    scan_count=48,
):
    """ビデオからサムネイルを生成（顔認識を使用）"""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("ビデオを開けません")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frame, faces = find_best_frame(cap, total_frames, scan_count)
        image = compose_thumbnail(selected_frame, faces, output_size)
        save_jpeg_under_size(image, output_path, target_size_kb)

        cap.release()
        return True
    except Exception as e:
        print(f"サムネイル生成中にエラーが発生: {str(e)}")
        if cap is not None:
            cap.release()
        return False

def rename_files(
    target_size_kb=80,
    output_size=DEFAULT_OUTPUT_SIZE,
    scan_count=48,
):
    # sourcesディレクトリのパス
    sources_dir = "sources"
    
    # サムネイル保存用ディレクトリを作成
    thumbnails_dir = os.path.join(sources_dir, "thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    # すべてのmp4ファイルを取得
    mp4_files = [f for f in os.listdir(sources_dir) if f.lower().endswith('.mp4')]
    
    # 共通のsuffixを見つける（5個以上で共通の場合）
    common_suffix = find_common_suffix(mp4_files, min_count=5)
    
    if common_suffix:
        print(f"共通のサフィックスが見つかりました: '{common_suffix}'")
        print(f"このサフィックスをファイル名から削除します。\n")
    else:
        print("共通のサフィックスが見つかりませんでした。\n")
    
    # ディレクトリ内のすべてのファイルに対して処理
    for filename in mp4_files:
        new_filename = remove_common_suffix(filename, common_suffix)
        
        # 完全なパスを生成
        old_file = os.path.join(sources_dir, filename)
        new_file = get_unique_path(sources_dir, new_filename, original_path=old_file)
        new_filename = os.path.basename(new_file)
        
        # ファイル名が変更された場合のみリネーム
        if old_file != new_file:
            os.rename(old_file, new_file)
            print(f"変更: {filename} -> {new_filename}")
        else:
            new_file = old_file
            print(f"変更なし: {filename}")
        
        # サムネイルを生成
        thumbnail_name = new_filename.replace('.mp4', '.jpg')
        thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name)
        if create_thumbnail(
            new_file,
            thumbnail_path,
            target_size_kb=target_size_kb,
            output_size=output_size,
            scan_count=scan_count,
        ):
            print(f"サムネイル生成完了: {thumbnail_name}")
        else:
            print(f"サムネイル生成失敗: {thumbnail_name}")
        print()

def build_parser():
    parser = argparse.ArgumentParser(description="Rename videos and create face-aware thumbnails.")
    parser.add_argument("--single", help="Create one thumbnail without renaming files.")
    parser.add_argument("--output", help="Output image path for --single.")
    parser.add_argument("--target-kb", type=int, default=80, help="Target JPEG size in KB.")
    parser.add_argument("--output-size", type=parse_output_size, default=DEFAULT_OUTPUT_SIZE, help="Thumbnail size, e.g. 750x1080.")
    parser.add_argument("--scan-count", type=int, default=48, help="Number of candidate frames to scan.")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.single:
        output_path = args.output
        if not output_path:
            stem, _ = os.path.splitext(args.single)
            output_path = f"{stem}.jpg"

        success = create_thumbnail(
            args.single,
            output_path,
            target_size_kb=args.target_kb,
            output_size=args.output_size,
            scan_count=args.scan_count,
        )
        if success:
            print(f"サムネイル生成完了: {output_path}")
            return 0
        return 1

    rename_files(
        target_size_kb=args.target_kb,
        output_size=args.output_size,
        scan_count=args.scan_count,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
