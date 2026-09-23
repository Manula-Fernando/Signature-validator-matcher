import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import Union, Tuple

def decode_image(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Decodes diverse input types (base64 string, file path, bytes, PIL Image, np.ndarray)
    into a standardized BGR numpy uint8 image.
    """
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        return image_input.copy()

    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return img

    if isinstance(image_input, str):
        # Check if base64 data URI or raw base64
        if "base64," in image_input:
            image_input = image_input.split("base64,")[1]
        
        # Try base64 decoding
        try:
            raw_bytes = base64.b64decode(image_input)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass

        # Otherwise treat as filesystem path
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not open or read image file at: {image_input}")
        return img

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def encode_image_base64(image: np.ndarray, format_ext: str = ".png") -> str:
    """Encodes a numpy image to base64 data URI string."""
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        raise ValueError("Could not encode image to format")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if format_ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def zhang_suen_thinning(binary_img: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization using the Zhang-Suen algorithm.
    Extracts the 1-pixel wide topological midline of the handwritten strokes,
    neutralizing stroke thickness differences caused by different pens
    (ballpoint vs gel pen vs marker vs stylus).
    
    binary_img: 2D uint8 array where foreground strokes are 255 and background is 0.
    """
    # Fast fallback: if opencv-contrib is present, cv2.ximgproc.thinning is available
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Pure OpenCV morphological skeletonization fallback
    skeleton = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary_img.copy()
    
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
            
    return skeleton


class SignaturePreprocessor:
    """
    Production-grade signature normalizer for offline signature verification.
    Provides background suppression, deskewing, pen-thickness normalization,
    and aspect-ratio preserved canvas projection.
    """

    def __init__(self, target_size: Tuple[int, int] = (105, 105)):
        self.target_size = target_size

    def clean_and_binarize(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suppresses document textures, security lines, and colored ink.
        Returns:
            (clean_gray, binary_stroke_mask)
            binary_stroke_mask: 255 for signature ink, 0 for paper background.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter preserves sharp stroke edges while smoothing paper grain
        filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        
        # Otsu thresholding with adaptive threshold fallback
        _, otsu_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Adaptive thresholding for uneven lighting / document shadows
        adaptive_thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        
        # Combine Otsu and Adaptive to capture faint ink tails while rejecting noise
        stroke_mask = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
        
        # Remove small isolated speckles / scanner dust
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stroke_mask, connectivity=8)
        cleaned_mask = np.zeros_like(stroke_mask)
        min_component_area = 15  # noise filter
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned_mask[labels == i] = 255
                
        return gray, cleaned_mask

    def normalize_pen_thickness(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Transforms any signature stroke to a standardized stroke width.
        Extracts the stroke skeleton and applies a uniform 2-pixel dilation.
        This ensures fine ballpoint and thick felt-tip pens map to identical representations.
        """
        skeleton = zhang_suen_thinning(binary_mask)
        # Dilate skeleton by standard 2-pixel kernel to restore natural stroke continuity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        standardized_strokes = cv2.dilate(skeleton, kernel, iterations=1)
        return standardized_strokes

    def crop_tight_bounding_box(self, binary_mask: np.ndarray, pad: int = 8) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops tight bounding box around the active signature ink.
        """
        points = cv2.findNonZero(binary_mask)
        if points is None:
            # Empty signature fallback
            return binary_mask, (0, 0, binary_mask.shape[1], binary_mask.shape[0])
            
        x, y, w, h = cv2.boundingRect(points)
        h_img, w_img = binary_mask.shape
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        cropped = binary_mask[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)

    def pad_and_resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the signature to target_size (e.g., 105x105) while strictly preserving
        aspect ratio and centering the stroke in a clean canvas.
        """
        target_w, target_h = self.target_size
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)

        # Scale factor preserving aspect ratio
        scale = min(target_w / w, target_h / h) * 0.88
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Centered canvas
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2

        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return canvas

    def preprocess_for_verification(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.
        Returns:
            normalized_tensor_input: 2D float32 array in [0, 1] formatted for Siamese Network (105, 105).
            raw_crop_display: Cropped signature for UI display.
            skeleton_display: Skeletonized stroke display for UI inspection.
        """
        bgr = decode_image(image_input)
        gray, stroke_mask = self.clean_and_binarize(bgr)
        
        # Pen thickness normalization
        skeleton = self.normalize_pen_thickness(stroke_mask)
        
        # Crop tight signature bounds
        cropped_skeleton, (x1, y1, x2, y2) = self.crop_tight_bounding_box(skeleton)
        raw_crop = bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else bgr
        
        # Resize to fixed canvas (105, 105)
        canvas_mask = self.pad_and_resize(cropped_skeleton)
        
        # Model input: Inverted so ink is dark on white or white on dark depending on model
        # For Mels22 verifier, input is grayscale normalized [0.0, 1.0]
        model_input = canvas_mask.astype(np.float32) / 255.0

        return model_input, raw_crop, canvas_mask
