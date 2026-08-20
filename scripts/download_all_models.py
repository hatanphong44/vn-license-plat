import paddle
from paddleocr import TextDetection, TextRecognition

PADDLE_DEVICE = "gpu:0"

print("Loading PP-OCRv6_small_det...")

det_model = TextDetection(
    model_name="PP-OCRv6_small_det",
    device=PADDLE_DEVICE,
)

print("PP-OCRv6_small_det loaded!")

print("Loading PP-OCRv6_small_rec...")

rec_model = TextRecognition(
    model_name="PP-OCRv6_small_rec",
    device=PADDLE_DEVICE,
)

print("PP-OCRv6_small_rec loaded!")

print("Device:", PADDLE_DEVICE)