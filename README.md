# Aadhaar Voice Verifier (NET AADHAR)

**AI-Powered Voice-Driven Identity Authentication System for the Visually Impaired**

A Python system that reads an Aadhaar card through a camera, authenticates it using the
card's secure QR code, cross-checks the printed number with computer vision, and speaks
the result aloud so visually impaired users can use it independently.

> ⚠️ **Academic prototype.** Not a replacement for official UIDAI verification. It does
> not validate the QR's digital signature. Only test with your own card or cards you have
> permission to use, and never commit images or data of real cards.

## How it works

1. **QR scan** – `pyzbar` decodes the Aadhaar QR (both the old XML format and the newer
   gzip-compressed secure QR) in a background thread, trying several image strategies
   (grayscale, adaptive threshold, 2x upscale) and caching the one that works.
2. **Card detection** – a custom **YOLOv8** model detects card regions
   (`AadharNumber`, `AadharEmblem`, `AadharName`, `AadharDOB`, ...).
3. **OCR confirmation** – **EasyOCR** reads the printed 12-digit number from the detected
   region; its last 4 digits must match the QR data.
4. **Vote-based decision** – the match must be seen in `OCR_VOTES_NEEDED` (3) frames, so a
   single misread can't pass or fail a card. If nothing is confirmed within
   `FRAME_WINDOW` (500) frames, the card is rejected.
5. **Voice output** – `pyttsx3` announces each stage and reads out the verified details.

**Design principle:** the QR data is authoritative; OCR only confirms that the printed
number matches it.

## Engineering notes

- Camera latency solved by draining the capture buffer (`grab()` x4, then `retrieve()`)
  with `CAP_PROP_BUFFERSIZE = 1`.
- QR decoding runs on its own thread so the video feed stays smooth.
- OCR restricted to digits and cropped to the densest text band to cope with the Tamil
  script printed on the card.
- Only the last 4 digits of the Aadhaar number are ever displayed.

## Setup

```bash
git clone https://github.com/<your-username>/aadhaar-voice-verifier.git
cd aadhaar-voice-verifier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pyzbar` also needs the ZBar library: `sudo apt install libzbar0` (Linux),
`brew install zbar` (macOS); on Windows it ships with the pip wheel.

## Run

Put your trained YOLO weights at `runs/detect/aadhar/weights/best.pt` (not included in the
repo), or pass `--model`.

```bash
# Laptop webcam
python aadhaar_verifier.py --source 0

# Phone as camera via DroidCam
export DROIDCAM_URL="http://<phone-ip>:4747/video"
python aadhaar_verifier.py --source droidcam

# Video file
python aadhaar_verifier.py --source video.mp4 --model path/to/best.pt
```

Keys: **Q** quit, **R** scan the next card.

## Tech stack

Python · YOLOv8 (Ultralytics) · EasyOCR · OpenCV · pyzbar · pyttsx3

## License

MIT
