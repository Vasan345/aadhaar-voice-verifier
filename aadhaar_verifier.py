import cv2, gzip, io, math, os, re, time, threading, argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import deque

import numpy as np
import pyttsx3
from pyzbar.pyzbar import decode, ZBarSymbol
from ultralytics import YOLO
import easyocr

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
# Set your phone's DroidCam address via an environment variable, e.g.
#   export DROIDCAM_URL="http://<phone-ip>:4747/video"
DROIDCAM_URL     = os.environ.get("DROIDCAM_URL", "http://192.168.0.100:4747/video")
YOLO_CONF        = 0.25
FRAME_WINDOW     = 500
OCR_VOTES_NEEDED = 3

CLASS_NAMES = ['AadharAddress','AadharBack','AadharDOB','AadharEmblem',
               'AadharFront','AadharGOI','AadharGender','AadharName',
               'AadharNumber','AadharUAIDI']

C_GREEN  = (0, 220, 0)
C_RED    = (0, 0, 220)
C_YELLOW = (0, 215, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_CYAN   = (255, 255, 0)
C_BLUE   = (255, 150, 0)


# ──────────────────────────────────────────────
# VOICE
# ──────────────────────────────────────────────
class Voice:
    def __init__(self):
        self._busy     = False
        self._voice_id = None
        try:
            tmp = pyttsx3.init()
            for v in tmp.getProperty('voices'):
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    self._voice_id = v.id
                    break
            tmp.stop()
        except Exception:
            pass
        print(f"      Voice id: {self._voice_id or 'default'}")

    def speak(self, text):
        if self._busy:
            return
        def _run():
            self._busy = True
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                if self._voice_id:
                    engine.setProperty('voice', self._voice_id)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[Voice Error] {e}")
            finally:
                self._busy = False
        threading.Thread(target=_run, daemon=True).start()


# ──────────────────────────────────────────────
# QR DECODE HELPERS
# ──────────────────────────────────────────────
def decode_new_qr(raw_bytes):
    try:
        if isinstance(raw_bytes, (int, str)):
            n = int(raw_bytes)
            raw_bytes = n.to_bytes(math.ceil(n.bit_length() / 8), 'big')
        if raw_bytes[:2] != b'\x1f\x8b':
            return None
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as f:
            data = f.read()
        if data and data[0] == 0xff:
            data = data[1:]
        fields = [p.decode('utf-8', errors='replace').strip()
                  for p in data.split(b'\xff')]

        composite   = fields[2] if len(fields) > 2 else ""
        digits_only = re.sub(r'\D', '', composite)
        last4       = digits_only[:4] if len(digits_only) >= 4 else "XXXX"

        name   = fields[3]  if len(fields) > 3  else "N/A"
        dob    = fields[4]  if len(fields) > 4  else "N/A"
        gen    = fields[5]  if len(fields) > 5  else "N/A"
        gender = "Male" if gen == "M" else ("Female" if gen == "F" else gen)
        careof = fields[6]  if len(fields) > 6  else "N/A"
        dist   = fields[7]  if len(fields) > 7  else "N/A"
        pin    = fields[11] if len(fields) > 11 else "N/A"
        state  = fields[13] if len(fields) > 13 else "N/A"
        street = fields[14] if len(fields) > 14 else "N/A"

        return dict(last4=last4, name=name, dob=dob, gender=gender,
                    careof=careof, street=street, district=dist,
                    state=state, pincode=pin,
                    display=f"XXXX-XXXX-{last4}")
    except Exception as e:
        print(f"[QR-New Error] {e}")
        return None


def decode_old_qr(text):
    try:
        d   = ET.fromstring(text).attrib
        uid = d.get('uid', '')
        return dict(last4=uid[-4:] if len(uid) >= 4 else "XXXX",
                    name=d.get('name',    'N/A'),
                    dob=d.get('dob',      'N/A'),
                    gender="Male" if d.get('gender') == 'M' else "Female",
                    careof=d.get('co',    'N/A'),
                    street=d.get('street','N/A'),
                    district=d.get('dist','N/A'),
                    state=d.get('state',  'N/A'),
                    pincode=d.get('pc',   'N/A'),
                    display=f"XXXX-XXXX-{uid[-4:] if len(uid) >= 4 else 'XXXX'}")
    except Exception as e:
        print(f"[QR-Old Error] {e}")
        return None


def _parse_qr_obj(obj):
    raw = obj.data
    try:
        txt = raw.decode('utf-8')
    except Exception:
        txt = None
    if txt and "PrintLetterBarcodeData" in txt:
        return decode_old_qr(txt)
    if raw[:2] == b'\x1f\x8b':
        return decode_new_qr(raw)
    if txt and txt.isdigit() and len(txt) > 100:
        return decode_new_qr(txt)
    return None


# ──────────────────────────────────────────────
# THREADED QR SCANNER
# ──────────────────────────────────────────────
class QRScanner:
    """
    Runs QR decode in a background thread.
    Main loop feeds frames in; results are read back non-blocking.
    """
    def __init__(self):
        self._frame    = None
        self._result   = None          # (data, rect) or None
        self._lock     = threading.Lock()
        self._event    = threading.Event()
        self._stop     = False
        self._strategy = None          # cache best strategy
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def feed(self, frame):
        """Give the worker a new frame to process."""
        with self._lock:
            self._frame = frame.copy()
        self._event.set()

    def get_result(self):
        """Return latest (data, rect) and clear it, or None."""
        with self._lock:
            r = self._result
            self._result = None
        return r

    def stop(self):
        self._stop = True
        self._event.set()

    def _attempt(self, img, label, scale=1):
        objs = decode(img, symbols=[ZBarSymbol.QRCODE])
        if not objs:
            return None, None
        data = _parse_qr_obj(objs[0])
        if not data:
            return None, None
        if scale > 1:
            x, y, rw, rh = objs[0].rect
            rect = (x // scale, y // scale, rw // scale, rh // scale)
        elif scale == 0:
            rect = None
        else:
            rect = objs[0].rect
        print(f"[QR] decoded via: {label}")
        self._strategy = label
        return data, rect

    def _worker(self):
        while not self._stop:
            self._event.wait()
            self._event.clear()
            if self._stop:
                break

            with self._lock:
                frame = self._frame
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # ── Fast path: last known good strategy ──
            if self._strategy == "gray":
                d, r = self._attempt(gray, "gray")
                if d:
                    with self._lock: self._result = (d, r)
                    continue

            if self._strategy == "adapt":
                adapt = cv2.adaptiveThreshold(gray, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 51, 9)
                d, r = self._attempt(adapt, "adapt")
                if d:
                    with self._lock: self._result = (d, r)
                    continue

            if self._strategy == "up2":
                up2 = cv2.resize(gray, (w*2, h*2),
                                 interpolation=cv2.INTER_CUBIC)
                d, r = self._attempt(up2, "up2", scale=2)
                if d:
                    with self._lock: self._result = (d, r)
                    continue

            if self._strategy == "up2_adapt":
                up2  = cv2.resize(gray, (w*2, h*2),
                                  interpolation=cv2.INTER_CUBIC)
                adp  = cv2.adaptiveThreshold(up2, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 51, 9)
                d, r = self._attempt(adp, "up2_adapt", scale=2)
                if d:
                    with self._lock: self._result = (d, r)
                    continue

            # ── Full strategy sweep ───────────────────
            # 1. plain gray
            d, r = self._attempt(gray, "gray")
            if d:
                with self._lock: self._result = (d, r)
                continue

            # 2. adaptive threshold
            adapt = cv2.adaptiveThreshold(gray, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 51, 9)
            d, r = self._attempt(adapt, "adapt")
            if d:
                with self._lock: self._result = (d, r)
                continue

            # 3. 2x upscale
            up2 = cv2.resize(gray, (w*2, h*2),
                             interpolation=cv2.INTER_CUBIC)
            d, r = self._attempt(up2, "up2", scale=2)
            if d:
                with self._lock: self._result = (d, r)
                continue

            # 4. 2x upscale + adaptive
            adp2 = cv2.adaptiveThreshold(up2, 255,
                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv2.THRESH_BINARY, 51, 9)
            d, r = self._attempt(adp2, "up2_adapt", scale=2)
            if d:
                with self._lock: self._result = (d, r)
                continue


# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────
def ocr_check_last4(reader, frame, bbox, qr_last4):
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 12
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "", None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5,
                          interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h_th = th.shape[0]
        row_black = (th == 0).sum(axis=1)
        if row_black.max() == 0:
            return False, "", None

        threshold_density = row_black.max() * 0.4
        digit_rows = row_black >= threshold_density

        in_band = False
        band_start = 0
        best_band = (0, h_th)
        best_len  = 0
        for i, is_digit in enumerate(digit_rows):
            if is_digit and not in_band:
                band_start = i
                in_band    = True
            elif not is_digit and in_band:
                in_band  = False
                band_len = i - band_start
                if band_len > best_len:
                    best_len  = band_len
                    best_band = (band_start, i)
        if in_band:
            band_len = h_th - band_start
            if band_len > best_len:
                best_band = (band_start, h_th)

        r0 = max(0, best_band[0]-10)
        r1 = min(h_th, best_band[1]+10)
        digit_crop = th[r0:r1, :]

        results  = reader.readtext(digit_crop,
                                   allowlist='0123456789 ', detail=0)
        raw_text = ' '.join(results)
        print(f"    [OCR] band={best_band}  raw={repr(raw_text)}")

        spaced = re.search(r'(\d{4})\s+(\d{4})\s+(\d{4})', raw_text)
        if spaced:
            aadhaar_num = spaced.group(1)+spaced.group(2)+spaced.group(3)
            ocr_last4   = spaced.group(3)
            print(f"    [OCR] spaced: {spaced.group()}  last4={ocr_last4}")
            return True, aadhaar_num, ocr_last4

        all_digits = re.sub(r'\D', '', raw_text)
        m = re.search(r'\d{12}', all_digits)
        if m:
            aadhaar_num = m.group()
            ocr_last4   = aadhaar_num[-4:]
            print(f"    [OCR] 12-digit: {aadhaar_num}  last4={ocr_last4}")
            return True, aadhaar_num, ocr_last4

        print(f"    [OCR] no match in: {all_digits}")
        return False, all_digits, None

    except Exception as e:
        print(f"[OCR Error] {e}")
        return False, "", None


# ──────────────────────────────────────────────
# GUI HELPERS
# ──────────────────────────────────────────────
def draw_progress(frame, done, total, x, y, bw, bh, label):
    pct   = min(done/total, 1.0)
    fill  = int(bw*pct)
    color = C_GREEN if pct > 0.6 else (C_YELLOW if pct > 0.3 else C_RED)
    cv2.rectangle(frame, (x,y), (x+bw, y+bh), (50,50,50), -1)
    if fill > 0:
        cv2.rectangle(frame, (x,y), (x+fill, y+bh), color, -1)
    cv2.rectangle(frame, (x,y), (x+bw, y+bh), C_WHITE, 1)
    cv2.putText(frame, label, (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1)


def draw_panel(frame, data, verdict=None, ocr_raw="", ocr_votes=None):
    h, w   = frame.shape[:2]
    pw, px = 330, w-342
    py, ph = 10, 410

    ov = frame.copy()
    cv2.rectangle(ov, (px-8,py), (px+pw, py+ph), (15,15,15), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    hc = (C_GREEN if verdict=="REAL" else
          C_RED   if verdict=="FAKE" else C_YELLOW)
    cv2.rectangle(frame, (px-8,py), (px+pw, py+34), hc, -1)
    htxt = ("VERIFIED" if verdict=="REAL" else
            "REJECTED" if verdict=="FAKE" else "SCANNING...")
    cv2.putText(frame, f"  AADHAAR  {htxt}", (px-4, py+23),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, C_BLACK, 1)

    if data:
        rows = [("Name",     data['name']),
                ("Aadhaar",  data['display']),
                ("DOB",      data['dob']),
                ("Gender",   data['gender']),
                ("Care Of",  data['careof']),
                ("Street",   data['street']),
                ("District", data['district']),
                ("State",    data['state']),
                ("Pincode",  data['pincode'])]
        if ocr_raw:
            rows.append(("OCR Read",
                         ocr_raw[-12:] if len(ocr_raw) > 12 else ocr_raw))
        if ocr_votes and data:
            qr_last4  = data['last4']
            confirmed = ocr_votes.get(qr_last4, 0)
            rows.append(("Confirms",
                         f"XXXX-XXXX-{qr_last4} ({confirmed}/{OCR_VOTES_NEEDED})"))
        yo = py+52
        for lbl, val in rows:
            cv2.putText(frame, f"{lbl}:", (px, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)
            vd = (str(val)[:27]+"..") if len(str(val)) > 29 else str(val)
            cv2.putText(frame, vd, (px+78, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)
            yo += 27

    if verdict:
        vc  = C_GREEN if verdict=="REAL" else C_RED
        vtx = ("AADHAAR VERIFIED - GENUINE CARD"
               if verdict=="REAL" else
               "AADHAAR REJECTED - FAKE / TAMPERED")
        cv2.rectangle(frame, (0, h-58), (w,h), vc, -1)
        cv2.putText(frame, vtx, (w//2-230, h-18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_BLACK, 2)


def draw_yolo_boxes(frame, results):
    boxes_info = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf < YOLO_CONF:
                continue
            cls = (CLASS_NAMES[cls_id]
                   if cls_id < len(CLASS_NAMES) else f"cls{cls_id}")
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            hi    = cls in ("AadharEmblem","AadharNumber")
            color = C_GREEN if hi else C_BLUE
            thick = 3 if hi else 1
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)
            lbl = f"{cls} {conf:.2f}"
            cv2.rectangle(frame,(x1,y1-18),(x1+len(lbl)*8,y1),color,-1)
            cv2.putText(frame,lbl,(x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.43,C_BLACK,1)
            boxes_info.append(dict(cls=cls,conf=conf,bbox=(x1,y1,x2,y2)))
    return boxes_info


# ──────────────────────────────────────────────
# MAIN VERIFIER
# ──────────────────────────────────────────────
class AadhaarVerifier:

    def __init__(self, model_path, source):
        print("\n" + "="*55)
        print("  AADHAAR VERIFICATION SYSTEM  v3.6")
        print("="*55)

        print("[1/4] YOLO model...")
        self.yolo = YOLO(model_path)
        print(f"      ✅ {model_path}")

        print("[2/4] EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("      ✅ ready")

        print("[3/4] Voice...")
        self.voice = Voice()
        print("      ✅ ready")

        print("[4/4] Camera...")
        if source == "droidcam":
            self.cap = cv2.VideoCapture(DROIDCAM_URL)
        elif source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed!")

        # Minimize internal buffer — always read the freshest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("      ✅ ready\n")

        self.qr_scanner = QRScanner()
        self._reset()

    def _reset(self):
        self.state       = "QR_SCAN"
        self.qr_data     = None
        self.frames_done = 0
        self.verdict     = None
        self.result_time = None
        self.ocr_raw     = ""
        self.ocr_votes   = {}
        self.last_rect   = None   # last known QR bounding rect

    def _grab_fresh_frame(self):
        """
        Drain the camera buffer by grabbing without decoding,
        then retrieve only the latest frame.
        """
        for _ in range(4):
            self.cap.grab()
        ret, frame = self.cap.retrieve()
        return ret, frame

    # ── STATE 1: QR SCAN ──────────────────────
    def _state_qr(self, frame):
        h, w = frame.shape[:2]

        sy = int((math.sin(time.time()*2.5)+1)/2*h)
        cv2.line(frame, (0,sy), (w,sy), C_GREEN, 2)
        cv2.putText(frame, "STEP 1 - SCAN AADHAAR QR CODE",
                    (10,32), cv2.FONT_HERSHEY_DUPLEX, 0.68, C_YELLOW, 2)
        cv2.putText(frame, "Point camera at QR code on Aadhaar card",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1)

        # Feed frame to background QR thread
        self.qr_scanner.feed(frame)

        # Draw last known rect while waiting
        if self.last_rect:
            x,y,rw,rh = self.last_rect
            cv2.rectangle(frame,(x,y),(x+rw,y+rh),C_YELLOW,2)

        # Check if background thread has a result
        result = self.qr_scanner.get_result()
        if result:
            data, rect = result
            self.last_rect = rect
            if rect:
                x,y,rw,rh = rect
                cv2.rectangle(frame,(x,y),(x+rw,y+rh),C_GREEN,3)
                cv2.putText(frame,"QR DETECTED!",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,C_GREEN,2)
            else:
                cx,cy = w//2, h//2
                cv2.rectangle(frame,(cx-80,cy-80),(cx+80,cy+80),C_GREEN,3)
                cv2.putText(frame,"QR DETECTED!",(cx-80,cy-90),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,C_GREEN,2)

            if data:
                self.qr_data = data
                self.state   = "YOLO_SCAN"
                print(f"\n[QR] Name={data['name']}  Last4={data['last4']}  "
                      f"DOB={data['dob']}  Gender={data['gender']}")
                self.voice.speak(
                    "Q R code scanned. Verifying Aadhaar card now.")

    # ── STATE 2: YOLO + OCR ───────────────────
    def _state_yolo(self, frame):
        h, w = frame.shape[:2]

        results    = self.yolo(frame, verbose=False, conf=YOLO_CONF)
        boxes_info = draw_yolo_boxes(frame, results)

        if self.frames_done % 50 == 0:
            if boxes_info:
                detected = [f"{b['cls']}({b['conf']:.2f})"
                            for b in boxes_info]
                print(f"  [f{self.frames_done:03d}] YOLO: {', '.join(detected)}")
            else:
                print(f"  [f{self.frames_done:03d}] YOLO: nothing detected")

        for b in boxes_info:
            if b['cls'] == "AadharNumber":
                matched, ocr_digits, ocr_last4 = ocr_check_last4(
                    self.reader, frame, b['bbox'], self.qr_data['last4'])
                self.ocr_raw = ocr_digits

                if matched and ocr_last4:
                    if ocr_last4 == self.qr_data['last4']:
                        self.ocr_votes[ocr_last4] = \
                            self.ocr_votes.get(ocr_last4, 0) + 1
                        count = self.ocr_votes[ocr_last4]
                        print(f"  [f{self.frames_done:03d}] "
                              f"OCR={ocr_digits}  last4={ocr_last4}  "
                              f"votes={count}/{OCR_VOTES_NEEDED}  MATCH")
                        if count >= OCR_VOTES_NEEDED:
                            print(f"  [f{self.frames_done:03d}] CONFIRMED")
                            self._set_verdict("REAL")
                            return
                    else:
                        print(f"  [f{self.frames_done:03d}] "
                              f"OCR={ocr_digits}  last4={ocr_last4}  "
                              f"MISMATCH vs QR={self.qr_data['last4']}"
                              f" — ignored")
                else:
                    print(f"  [f{self.frames_done:03d}] "
                          f"OCR={ocr_digits}  need 12 digits")
                break

        self.frames_done += 1

        cv2.putText(frame, "STEP 2 - YOLO + OCR VERIFICATION",
                    (10,32), cv2.FONT_HERSHEY_DUPLEX, 0.65, C_YELLOW, 2)
        cv2.putText(frame,
                    f"Scanning: {self.frames_done}/{FRAME_WINDOW} frames",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1)

        if self.ocr_raw:
            cv2.putText(frame, f"OCR: {self.ocr_raw}",
                        (10, h-130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)

        qr_last4  = self.qr_data['last4']
        confirmed = self.ocr_votes.get(qr_last4, 0)
        cv2.putText(frame,
                    f"Confirmed: XXXX-XXXX-{qr_last4}"
                    f" ({confirmed}/{OCR_VOTES_NEEDED})",
                    (10, h-150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_YELLOW, 1)

        draw_progress(frame, self.frames_done, FRAME_WINDOW,
                      10, h-100, w-350, 22,
                      f"Frame {self.frames_done}/{FRAME_WINDOW}"
                      " - keep card steady")
        draw_panel(frame, self.qr_data,
                   ocr_raw=self.ocr_raw, ocr_votes=self.ocr_votes)

        if self.frames_done >= FRAME_WINDOW:
            self._set_verdict("FAKE")

    # ── VERDICT ───────────────────────────────
    def _set_verdict(self, verdict):
        self.verdict     = verdict
        self.state       = "RESULT"
        self.result_time = time.time()

        if verdict == "REAL":
            d = self.qr_data
            print(f"\n{'='*50}")
            print(f"  REAL - confirmed at frame {self.frames_done}")
            print(f"{'='*50}")
            for k,v in [("Name",    d['name']),  ("Aadhaar", d['display']),
                        ("DOB",     d['dob']),    ("Gender",  d['gender']),
                        ("Care Of", d['careof']), ("Street",  d['street']),
                        ("District",d['district']),("State",  d['state']),
                        ("Pincode", d['pincode'])]:
                print(f"  {k:<9}: {v}")
            print(f"{'='*50}\n")

            parts = ["Aadhaar card verified successfully."]
            parts.append(f"Name: {d['name']}.")
            parts.append(f"Date of birth: {d['dob']}.")
            parts.append(f"Gender: {d['gender']}.")
            if d.get('careof',   'N/A') != 'N/A':
                parts.append(f"Care of: {d['careof']}.")
            if d.get('street',   'N/A') != 'N/A':
                parts.append(f"Street: {d['street']}.")
            if d.get('district', 'N/A') != 'N/A':
                parts.append(f"District: {d['district']}.")
            if d.get('state',    'N/A') != 'N/A':
                parts.append(f"State: {d['state']}.")
            if d.get('pincode',  'N/A') != 'N/A':
                parts.append(f"Pincode: {d['pincode']}.")
            parts.append("This is a genuine Aadhaar card.")
            self.voice.speak(" ".join(parts))
        else:
            print(f"\n{'='*50}")
            print(f"  FAKE - no confirmed match in {FRAME_WINDOW} frames")
            print(f"{'='*50}\n")
            self.voice.speak(
                "Warning! Aadhaar verification failed. "
                "Could not confirm a valid Aadhaar number on the card. "
                "This card may be fake or tampered.")

    # ── STATE 3: RESULT ───────────────────────
    def _state_result(self, frame):
        h, w    = frame.shape[:2]
        elapsed = time.time() - self.result_time

        draw_panel(frame, self.qr_data, verdict=self.verdict,
                   ocr_raw=self.ocr_raw, ocr_votes=self.ocr_votes)

        if elapsed < 2.0 and int(elapsed*4) % 2 == 0:
            ov    = frame.copy()
            color = (0,60,0) if self.verdict=="REAL" else (0,0,60)
            cv2.rectangle(ov,(0,0),(w,h),color,-1)
            cv2.addWeighted(ov,0.3,frame,0.7,0,frame)

        cv2.putText(frame, "R - scan next  |  Q - quit",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)

    # ── MAIN LOOP ─────────────────────────────
    def run(self):
        print("Running!  Q=quit  R=reset\n")
        while True:
            # Always grab the freshest frame — drain stale buffer
            ret, frame = self._grab_fresh_frame()
            if not ret:
                print("[ERROR] Frame read failed")
                break

            if   self.state == "QR_SCAN":   self._state_qr(frame)
            elif self.state == "YOLO_SCAN":  self._state_yolo(frame)
            elif self.state == "RESULT":     self._state_result(frame)

            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                        (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1)

            cv2.imshow("Aadhaar Verification System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset()
                print("\n[RESET] Ready for next person...\n")

        self.qr_scanner.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Closed.")


# ──────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="runs/detect/aadhar/weights/best.pt")
    ap.add_argument("--source", default="droidcam",
                    help="droidcam | 0 | video.mp4")
    args = ap.parse_args()
    try:
        AadhaarVerifier(args.model, args.source).run()
    except Exception as e:
        print(f"\n[FATAL] {e}")
