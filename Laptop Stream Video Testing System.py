import argparse
import socket
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


class CameraStream:
    def __init__(self, source, width, height, fps, quality):
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.cap = None

    def open(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps and self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def frames(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()

        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                self.open()
                continue

            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            yield frame

    def jpeg_frames(self):
        for frame in self.frames():

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
            )
            if not ok:
                continue

            yield buffer.tobytes()


def serve_raw_stream(camera, host, port):
    header_fmt = "!4sQIIII"
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    print(f"raw stream url: raw://<orange-pi-ip>:{port}")
    print("raw stream sends uncompressed BGR frames")

    while True:
        conn, addr = server.accept()
        print(f"client connected: {addr[0]}:{addr[1]}")
        try:
            with conn:
                for frame in camera.frames():
                    frame = np.ascontiguousarray(frame)
                    height, width, channels = frame.shape
                    payload = frame.tobytes()
                    header = struct.pack(
                        header_fmt,
                        b"RAW0",
                        time.time_ns(),
                        width,
                        height,
                        channels,
                        len(payload),
                    )
                    conn.sendall(header)
                    conn.sendall(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("client disconnected")


def make_handler(camera):
    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/health"):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"orange pi camera stream ok\n")
                return

            if self.path != "/video_feed":
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()

            try:
                for jpg in camera.jpeg_frames():
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

    return StreamHandler


def main():
    parser = argparse.ArgumentParser(description="MJPEG camera stream for Orange Pi")
    parser.add_argument("--mode", default="raw", choices=["raw", "mjpeg"], help="raw为未压缩传输，mjpeg为JPG压缩传输")
    parser.add_argument("--camera", default="0", help="Camera index or video path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=0, help="0表示不限制帧率，能传多少传多少")
    parser.add_argument("--quality", type=int, default=70)
    args = parser.parse_args()

    camera = CameraStream(args.camera, args.width, args.height, args.fps, args.quality)
    if not camera.open():
        print(f"camera open failed: {args.camera}")
    else:
        print(f"camera opened: {args.camera}")

    if args.mode == "raw":
        serve_raw_stream(camera, args.host, args.port)
    else:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(camera))
        print(f"stream url: http://<orange-pi-ip>:{args.port}/video_feed")
        server.serve_forever()


if __name__ == "__main__":
    main()
