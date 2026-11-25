"""
Local Web Server for Driver Drowsiness Mobile App
Run: python serve.py
"""

import http.server
import socket
import socketserver
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and SharedArrayBuffer headers for MediaPipe WASM acceleration
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("=" * 65)
    print("  EDGE AI DRIVER DROWSINESS MONITOR - MOBILE WEB APP SERVER")
    print("=" * 65)
    print(f" [PC Localhost] : http://localhost:{PORT}")
    print(f" [Dien thoai]   : http://{local_ip}:{PORT}")
    print("=" * 65)
    print(" Huong dan mo tren dien thoai:")
    print(f" 1. Ket noi dien thoai vao cung mang Wi-Fi voi PC.")
    print(f" 2. Mo trinh duyet Chrome/Safari tren dien thoai va truy cap: http://{local_ip}:{PORT}")
    print(" 3. Cho phep quyen Camera khi duoc hoi.")
    print(" 4. Bam nut 'Them vao man hinh chinh' (Add to Home screen) de cai thanh App!")
    print("=" * 65)
    print(" Nhan Ctrl+C de dung server.\n")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDa dung server.")
            sys.exit(0)
