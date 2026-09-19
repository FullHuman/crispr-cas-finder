#!/usr/bin/env python3
"""Dev server with COOP/COEP headers required for SharedArrayBuffer (WASM threads)."""
import http.server
import os
import sys

# Serve from the directory where this script lives (www/)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        # Safari supports require-corp, but not the credentialless policy.
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
print(f"Serving on http://localhost:{port} with COOP/COEP headers")
http.server.HTTPServer(("", port), COOPCOEPHandler).serve_forever()
