#!/usr/bin/env python3
"""Lightweight mock of Ollama HTTP API for local testing.

Endpoints implemented:
- GET  /api/tags      -> returns available models
- POST /api/generate  -> accepts payload {model, prompt, options...} and returns {response}

This lets us test prompt->API->response flows without starting the real Ollama server.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

HOST = '127.0.0.1'
PORT = 11434

class MockHandler(BaseHTTPRequestHandler):
    def _set_json(self, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/api/tags') or self.path.startswith('/api/tags/'): 
            models = [{'name': 'llama3.2:latest'}, {'name': 'mistral:7b-instruct'}]
            self._set_json(200)
            self.wfile.write(json.dumps({'models': models}).encode('utf-8'))
            return
        # default 404
        self.send_error(404, 'Not Found')

    def do_POST(self):
        if self.path == '/api/generate':
            length = int(self.headers.get('content-length', '0'))
            body = self.rfile.read(length).decode('utf-8')
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}
            model = payload.get('model', 'unknown')
            prompt = payload.get('prompt', '') or ''

            # Simple heuristics to mimic refusal vs answer behavior
            # If the prompt explicitly contains contexts but they are empty -> refuse
            if 'Instruction: Answer using ONLY the provided Contexts' in prompt:
                # Check if contexts block after 'Contexts:' appears empty
                if 'Contexts:\n\n' in prompt or prompt.split('Contexts:\n')[-1].strip().startswith('\n'):
                    answer = "Insufficient evidence to answer."
                else:
                    # if there are numbered contexts like [1], include a short mock answer citing [1]
                    if '[1]' in prompt or '[1]' in prompt.split('Contexts:')[-1]:
                        # Try to extract the first numbered passage content to produce a context-specific mock answer
                        try:
                            import re
                            contexts_block = prompt.split('Contexts:\n', 1)[1]
                            m = re.search(r'\[1\]\s*(.+?)(?:\n\[\d+\]|\n\n|$)', contexts_block, re.S)
                            if m:
                                first_passage = m.group(1).strip().replace('\n', ' ')[:200]
                            else:
                                first_passage = contexts_block.strip().split('\n\n')[0].replace('\n', ' ')[:200]
                            answer = f"Based on the provided context [1]: \"{first_passage}\""
                        except Exception:
                            answer = "Based on the provided context [1], the patient shows signs consistent with the described symptom."
                    else:
                        # generic echo answer
                        snippet = prompt.replace('\n', ' ')[:200]
                        answer = f"MOCK ({model}) answer: {snippet}"
            else:
                # no explicit contexts instruction; if prompt mentions no retrieved contexts -> refuse
                if 'You have no retrieved contexts' in prompt:
                    answer = "Insufficient evidence to answer."
                else:
                    snippet = prompt.replace('\n', ' ')[:200]
                    answer = f"MOCK ({model}) answer: {snippet}"

            resp = {'response': answer}
            self._set_json(200)
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode('utf-8'))
            return

        # unknown path
        self.send_error(404, 'Not Found')

    def log_message(self, format, *args):
        # keep logs concise
        print("[mock-ollama] %s - - %s" % (self.address_string(), format%args))

if __name__ == '__main__':
    print(f"Starting mock Ollama server at http://{HOST}:{PORT}")
    server = HTTPServer((HOST, PORT), MockHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down mock server')
        server.server_close()
