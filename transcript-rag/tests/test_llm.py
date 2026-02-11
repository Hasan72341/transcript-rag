import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import httpx
import pytest

from llm import VLLMWrapper


@pytest.fixture
def model():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt = body['messages'][-1]['content']
            self.send_response(503 if prompt == 'fail' else 200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            content = None if prompt == 'empty' else f"{body['model']}: {prompt}"
            self.wfile.write(json.dumps({'choices': [{'message': {'content': content}}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield VLLMWrapper(server_url=f'http://127.0.0.1:{server.server_port}/v1', model_name='test-model')
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_sync_and_batch_calls_use_the_configured_model(model):
    assert model.invoke('Question') == 'test-model: Question'
    assert asyncio.run(model.batch_acall(['One', 'Two'])) == ['test-model: One', 'test-model: Two']


def test_empty_and_failed_model_responses_are_not_answers(model):
    with pytest.raises(ValueError, match='empty answer'):
        model.invoke('empty')
    with pytest.raises(httpx.HTTPStatusError):
        model.invoke('fail')
