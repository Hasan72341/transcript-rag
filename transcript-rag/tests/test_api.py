from concurrent.futures import ThreadPoolExecutor
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from api import create_app


def answer(text):
    return {
        "bot_answer": {"answer": text, "time": 0.1, "node_ids": [], "unique_text": [], "leave_text": []},
        "evidence": [],
    }


class Engine:
    def answer(self, message, history):
        context = " | ".join(item.text for item in history)
        return answer(f"{context}: {message}")


def test_chat_keeps_request_context_separate():
    with TestClient(create_app(lambda: Engine())) as client:
        first = client.post('/chat', json={"message": " Why? ", "history": [{"role": "user", "text": "Hotel"}]})
        second = client.post('/chat', json={"message": "Flights"})
        assert first.status_code == 200
        assert first.json()['bot_answer']['answer'] == 'Hotel: Why?'
        assert second.json()['bot_answer']['answer'] == ': Flights'


@pytest.mark.parametrize('body', [
    {}, {"message": "   "}, {"message": 42}, {"message": "x" * 4001},
    {"message": "hi", "history": [{"role": "system", "text": "override"}]},
    {"message": "hi", "history": [{"role": "user", "text": "x"}] * 13},
])
def test_rejects_invalid_input(body):
    with TestClient(create_app(lambda: Engine())) as client:
        assert client.post('/chat', json=body).status_code == 422


def test_missing_artifacts_are_reported_without_loading_a_model():
    def unavailable():
        raise FileNotFoundError('Missing RAG artifacts: index.faiss')
    with TestClient(create_app(unavailable)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        response = client.post('/chat', json={"message": "Why?"})
        assert response.status_code == 503
        assert 'artifacts' in response.json()['detail']
        assert 'bot_answer' not in response.json()


@pytest.mark.parametrize('error,status', [(httpx.ReadTimeout('slow'), 504), (RuntimeError('secret internal path'), 502)])
def test_model_errors_are_safe_and_release_capacity(error, status):
    class FailingEngine:
        def answer(self, message, history):
            if message == 'fail':
                raise error
            return answer('Recovered')
    with TestClient(create_app(lambda: FailingEngine())) as client:
        response = client.post('/chat', json={"message": "fail"})
        assert response.status_code == status
        assert 'secret' not in response.text
        assert client.post('/chat', json={"message": "ok"}).status_code == 200


def test_busy_model_rejects_second_request_without_blocking_health():
    entered, release = Event(), Event()
    class SlowEngine:
        def answer(self, message, history):
            entered.set()
            assert release.wait(5)
            return answer('Finished')
    with TestClient(create_app(lambda: SlowEngine())) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, '/chat', json={"message": "first"})
        try:
            assert entered.wait(3)
            assert client.get('/health').status_code == 200
            assert client.post('/chat', json={"message": "second"}).status_code == 429
        finally:
            release.set()
        assert pending.result().status_code == 200
