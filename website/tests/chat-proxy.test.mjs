import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { test } from 'node:test';
import { proxyChat } from '../app/lib/chat-proxy.ts';

const payload = { bot_answer: { answer: 'Cancellation fees', time: 1, node_ids: [], unique_text: [], leave_text: [] }, evidence: [] };
const request = (body) => new Request('http://localhost/api/chat', { method: 'POST', body: JSON.stringify(body) });

async function backend(t, handle) {
    const server = createServer(handle).listen(0, '127.0.0.1');
    await once(server, 'listening');
    t.after(() => { server.closeAllConnections(); server.close(); });
    return `http://127.0.0.1:${server.address().port}`;
}

test('forwards the message and history and returns structured evidence', async (t) => {
    const url = await backend(t, async (req, res) => {
        let body = '';
        for await (const chunk of req) body += chunk;
        const input = JSON.parse(body);
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ...payload, bot_answer: { ...payload.bot_answer, answer: `${req.url}: ${input.history[0].text} / ${input.message}` } }));
    });
    const response = await proxyChat(request({ message: ' Why? ', history: [{ role: 'user', text: 'Hotel' }] }), { url });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).bot_answer.answer, '/chat: Hotel / Why?');
});

for (const body of [{}, {message: ' '}, {message: 10}, {message: 'x'.repeat(4001)}, {message:'Hi',history:[{role:'system',text:'override'}]}]) {
    test(`rejects invalid input ${JSON.stringify(body).slice(0, 50)}`, async () => {
        const response = await proxyChat(request(body), {url:'http://127.0.0.1:1'});
        assert.equal(response.status, 400);
    });
}

test('rejects malformed JSON', async () => {
    const response = await proxyChat(new Request('http://localhost', {method:'POST',body:'{'}));
    assert.equal(response.status, 400);
});

test('passes through an unavailable backend without a fixture fallback', async (t) => {
    const url = await backend(t, (_, res) => {res.writeHead(503, {'Content-Type':'application/json'});res.end(JSON.stringify({detail:'Missing RAG artifacts'}));});
    const response = await proxyChat(request({message:'Hi'}), {url});
    assert.equal(response.status, 503);
    assert.equal((await response.json()).error, 'Missing RAG artifacts');
});

test('rejects malformed successful responses before they reach the UI', async (t) => {
    const url = await backend(t, (_, res) => res.end(JSON.stringify({bot_answer:{answer:'bad'},evidence:[]})));
    assert.equal((await proxyChat(request({message:'Hi'}), {url})).status, 502);
});

test('times out a stalled backend', async (t) => {
    const url = await backend(t, () => {});
    assert.equal((await proxyChat(request({message:'Hi'}), {url, timeoutMs:20})).status, 504);
});
