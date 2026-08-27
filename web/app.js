/*
 * Skein DAG viewer.
 *
 * Fetches the graph once, then applies status updates from the WebSocket. The
 * split matters: re-fetching and re-laying out on every event would make the
 * nodes jump around, and a workflow emits an event every few milliseconds
 * during a fan-out.
 */
'use strict';

const el = (id) => document.getElementById(id);

const NODE_W = 150;
const NODE_H = 44;
const GAP_X = 70;
const GAP_Y = 22;

let socket = null;
let graph = null;
const statusByStep = new Map();

function setStatus(text, kind) {
  const node = el('status');
  node.textContent = text;
  node.className = 'status' + (kind ? ' ' + kind : '');
}

function show(id, visible) {
  el(id).classList.toggle('hidden', !visible);
}

/* Longest-path depth: a node sits one level right of its deepest dependency.
 * Using the *longest* path rather than the shortest is what makes every edge
 * point strictly rightwards, so no arrow doubles back. */
function assignDepths(nodes, edges) {
  const incoming = new Map(nodes.map((n) => [n.id, []]));
  edges.forEach((e) => {
    if (incoming.has(e.target)) incoming.get(e.target).push(e.source);
  });

  const depth = new Map();
  const visiting = new Set();

  function resolve(id) {
    if (depth.has(id)) return depth.get(id);
    // Guard against a cycle. The server rejects cyclic workflows, so this can
    // only fire on malformed input, and returning 0 keeps the page usable
    // rather than hanging the tab.
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const parents = incoming.get(id) || [];
    const value = parents.length === 0 ? 0 : Math.max(...parents.map(resolve)) + 1;
    visiting.delete(id);
    depth.set(id, value);
    return value;
  }

  nodes.forEach((n) => resolve(n.id));
  return depth;
}

function layout(nodes, edges) {
  const depth = assignDepths(nodes, edges);
  const columns = new Map();
  nodes.forEach((n) => {
    const d = depth.get(n.id) || 0;
    if (!columns.has(d)) columns.set(d, []);
    columns.get(d).push(n);
  });

  const positions = new Map();
  columns.forEach((column, d) => {
    column.forEach((node, index) => {
      positions.set(node.id, {
        x: 20 + d * (NODE_W + GAP_X),
        y: 20 + index * (NODE_H + GAP_Y),
      });
    });
  });

  const width = 40 + (Math.max(...columns.keys()) + 1) * (NODE_W + GAP_X);
  const height =
    40 + Math.max(...[...columns.values()].map((c) => c.length)) * (NODE_H + GAP_Y);
  return { positions, width, height };
}

function draw() {
  if (!graph) return;
  const svg = el('graph');
  const { positions, width, height } = layout(graph.nodes, graph.edges);

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.replaceChildren();

  const ns = 'http://www.w3.org/2000/svg';

  const defs = document.createElementNS(ns, 'defs');
  defs.innerHTML =
    '<marker id="arrow" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" ' +
    'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker>';
  svg.appendChild(defs);

  graph.edges.forEach((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) return;
    const path = document.createElementNS(ns, 'path');
    const x1 = from.x + NODE_W;
    const y1 = from.y + NODE_H / 2;
    const x2 = to.x;
    const y2 = to.y + NODE_H / 2;
    const mid = (x1 + x2) / 2;
    path.setAttribute('d', `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`);
    path.setAttribute('class', 'edge' + (edge.kind === 'branch' ? ' branch' : ''));
    path.setAttribute('marker-end', 'url(#arrow)');
    svg.appendChild(path);
  });

  graph.nodes.forEach((node) => {
    const pos = positions.get(node.id);
    const status = statusByStep.get(node.id) || node.status || 'pending';

    const group = document.createElementNS(ns, 'g');
    group.setAttribute('transform', `translate(${pos.x},${pos.y})`);
    group.setAttribute('class', `node ${status}`);

    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', NODE_W);
    rect.setAttribute('height', NODE_H);
    rect.setAttribute('rx', 6);
    group.appendChild(rect);

    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', 10);
    label.setAttribute('y', 18);
    label.setAttribute('class', 'node-id');
    label.textContent = node.id;
    group.appendChild(label);

    const sub = document.createElementNS(ns, 'text');
    sub.setAttribute('x', 10);
    sub.setAttribute('y', 33);
    sub.setAttribute('class', 'node-sub');
    sub.textContent = node.target ? `${node.kind} · ${node.target}` : node.kind;
    group.appendChild(sub);

    const title = document.createElementNS(ns, 'title');
    title.textContent = `${node.id} — ${status}`;
    group.appendChild(title);

    svg.appendChild(group);
  });
}

function appendEvent(event) {
  const line = document.createElement('div');
  line.className = 'event ' + (event.kind || '');
  const parts = [event.kind];
  if (event.step_id) parts.push(event.step_id);
  if (event.attempt && event.attempt > 1) parts.push(`attempt ${event.attempt}`);
  if (event.duration_s != null) parts.push(`${event.duration_s.toFixed(3)}s`);
  if (event.error_code) parts.push(event.error_code);
  line.textContent = parts.join('  ·  ');
  const container = el('events');
  container.prepend(line);
  // Cap the DOM. A fan-out over hundreds of items emits thousands of events,
  // and an unbounded log node list makes the tab unresponsive.
  while (container.childElementCount > 300) container.lastElementChild.remove();
}

const TERMINAL_KINDS = {
  step_succeeded: 'succeeded',
  step_failed: 'failed',
  step_cancelled: 'cancelled',
  step_skipped: 'skipped',
};

function applyEvent(event) {
  if (event.kind === 'keepalive') return;

  if (event.kind === 'step_started' && event.step_id) {
    statusByStep.set(event.step_id, 'running');
  } else if (TERMINAL_KINDS[event.kind] && event.step_id) {
    statusByStep.set(event.step_id, TERMINAL_KINDS[event.kind]);
  }

  if (event.kind === 'run_finished') {
    const status = (event.payload && event.payload.status) || 'finished';
    el('run-status').textContent = status;
    el('run-status').className = 'pill ' + status;
    el('cancel-btn').disabled = true;
  }

  if (event.kind === 'stream_end') {
    setStatus(`stream closed (${event.status})`, 'ok');
    el('cancel-btn').disabled = true;
    return;
  }

  appendEvent(event);
  draw();
}

async function watch(runId) {
  if (socket) {
    socket.close();
    socket = null;
  }
  statusByStep.clear();
  el('events').replaceChildren();

  setStatus('loading graph…');
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/graph`);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const problem = await response.json();
      if (problem.detail) message = `${problem.code}: ${problem.detail}`;
    } catch (ignored) { /* not a problem document */ }
    setStatus(message, 'error');
    return;
  }

  graph = await response.json();
  graph.nodes.forEach((n) => statusByStep.set(n.id, n.status));
  el('run-name').textContent = graph.workflow;
  show('summary', true);
  show('graph-card', true);
  show('events-card', true);
  el('cancel-btn').disabled = false;
  draw();

  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(
    `${scheme}://${window.location.host}/runs/${encodeURIComponent(runId)}/stream`
  );

  socket.onopen = () => setStatus('streaming', 'ok');
  socket.onmessage = (message) => {
    try {
      applyEvent(JSON.parse(message.data));
    } catch (error) {
      // A malformed frame should not take the whole view down.
      console.warn('bad frame', error);
    }
  };
  socket.onerror = () => setStatus('websocket error', 'error');
  socket.onclose = (event) => {
    if (event.code === 4404) setStatus('run not found', 'error');
    el('cancel-btn').disabled = true;
  };

  window.location.hash = runId;
}

el('lookup').addEventListener('submit', (event) => {
  event.preventDefault();
  const runId = el('run-id').value.trim();
  if (runId) watch(runId);
});

el('cancel-btn').addEventListener('click', async () => {
  const runId = el('run-id').value.trim();
  if (!runId) return;
  el('cancel-btn').disabled = true;
  setStatus('cancelling…');
  const response = await fetch(`/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
  setStatus(response.ok ? 'cancel requested' : 'cancel failed', response.ok ? 'ok' : 'error');
});

const fromHash = window.location.hash.replace(/^#/, '').trim();
if (fromHash) {
  el('run-id').value = fromHash;
  watch(fromHash);
}
