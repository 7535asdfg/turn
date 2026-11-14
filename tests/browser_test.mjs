/*
 * 端到端浏览器测试（无头 Edge + CDP）
 *   1. 启动本地 HTTP 服务
 *   2. 打开 index.html，切到“多角度重建”
 *   3. 通过 DOM.setFileInputFiles 选中 tests/synthetic/view_*.png
 *   4. 点击生成，等待 SfM + MVS 完成
 *   5. 截图并输出控制台错误
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const HTTP_PORT = 8123;
const CDP_PORT = 9223;
const SYN = path.join(__dirname, 'synthetic');

const SINGLE_ONLY = process.argv.includes('--single');
const files = fs.readdirSync(SYN)
  .filter((f) => /^view_\d+\.png$/i.test(f))
  .sort()
  .map((f) => path.join(SYN, f));
if (files.length < 2) {
  console.error('缺少合成测试图片，请先运行 python -I tests/gen_synthetic.py');
  process.exit(1);
}
console.log('测试图片数量:', files.length);

const server = spawn('python', ['-m', 'http.server', String(HTTP_PORT), '--bind', '127.0.0.1'], {
  cwd: ROOT, stdio: 'ignore',
});

const userDir = path.join(process.env.TEMP || ROOT, 'p2m-edge-' + Date.now());
const edge = spawn(EDGE, [
  '--headless=new',
  '--remote-debugging-port=' + CDP_PORT,
  '--user-data-dir=' + userDir,
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-extensions',
  '--enable-unsafe-swiftshader',
  '--window-size=1400,900',
  'about:blank',
], { stdio: ['ignore', 'pipe', 'pipe'] });

let edgeErr = '';
edge.stderr.on('data', (d) => { edgeErr += d.toString(); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let s = '';
      res.on('data', (d) => (s += d));
      res.on('end', () => {
        try { resolve(JSON.parse(s)); } catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

async function waitForCDP() {
  for (let i = 0; i < 80; i++) {
    try {
      const v = await getJson(`http://127.0.0.1:${CDP_PORT}/json/version`);
      if (v.webSocketDebuggerUrl) return v;
    } catch { /* retry */ }
    await sleep(250);
  }
  throw new Error('无法连接 Edge CDP');
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      } else if (msg.method) {
        this.events.push(msg);
        if (this.onEvent) this.onEvent(msg);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
}

async function evaluate(cdp, expression) {
  const r = await cdp.send('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true,
  });
  if (r.exceptionDetails) {
    throw new Error('页面脚本异常: ' + (r.exceptionDetails.text || JSON.stringify(r.exceptionDetails)));
  }
  return r.result.value;
}

async function main() {
  const version = await waitForCDP();
  console.log('CDP 就绪:', version.Browser);
  const list = await getJson(`http://127.0.0.1:${CDP_PORT}/json/list`);
  const page = list.find((t) => t.type === 'page') || list[0];
  if (!page || !page.webSocketDebuggerUrl) throw new Error('找不到页面 target');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve);
    ws.addEventListener('error', reject);
  });
  const cdp = new CDP(ws);
  cdp.onEvent = (msg) => {
    if (msg.method === 'Runtime.consoleAPICalled') {
      const args = (msg.params.args || []).map((a) => a.value ?? a.description ?? '');
      console.log('[console.' + msg.params.type + ']', args.join(' '));
    } else if (msg.method === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      console.error('[page exception]', d.text, d.exception?.description || '');
    } else if (msg.method === 'Log.entryAdded') {
      console.log('[log]', msg.params.entry.level, msg.params.entry.text);
    }
  };
  await cdp.send('Runtime.enable');
  await cdp.send('Page.enable');
  await cdp.send('DOM.enable');
  await cdp.send('Log.enable');

  const prevOnEvent = cdp.onEvent;
  let loadedResolve;
  const loaded = new Promise((r) => { loadedResolve = r; });
  cdp.onEvent = (msg) => {
    prevOnEvent(msg);
    if (msg.method === 'Page.loadEventFired') loadedResolve();
  };
  await cdp.send('Page.navigate', { url: `http://127.0.0.1:${HTTP_PORT}/index.html` });
  await loaded;
  cdp.onEvent = prevOnEvent;
  await sleep(3000); // 等待 CDN 模块加载

  const ready = await evaluate(cdp, `({
    hasTabs: document.querySelectorAll('.mode-tab').length,
    hasModule: !!document.getElementById('multiThumbs'),
    mode: document.querySelector('.mode-tab.active')?.dataset.mode,
    status: document.getElementById('statusText').textContent
  })`);
  console.log('页面状态:', ready);
  if (ready.hasTabs !== 2) throw new Error('页面模块未加载（可能 CDN 加载失败）');

  const doc = await cdp.send('DOM.getDocument', { depth: -1 });

  if (SINGLE_ONLY) {
    const input = await cdp.send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '#fileInput' });
    await cdp.send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [files[0]] });
    console.log('单张模式：已提交 1 个文件，等待深度管线启动…');
    await sleep(8000);
    const st = await evaluate(cdp, `({
      status: document.getElementById('statusText').textContent,
      mode: document.querySelector('.mode-tab.active')?.dataset.mode,
      fileMeta: !document.getElementById('fileMeta').classList.contains('hidden'),
      depthPreview: !document.getElementById('depthPreviewWrap').classList.contains('hidden'),
      generateText: document.getElementById('btnGenerate').textContent
    })`);
    console.log('单张模式状态:', st);
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(path.join(SYN, 'browser_single.png'), Buffer.from(shot.data, 'base64'));
    if (/失败/.test(st.status)) throw new Error('单张模式启动失败: ' + st.status);
    console.log('单张模式启动检查通过');
    return;
  }

  // 切到多角度模式
  await evaluate(cdp, `document.querySelector('.mode-tab[data-mode="multi"]').click()`);
  await sleep(300);

  const q = await cdp.send('DOM.querySelector', {
    nodeId: doc.root.nodeId,
    selector: '#multiFileInput',
  });
  if (!q.nodeId) throw new Error('找不到多图输入框');
  await cdp.send('DOM.setFileInputFiles', { nodeId: q.nodeId, files });
  console.log('已提交', files.length, '个文件，等待读取…');

  for (let i = 0; i < 120; i++) {
    const count = await evaluate(cdp, `document.getElementById('multiCount').textContent`);
    if (Number(count) >= files.length) break;
    await sleep(250);
  }
  const count = await evaluate(cdp, `document.getElementById('multiCount').textContent`);
  console.log('已读取图片:', count);
  if (Number(count) < 2) throw new Error('图片读取失败');

  await evaluate(cdp, `document.getElementById('btnGenerate').click()`);
  console.log('开始 SfM + MVS…');

  let status = '';
  const t0 = Date.now();
  for (let i = 0; i < 600; i++) {
    status = await evaluate(cdp, `document.getElementById('statusText').textContent`);
    if (i % 8 === 0) console.log(`  ${((Date.now() - t0) / 1000).toFixed(0)}s ${status}`);
    if (/^完成|^重建失败|^已取消/.test(status)) break;
    await sleep(500);
  }
  const result = await evaluate(cdp, `({
    status: document.getElementById('statusText').textContent,
    cams: document.getElementById('statCams').textContent,
    sparse: document.getElementById('statSparse').textContent,
    verts: document.getElementById('statVerts').textContent,
    tris: document.getElementById('statTris').textContent,
    hud: {
      verts: document.getElementById('hudVerts').textContent,
      tris: document.getElementById('hudTris').textContent,
      fps: document.getElementById('hudFps').textContent
    },
    canvas: (() => { const c=document.getElementById('view'); return {w:c.width,h:c.height}; })(),
    emptyHidden: document.getElementById('empty').classList.contains('hide')
  })`);
  console.log('结果:', result);

  const shot = await cdp.send('Page.captureScreenshot', { format: 'png' });
  const outPng = path.join(SYN, 'browser_result.png');
  fs.writeFileSync(outPng, Buffer.from(shot.data, 'base64'));
  console.log('截图已保存', outPng);

  if (!/^完成/.test(result.status)) throw new Error('重建未成功');
  if (Number(result.cams) < 8) throw new Error('注册相机数量过少: ' + result.cams);
  console.log('浏览器端到端测试通过');
}

main()
  .catch((e) => {
    console.error('测试失败:', e);
    if (edgeErr) console.error('Edge stderr:', edgeErr.slice(0, 3000));
    process.exitCode = 1;
  })
  .finally(async () => {
    try { edge.kill(); } catch {}
    try { server.kill(); } catch {}
    await sleep(500);
    try { fs.rmSync(userDir, { recursive: true, force: true }); } catch {}
  });