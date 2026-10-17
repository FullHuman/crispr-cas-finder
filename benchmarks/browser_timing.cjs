// Time the browser build (WebAssembly) on one FASTA file, with the threaded
// worker and with a worker limited to one thread.
//
// Run after ./build_wasm.sh and `npm ci --prefix crispr-cas-finder-wasm/tests`.
//   node benchmarks/browser_timing.cjs [genome.fasta]
// BROWSER=chromium|webkit (default chromium), REPS (default 3).
// Prints one CSV row per run. The browser build uses the bundled CasFinder 2.0.3.
const path = require('node:path');
const fs = require('node:fs/promises');
const http = require('node:http');
const repo = path.resolve(__dirname, '..');
const playwright = require(path.join(repo, 'crispr-cas-finder-wasm/tests/node_modules/playwright'));
const root = path.join(repo, 'crispr-cas-finder-wasm/www');
const browserName = process.env.BROWSER || 'chromium';
const reps = Number(process.env.REPS || 3);

(async () => {
  const deployment = JSON.parse(await fs.readFile(path.join(root, 'vercel.json'), 'utf8'));
  const headers = Object.fromEntries(deployment.headers[0].headers.map(({key, value}) => [key, value]));
  const types = {'.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.css': 'text/css', '.html': 'text/html'};
  const server = http.createServer(async (req, res) => {
    for (const [name, value] of Object.entries(headers)) res.setHeader(name, value);
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname === '/sequential-worker.js') {
      res.setHeader('Content-Type', 'text/javascript');
      return res.end('import "./worker.js"; Object.defineProperty(navigator, "hardwareConcurrency", {value: 1});');
    }
    const filename = path.join(root, pathname === '/' ? 'index.html' : pathname);
    if (!filename.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
    try {
      const body = await fs.readFile(filename);
      res.setHeader('Content-Type', types[path.extname(filename)] || 'application/octet-stream');
      res.end(body);
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const fastaPath = process.argv[2] || path.join(repo, 'crispr-cas-finder-cli/data/ecoli.fasta');
  const fasta = await fs.readFile(fastaPath, 'utf8');
  const browser = await playwright[browserName].launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    console.log('browser,worker,cas,rep,seconds,arrays,cas_systems');
    for (const cas of [true, false]) {
      for (const workerPath of ['worker.js', 'sequential-worker.js']) {
        for (let rep = 1; rep <= reps; rep++) {
          // The clock covers worker start-up, WebAssembly instantiation and,
          // with Cas detection, loading the bundled models.
          const run = await page.evaluate(({fasta, workerPath, cas}) => new Promise((resolve, reject) => {
            const start = performance.now();
            const worker = new Worker(workerPath, {type: 'module'});
            worker.onerror = e => { worker.terminate(); reject(new Error(e.message)); };
            worker.onmessage = ({data}) => {
              if (data.progress) return;
              worker.terminate();
              if (data.error) return reject(new Error(data.error));
              const result = data.result;
              const arrays = Array.isArray(result) ? result : result.crisprs;
              resolve({seconds: (performance.now() - start) / 1000, arrays: arrays.length, cas: Array.isArray(result) ? 0 : result.cas_clusters.length});
            };
            worker.postMessage(cas
              ? {id: 1, type: 'run_full_analysis', payload: {fasta, options: {min_evidence_level: 1}, casOpts: {genetic_code: 11}}}
              : {id: 1, type: 'find_repeats', payload: {fasta, options: {min_evidence_level: 1}}});
          }), {fasta, workerPath, cas});
          console.log([browserName, workerPath === 'worker.js' ? 'threaded' : 'single', cas, rep, run.seconds.toFixed(2), run.arrays, run.cas].join(','));
        }
      }
    }
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
