// Run after ./build_wasm.sh. Requires Playwright and Chromium or WebKit.
// PLAYWRIGHT_MODULE may point to an existing Playwright installation.
const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const root = path.resolve(__dirname, '..', 'www');
const browserName = process.env.SMOKE_BROWSER || 'chromium';
assert.ok(['chromium', 'webkit'].includes(browserName), 'SMOKE_BROWSER must be chromium or webkit');

(async () => {
  // Exercise the deployment headers instead of masking configuration bugs with
  // a separate set of test-only values. Both Vercel entry points must agree.
  const deployment = JSON.parse(await fs.readFile(path.join(root, 'vercel.json'), 'utf8'));
  const rootDeployment = JSON.parse(await fs.readFile(path.resolve(root, '../../vercel.json'), 'utf8'));
  const headers = Object.fromEntries(deployment.headers[0].headers.map(({key, value}) => [key, value]));
  assert.deepEqual(Object.fromEntries(rootDeployment.headers[0].headers.map(({key, value}) => [key, value])), headers);
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
      res.setHeader('Content-Type', ({'.js':'text/javascript', '.wasm':'application/wasm', '.json':'application/json', '.css':'text/css', '.html':'text/html'})[path.extname(filename)] || 'application/octet-stream');
      res.end(body);
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await playwright[browserName].launch({ headless: true, executablePath: process.env.BROWSER_EXECUTABLE });
    const page = await browser.newPage();
    page.on('console', message => { if (message.text().startsWith('PROGRESS:')) console.log(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    assert.equal(await page.evaluate(() => crossOriginIsolated), true);
    const api = await page.evaluate(async () => {
      const wasm = await import('./pkg/crispr_cas_finder_wasm.js');
      await wasm.default();
      wasm.init_panic_hook();
      const fails = (fn, pattern) => { try { fn(); } catch(e) { return pattern.test(String(e)); } return false; };
      const result = {
        wrongType: fails(() => wasm.find_repeats('>a\nACGT', { min_repeat_length: 'bad' }), /invalid options/),
        wrongRange: fails(() => wasm.find_repeats('>a\nACGT', { min_repeat_length: 0 }), /repeat lengths/),
        missingHeader: fails(() => wasm.find_repeats('ACGT', {}), /FASTA/),
        emptySequence: fails(() => wasm.find_repeats('>a\n', {}), /empty FASTA/),
        defaults: Array.isArray(wasm.find_repeats('>a\nACGT', undefined)),
      };
      // Empty model list isolates the state machine from gene-prediction fixtures.
      const fasta = '>tiny\n' + 'ACGT'.repeat(100);
      const prep = wasm.cas_prepare(fasta, [], { metagenome: true });
      result.preparationObject = Array.isArray(prep.needed_profiles) && Number.isInteger(prep.gene_count);
      result.reentrant = fails(() => wasm.cas_prepare(fasta, [], {}), /already active/);
      wasm.cas_abort();
      wasm.cas_prepare(fasta, [], { metagenome: true });
      result.recovered = Array.isArray(wasm.cas_finalize());
      return result;
    });
    for (const [name, success] of Object.entries(api)) assert.equal(success, true, name);
    console.log('PASS: WASM input validation and CAS lifecycle');

    // A missing worker script must reject the pending UI request and permit retry.
    await page.route('**/worker.js', route => route.abort());
    await page.locator('#fasta-input').fill('>tiny\nACGTACGTACGT');
    await page.locator('#enable-cas').uncheck();
    await page.locator('#analyze-btn').click();
    await page.waitForFunction(() => document.querySelector('#status-panel').textContent.includes('Analysis failed'));
    assert.equal(await page.locator('#analyze-btn').isEnabled(), true);
    await page.unroute('**/worker.js');
    await page.locator('#analyze-btn').click();
    await page.waitForFunction(() => document.querySelector('#status-panel').textContent.includes('Analysis complete'));
    console.log('PASS: worker failure reaches UI and retry succeeds');

    const fasta = await fs.readFile(process.env.SMOKE_FASTA || path.resolve(__dirname, '../../crispr-cas-finder-cli/data/ecoli.fasta'), 'utf8');
    const outputs = [];
    for (const workerPath of (process.env.SMOKE_WORKER ? [process.env.SMOKE_WORKER] : ['worker.js', 'sequential-worker.js'])) {
      const result = await page.evaluate(({fasta, workerPath}) => new Promise((resolve, reject) => {
        const worker = new Worker(workerPath, {type:'module'});
        const progress = [];
        const timeout = setTimeout(() => { worker.terminate(); reject(new Error('analysis timeout')); }, 900000);
        worker.onerror = e => { clearTimeout(timeout); worker.terminate(); reject(new Error(e.message)); };
        worker.onmessage = ({data}) => {
          if (data.progress) { progress.push(data.progress.message); if (!data.progress.current || data.progress.current % 20 === 0) console.log("PROGRESS:", workerPath, data.progress.message); return; }
          clearTimeout(timeout); worker.terminate();
          if (data.error) reject(new Error(data.error)); else resolve({result:data.result, progress});
        };
        worker.postMessage({id:1, type:'run_full_analysis', payload:{fasta, options:{min_evidence_level:1}, casOpts:{genetic_code:11}}});
      }), { fasta, workerPath });
      assert.ok(result.result.crisprs.length > 0, 'E. coli arrays');
      assert.ok(result.result.cas_clusters.length > 0, 'E. coli CAS systems');
      assert.ok(result.progress.some(x => x.includes(workerPath === 'worker.js' ? 'in parallel' : 'sequentially')), workerPath);
      outputs.push(result.result);
      console.log(`PASS: ${workerPath}: ${result.result.crisprs.length} CRISPR arrays, ${result.result.cas_clusters.length} CAS systems`);
    }
    if (outputs.length === 2) assert.deepEqual(outputs[0], outputs[1], 'threaded and sequential results');
    if (process.env.SMOKE_OUTPUT) await fs.writeFile(process.env.SMOKE_OUTPUT, JSON.stringify(outputs[0], null, 2));
    if (outputs.length === 2) console.log('PASS: threaded/sequential equivalence');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
