# hmmer-wasm

WebAssembly bindings for HMMER, built with
[wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/).

## Building

```bash
wasm-pack build --target web crates/hmmer-wasm
```

## API

### `hmmsearch(hmm_path, seq_path, e_threshold)`

Search an HMM against a sequence database. Returns a JavaScript object with:

```javascript
{
  nhits: number,
  nseq: number,
  hits: [{ name: string, score: number, evalue: number, ndom: number }]
}
```
