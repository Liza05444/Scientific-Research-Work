import { pipeline, env } from '@xenova/transformers';
import fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = join(SCRIPT_DIR, '..');
const DATA_SBERQUAD = join(ROOT, 'data', 'sberquad.json');
const RESULTS_JSON = join(ROOT, 'results', 'results_models.json');

env.allowRemoteModels = true;

const MODELS = [
  { id: 'gte-small', load: 'Xenova/gte-small', prefix: null, pooling: 'mean' },
  { id: 'e5-small', load: 'Xenova/multilingual-e5-small', prefix: { q: 'query: ', p: 'passage: ' }, pooling: 'mean' },
  { id: 'e5-base', load: 'Xenova/multilingual-e5-base', prefix: { q: 'query: ', p: 'passage: ' }, pooling: 'mean' },
  { id: 'e5-large', load: 'Xenova/multilingual-e5-large', prefix: { q: 'query: ', p: 'passage: ' }, pooling: 'mean' },
  { id: 'bge-m3', load: 'Xenova/bge-m3', prefix: null, pooling: 'cls' },
  { id: 'miniLM', load: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', prefix: null, pooling: 'mean' }
];

function countCosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function embed(extractor, texts, pooling = 'mean') {
  const out = await extractor(texts, { pooling, normalize: true });
  const embeddingValues = Array.from(out.data);
  const [rows, cols] = out.dims;
  return Array.from({ length: rows }, (_, i) => embeddingValues.slice(i * cols, (i + 1) * cols));
}

async function loadData() {
  try {
    const data = JSON.parse(await fs.readFile(DATA_SBERQUAD, 'utf-8'));
    return { contexts: data.contexts, queries: data.queries };
  } catch {
    console.log('data/sberquad.json not found. Run: python3 tools/download_data.py');
    process.exit(1);
  }
}

async function evaluate(model, contexts, queries) {
  const extractor = await pipeline('feature-extraction', model.load);

  const contextTexts = model.prefix ? contexts.map(text => model.prefix.p + text) : contexts;
  const contextVectors = [];
  for (let i = 0; i < contextTexts.length; i += 32) {
    const batch = contextTexts.slice(i, i + 32);
    const vectors = await embed(extractor, batch, model.pooling);
    contextVectors.push(...vectors);
  }

  let hit1 = 0, hit3 = 0, hit5 = 0, mrr = 0;

  for (let i = 0; i < queries.length; i += 32) {
    const batch = queries.slice(i, i + 32);
    const qTexts = model.prefix ? batch.map(q => model.prefix.q + q.question) : batch.map(q => q.question);
    const qVectors = await embed(extractor, qTexts, model.pooling);

    for (let j = 0; j < batch.length; j++) {
      const q = batch[j];
      const qVector = qVectors[j];

      const scores = contextVectors.map((pVector, idx) => ({ idx, score: countCosineSimilarity(qVector, pVector) }));
      scores.sort((a, b) => b.score - a.score);
      const top5 = scores.slice(0, 5).map(s => s.idx);

      const rank = top5.indexOf(q.correctParagraphKey);
      if (rank >= 0) {
        if (rank === 0) hit1++;
        if (rank < 3) hit3++;
        if (rank < 5) hit5++;
        mrr += 1 / (rank + 1);
      }
    }
  }

  const n = queries.length;

  return {
    model: model.id,
    queries: n,
    'Hit@1': (hit1 / n).toFixed(4),
    'Hit@3': (hit3 / n).toFixed(4),
    'Hit@5': (hit5 / n).toFixed(4),
    'MRR@5': (mrr / n).toFixed(4)
  };
}

async function main() {
  let { contexts, queries } = await loadData();

  const results = [];
  for (const model of MODELS) {
    try {
      const res = await evaluate(model, contexts, queries);
      results.push(res);
    } catch (e) {
      console.warn(`Skipped ${model.id}: ${e.message}`);
    }
  }

  console.log('\nRESULTS');
  console.table(results);

  await fs.mkdir(join(ROOT, 'results'), { recursive: true });
  await fs.writeFile(RESULTS_JSON, JSON.stringify(results, null, 2));
}

main().catch(console.error);
