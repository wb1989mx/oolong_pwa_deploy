/* 乌龙茶识别 PWA - 公共逻辑 */
const CLASSES = ["东方美人", "台湾乌龙", "广东单丛", "武夷岩茶", "浓香铁观音", "清香铁观音", "陈香铁观音", "黄金桂"];
const MODEL_URL = "model/oolong_v12_single.onnx";
const WASM_PATHS = "./lib/";

let session = null;
let sessionLoading = null;

/* ---------- 模型加载（懒加载，仅首次推理时） ---------- */
async function loadModel(onProgress) {
  if (session) return session;
  if (sessionLoading) return sessionLoading;
  sessionLoading = (async () => {
    // 运行时 wasm 走 CDN（仓库体积受限），SW 缓存后离线可用
    ort.env.wasm.wasmPaths = {
      mjs: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.mjs",
      wasm: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.wasm",
    };
    if (onProgress) onProgress("正在加载模型…");
    const sess = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    session = sess;
    return sess;
  })();
  try {
    return await sessionLoading;
  } finally {
    sessionLoading = null;
  }
}

/* ---------- 图像预处理（与训练一致：拉伸 224×224 + ImageNet 归一化） ---------- */
function preprocess(img) {
  const W = 224, H = 224;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, W, H);
  const data = ctx.getImageData(0, 0, W, H).data;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const N = W * H;
  const out = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    const r = data[i * 4] / 255, g = data[i * 4 + 1] / 255, b = data[i * 4 + 2] / 255;
    out[i] = (r - mean[0]) / std[0];
    out[N + i] = (g - mean[1]) / std[1];
    out[2 * N + i] = (b - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", out, [1, 3, H, W]);
}

function softmax(logits) {
  const ex = logits.map(Math.exp);
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(x => x / s);
}

/* ---------- 推理：返回 [{cls, prob, idx}] 降序 ---------- */
async function predict(img) {
  const sess = await loadModel();
  const tensor = preprocess(img);
  const feeds = { input: tensor };
  const results = await sess.run(feeds);
  const logits = Array.from(results.logits.data);
  const probs = softmax(logits);
  return probs
    .map((p, i) => ({ cls: CLASSES[i], prob: p, idx: i }))
    .sort((a, b) => b.prob - a.prob);
}

/* ---------- IndexedDB：识别历史 + 待入库清单 ---------- */
const DB_NAME = "oolong_pwa";
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("history")) db.createObjectStore("history", { keyPath: "id", autoIncrement: true });
      if (!db.objectStoreNames.contains("inbox")) db.createObjectStore("inbox", { keyPath: "id", autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function dbAdd(store, obj) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).add(obj);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function dbAll(store) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readonly");
    const req = tx.objectStore(store).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function dbClear(store) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function dbDelete(store, id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/* ---------- 品种说明 ---------- */
const CLASS_DESC = {
  "东方美人": "白毫显露、红黄白三色相间，条索细紧卷曲（含大田美人茶），带蜜香花果香。",
  "台湾乌龙": "颗粒紧结圆润（冻顶/高山茶），色泽墨绿带油光，通常带茶梗。",
  "广东单丛": "条索壮结挺直、色泽黄褐油润（凤凰/岭头单丛），香气高锐。",
  "武夷岩茶": "条索紧结、叶端扭曲，色泽铁青带褐（大红袍/肉桂/水仙），岩韵明显。",
  "浓香铁观音": "条索卷曲紧结，色泽乌褐油润（焙火较足），带焙火香/焦糖香。",
  "清香铁观音": "条索卷曲紧结、砂绿显，形似蜻蜓头、螺旋体，带兰花香/清香。",
  "陈香铁观音": "老铁陈化，条索卷曲，色泽深褐油润，带陈香/药香，汤色橙红。",
  "黄金桂": "条索细紧，色泽黄绿带润，汤色金黄，香气清雅带桂花韵。",
};
const CLASS_HINT = {
  "东方美人": "辨识要点：看白毫与红褐黄三色相间，蜜香显著。",
  "台湾乌龙": "辨识要点：看颗粒形态与茶梗，区别于条索形。",
  "广东单丛": "辨识要点：看条索壮结挺直、黄褐油润。",
  "武夷岩茶": "辨识要点：看叶端扭曲、铁青带褐、岩韵。",
  "浓香铁观音": "辨识要点：看乌褐油润与焙火气息，区别于砂绿清香。",
  "清香铁观音": "辨识要点：看砂绿与蜻蜓头卷曲形态。",
  "陈香铁观音": "辨识要点：看深褐陈化色泽，老铁药香明显。",
  "黄金桂": "辨识要点：看条索细紧、黄绿带润。",
};

/* ---------- 公共 UI ---------- */
function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => reject(new Error("图片读取失败"));
    img.src = url;
  });
}
function fmtPct(p) { return (p * 100).toFixed(1) + "%"; }
