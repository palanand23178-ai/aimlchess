const path = require("path");
const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PROJECT_ROOT = path.resolve(__dirname, "..");
const INTERNAL_PORT = Number(process.env.INTERNAL_PORT || 3001);

let fastApiProcess = null;
let status = "starting"; // starting | ready | error

// Simple queue to avoid concurrent inference calls into the same model.
let chain = Promise.resolve();

function startFastApiOnce() {
  if (fastApiProcess) return;

  status = "starting";
  fastApiProcess = spawn(
    "python3",
    ["-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", String(INTERNAL_PORT)],
    {
      cwd: PROJECT_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    }
  );

  fastApiProcess.stdout.on("data", (data) => {
    const msg = data.toString().trimEnd();
    if (msg) console.log(`[fastapi] ${msg}`);
  });
  fastApiProcess.stderr.on("data", (data) => {
    const msg = data.toString().trimEnd();
    if (msg) console.error(`[fastapi] ${msg}`);
  });

  fastApiProcess.on("exit", (code, signal) => {
    console.log(`[fastapi] exited (code=${code} signal=${signal})`);
    fastApiProcess = null;
    if (status !== "ready") status = "error";
  });
}

async function waitForFastApiReady(timeoutMs = 30000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const res = await fetch(`http://127.0.0.1:${INTERNAL_PORT}/openapi.json`);
      if (res.ok) return true;
    } catch {
      // keep retrying
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

async function forwardAiMove(payload) {
  const res = await fetch(`http://127.0.0.1:${INTERNAL_PORT}/ai_move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { raw: text };
  }

  if (!res.ok) {
    const err = new Error(`AI backend error: HTTP ${res.status}`);
    err.details = data;
    throw err;
  }

  return data;
}

startFastApiOnce();
waitForFastApiReady()
  .then((ok) => {
    status = ok ? "ready" : "error";
    console.log(`[backend] FastAPI ${status === "ready" ? "ready" : "failed to start"}`);
  })
  .catch(() => {
    status = "error";
    console.error("[backend] FastAPI readiness check failed");
  });

app.get("/health", (req, res) => {
  res.json({ status, internalPort: INTERNAL_PORT });
});

app.post("/ai_move", (req, res) => {
  const payload = req.body || {};
  if (typeof payload.fen !== "string" || !payload.fen.trim()) {
    return res.status(400).json({ error: "Missing required field: fen" });
  }

  // Serialize requests to avoid concurrent GPU/CPU inference calls.
  chain = chain
    .then(async () => {
      if (status !== "ready") {
        // Let the user retry if the model is still loading.
        const err = new Error("AI backend not ready yet");
        err.statusCode = 503;
        throw err;
      }
      return forwardAiMove(payload);
    })
    .then((data) => {
      res.json(data);
    })
    .catch((err) => {
      const code = err.statusCode || 500;
      res.status(code).json({
        error: err.message || "Unexpected error",
        details: err.details,
      });
    });
});

const PORT = Number(process.env.PORT || 3000);
app.listen(PORT, () => {
  console.log(`[backend] Node backend listening on http://localhost:${PORT}`);
});

process.on("SIGINT", () => {
  if (fastApiProcess) fastApiProcess.kill("SIGINT");
  process.exit(0);
});

