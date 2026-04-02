const path = require("path");
const express = require("express");

const app = express();

const PUBLIC_DIR = path.resolve(__dirname, "public");

app.use(express.static(PUBLIC_DIR, { extensions: ["html"] }));

app.get("/", (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, "index.html"));
});

const PORT = Number(process.env.PORT || 8000);
app.listen(PORT, () => {
  console.log(`[frontend] UI server listening on http://localhost:${PORT}`);
});

