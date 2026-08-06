const path = require("path");
const fs = require("fs");

const root = __dirname;
const isWin = process.platform === "win32";
const venvPython = isWin
  ? path.join(root, "venv", "Scripts", "python.exe")
  : path.join(root, "venv", "bin", "python");
const systemPython = isWin ? "python" : "python3";
const script = fs.existsSync(venvPython) ? venvPython : systemPython;

module.exports = {
  apps: [
    {
      name: "object-detector",
      cwd: root,
      script,
      args: "app.py",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: "10s",
      env: {
        DETECTOR_HOST: "0.0.0.0",
        DETECTOR_PORT: "7860",
        // auto | cpu | cuda — UI can still override per request
        DETECTOR_DEVICE: "auto",
        // Optional: absolute path for model ID root
        // DETECTOR_ROOT: root,
        // Optional: extra scan dirs (os.pathsep: : on Unix, ; on Windows)
        // DETECTOR_SCAN_ROOTS: "",
      },
    },
  ],
};
