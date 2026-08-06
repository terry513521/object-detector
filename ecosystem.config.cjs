module.exports = {
  apps: [
    {
      name: "object-detector",
      cwd: "/root/object-detector",
      script: "venv/bin/python",
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
      },
    },
  ],
};
