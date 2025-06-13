#!/bin/bash
export OLLAMA_HOST=127.0.0.1

# ✅ Install if missing
if ! command -v ollama &> /dev/null; then
  echo "⬇️ Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "✅ Ollama already installed."
fi

echo "🛑 Restarting systemd Ollama service..."
sudo systemctl restart ollama
echo "🚀 Ollama is now running on localhost:11434"