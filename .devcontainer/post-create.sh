#!/bin/bash
# ============================================================
# Common post-create setup (shared between CPU and GPU)
# [Template] research-project-template 由来
# ============================================================
set -e

PROJECT_NAME=${1:-$(basename "$(pwd)")}

# Claude Code config ownership
sudo chown -R "$(id -u):$(id -g)" /home/vscode/.claude

# Symlink for claude settings
ln -sf /home/vscode/.claude/.claude.json /home/vscode/.claude.json

# Deterministic machine-id (per project)
echo -n "devcontainer-${PROJECT_NAME}" | md5sum | cut -c1-32 | sudo tee /etc/machine-id > /dev/null

# TZ environment variable (claude-auto-retry がレート制限の再開時刻をパースするために必要)
# ホストの TZ が未設定の場合、/etc/timezone から fallback
if [ -z "$TZ" ] && [ -f /etc/timezone ]; then
  TZ=$(cat /etc/timezone)
  echo "export TZ='${TZ}'" | sudo tee /etc/profile.d/tz.sh > /dev/null
  echo "TZ set from /etc/timezone: $TZ"
fi

# claude-auto-retry (レート制限自動リトライ)
# 旧 Dockerfile の alias が残っている場合は除去（claude() 関数と競合するため）
sudo sed -i '/alias claude=/d' /etc/bash.bashrc 2>/dev/null || true

if ! command -v claude-auto-retry &> /dev/null; then
  sudo npm i -g claude-auto-retry
fi
# Note: claude-auto-retry install は postStartCommand (post-start.sh) で実行
# （postCreateCommand 時点では .bashrc が updateRemoteUserUID により上書きされる可能性があるため）

# claude-san symlink
sudo ln -sf "$(pwd)/claude-san" /usr/local/bin/claude-san

# --- [Project] stable_matching_with_weavenet 固有のセットアップ ---
# Python 依存 (pytorch-lightning, weavenet, torchviz, hydra, pyrootutils, ...) と
# graphviz バイナリは Dockerfile でシステム pip/apt に焼き込み済み。
# ここでは rebuild 毎に root 所有で再作成されてしまう ~/.config を vscode user に戻し、
# matplotlib などが ~/.config/<pkg> 配下にキャッシュを作れるようにする。
# (~/.config/gh は bind mount なので所有者は変えられないが、親ディレクトリ ~/.config は変えられる)
sudo chown vscode:vscode /home/vscode/.config 2>/dev/null || true
