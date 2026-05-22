#!/usr/bin/env bash
# Ensure host-side files/dirs exist before the devcontainer's bind mounts grab them.
#
# Background: docker-compose.yml bind-mounts ${HOME}/.gitconfig and
# ${HOME}/.config/gh into the container. If the host source path is missing,
# Docker auto-creates a *directory* at that location, which is the wrong type
# for .gitconfig (must be a file) and breaks every git invocation inside the
# container with `fatal: ... .gitconfig: Is a directory`.
#
# This script runs as a devcontainer `initializeCommand` on the host (Mac/Linux)
# and is a no-op once the files exist.
set -eu

if [ -e "$HOME/.gitconfig" ] && [ ! -f "$HOME/.gitconfig" ]; then
  echo "ERROR: $HOME/.gitconfig exists but is not a regular file." >&2
  echo "       This is usually an empty directory left over from a previous" >&2
  echo "       devcontainer build when ~/.gitconfig was missing. Remove it" >&2
  echo "       (e.g. 'rmdir ~/.gitconfig') and re-run." >&2
  exit 1
fi
[ -f "$HOME/.gitconfig" ] || touch "$HOME/.gitconfig"
mkdir -p "$HOME/.config/gh"
