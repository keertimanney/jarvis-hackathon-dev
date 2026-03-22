#!/bin/bash
# Launch all EVA services in one script.
# Usage: ./launch.sh [--no-robot] [--no-tunnel]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/venv/bin/python"
MCP_PORT=8001
WEB_PORT=8080
NO_ROBOT=false
NO_TUNNEL=false

for arg in "$@"; do
  case $arg in
    --no-robot)  NO_ROBOT=true ;;
    --no-tunnel) NO_TUNNEL=true ;;
  esac
done

# Colors
G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; R='\033[0;31m'; NC='\033[0m'

cleanup() {
  echo -e "\n${Y}Shutting down all services...${NC}"
  [ -n "$WEB_PID" ]    && kill $WEB_PID    2>/dev/null && echo "  Stopped web server"
  [ -n "$MCP_PID" ]    && kill $MCP_PID    2>/dev/null && echo "  Stopped MCP server"
  [ -n "$TUNNEL_PID" ] && kill $TUNNEL_PID 2>/dev/null && echo "  Stopped cloudflared"
  [ -n "$DAEMON_PID" ] && kill $DAEMON_PID 2>/dev/null && echo "  Stopped reachy daemon"
  wait 2>/dev/null
  echo -e "${G}All services stopped.${NC}"
}
trap cleanup EXIT INT TERM

# --- 1. Reachy daemon ---
if [ "$NO_ROBOT" = false ]; then
  echo -e "${B}[1/4] Starting Reachy daemon...${NC}"
  reachy-mini-daemon > /tmp/eva-daemon.log 2>&1 &
  DAEMON_PID=$!
  sleep 3
  if kill -0 $DAEMON_PID 2>/dev/null; then
    echo -e "${G}  Reachy daemon running (PID $DAEMON_PID)${NC}"
  else
    echo -e "${R}  Reachy daemon failed to start. Check /tmp/eva-daemon.log${NC}"
    DAEMON_PID=""
  fi
else
  echo -e "${Y}[1/4] Skipping Reachy daemon (--no-robot)${NC}"
fi

# --- 2. MCP server ---
echo -e "${B}[2/4] Starting MCP server on port $MCP_PORT...${NC}"
PYTHONPATH="$SCRIPT_DIR" "$PYTHON" pipeline/mcp_server.py --port $MCP_PORT > /tmp/eva-mcp.log 2>&1 &
MCP_PID=$!
sleep 2
if kill -0 $MCP_PID 2>/dev/null; then
  echo -e "${G}  MCP server running (PID $MCP_PID)${NC}"
else
  echo -e "${R}  MCP server failed. Check /tmp/eva-mcp.log${NC}"
  exit 1
fi

# --- 3. Cloudflared tunnel ---
if [ "$NO_TUNNEL" = false ]; then
  echo -e "${B}[3/4] Starting cloudflared tunnel...${NC}"
  cloudflared tunnel --url http://localhost:$MCP_PORT > /tmp/eva-tunnel.log 2>&1 &
  TUNNEL_PID=$!

  # Wait for the tunnel URL
  MCP_PUBLIC_URL=""
  for i in $(seq 1 30); do
    URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/eva-tunnel.log 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
      MCP_PUBLIC_URL="$URL"
      break
    fi
    sleep 1
  done

  if [ -n "$MCP_PUBLIC_URL" ]; then
    echo -e "${G}  Tunnel ready: $MCP_PUBLIC_URL${NC}"
  else
    echo -e "${R}  Tunnel URL not found after 30s. Check /tmp/eva-tunnel.log${NC}"
    echo -e "${Y}  Continuing without MCP tools (agent won't have tool access)${NC}"
  fi
else
  echo -e "${Y}[3/4] Skipping tunnel (--no-tunnel)${NC}"
  MCP_PUBLIC_URL=""
fi

# --- 4. Web server ---
echo -e "${B}[4/4] Starting web server on port $WEB_PORT...${NC}"
export MCP_PUBLIC_URL
export PYTHONPATH="$SCRIPT_DIR"
"$PYTHON" pipeline/agora_web_server.py --port $WEB_PORT --no-browser > /tmp/eva-web.log 2>&1 &
WEB_PID=$!
sleep 8

if kill -0 $WEB_PID 2>/dev/null; then
  echo -e "${G}  Web server running (PID $WEB_PID)${NC}"
else
  echo -e "${R}  Web server failed. Check /tmp/eva-web.log${NC}"
  exit 1
fi

# --- Ready ---
IP=$(hostname -I | awk '{print $1}')
echo ""
echo -e "${G}========================================${NC}"
echo -e "${G}  JARVIS is ready!${NC}"
echo -e "${G}========================================${NC}"
echo -e "  Dashboard:  ${B}http://${IP}:${WEB_PORT}${NC}"
[ -n "$MCP_PUBLIC_URL" ] && echo -e "  MCP Tunnel: ${B}${MCP_PUBLIC_URL}${NC}"
echo -e "  Logs:       /tmp/eva-{daemon,mcp,tunnel,web}.log"
echo ""
echo -e "${Y}Press Ctrl+C to stop all services${NC}"
echo ""

# Tail the web server log so user can see activity
tail -f /tmp/eva-web.log
