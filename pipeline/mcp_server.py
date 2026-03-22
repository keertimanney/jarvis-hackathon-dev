"""MCP server exposing OpenClaw as a tool for Agora Conversational AI.

Runs as a Streamable HTTP MCP server. Agora's LLM calls the
"execute_desktop_command" tool when the user asks to do something
on the computer (open youtube, search google, etc.).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Disable DNS rebinding protection so tunnel (cloudflare/ngrok) can reach us
_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
mcp = FastMCP("OpenClaw Desktop Control", stateless_http=True, transport_security=_security)


@mcp.tool()
def execute_desktop_command(command: str) -> str:
    """Execute a desktop command on the computer using OpenClaw.

    Use this tool when the user asks you to do something on the computer,
    such as: open a website, search for something, play a video, open an
    application, click something, type text, scroll, navigate tabs, etc.

    Args:
        command: The natural language command to execute, e.g. "open youtube",
                 "search for cats on google", "play lo-fi music".
    """
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "main", "--message", command, "--json"],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()

        # Try to parse JSON response
        try:
            data = json.loads(output)
            return data.get("response", data.get("message", str(data)))
        except (json.JSONDecodeError, ValueError):
            # Filter out banner lines
            lines = output.split("\n")
            content = [l for l in lines if not l.startswith("\U0001f99e") and l.strip()]
            return "\n".join(content) if content else output

    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."
    except FileNotFoundError:
        return "OpenClaw CLI not found. Make sure it's installed."
    except Exception as e:
        return f"Error executing command: {e}"


_WEB_SERVER_URL = "http://localhost:8080"


def _call_mode_api(path, json_body=None):
    """Call a mode endpoint on the local web server."""
    import requests
    try:
        resp = requests.post(f"{_WEB_SERVER_URL}{path}", json=json_body, timeout=15)
        return resp.json().get("message", "OK")
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def start_person_tracking() -> str:
    """Start person tracking mode. The robot will follow the nearest person
    with its head and body. Use this when having a conversation or when
    the user wants the robot to look at them.
    Stops any other active mode (gesture control or dance).
    """
    return _call_mode_api("/api/mode/person-tracking/start")


@mcp.tool()
def stop_person_tracking() -> str:
    """Stop person tracking mode. The robot stops following people
    and returns to a neutral position."""
    return _call_mode_api("/api/mode/person-tracking/stop")


@mcp.tool()
def start_gesture_control() -> str:
    """Switch to gesture control mode. The user can control the desktop
    computer with hand gestures:
    - Swipe left/right with index finger = browser back/forward
    - Swipe up/down with index finger = scroll
    - Fist = click
    - OK sign (thumb + index) = Enter/confirm
    - Peace sign + motion = fast scroll
    Stops any other active mode (person tracking or dance).
    """
    return _call_mode_api("/api/mode/gesture-control/start")


@mcp.tool()
def stop_gesture_control() -> str:
    """Stop gesture control mode and return to idle."""
    return _call_mode_api("/api/mode/gesture-control/stop")


@mcp.tool()
def start_dance() -> str:
    """Start dance mode. The robot listens for music beats and dances
    along with a 4-beat sway-and-bop sequence. Use this when playing
    music for the user. Call execute_desktop_command to play music first,
    then call this tool.
    Stops any other active mode (person tracking or gesture control).
    """
    return _call_mode_api("/api/mode/dance/start")


@mcp.tool()
def stop_dance() -> str:
    """Stop dance mode. Robot returns to neutral position."""
    return _call_mode_api("/api/mode/dance/stop")


@mcp.tool()
def play_emotion(emotion_id: str) -> str:
    """Play a robot emotion animation. Works in any mode.

    Common emotions: amazed1, cheerful1, curious1, confused1, dance1,
    dance2, dance3, enthusiastic1, grateful1, laughing1, sad1,
    surprised1, welcoming1, yes1, no1, loving1, proud1, shy1.

    Args:
        emotion_id: The emotion animation to play (e.g. "cheerful1").
    """
    return _call_mode_api("/api/mode/emotion", {"emotion_id": emotion_id})


@mcp.tool()
def move_head(direction: str) -> str:
    """Move the robot's head in a direction. Works in any mode.

    Args:
        direction: One of: left, right, up, down, front, nod
    """
    return _call_mode_api("/api/mode/head", {"direction": direction})


def run_server(port=8000):
    """Start the MCP server (blocking)."""
    import uvicorn
    app = mcp.streamable_http_app()
    print(f"[mcp] OpenClaw MCP server starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenClaw MCP Server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)
