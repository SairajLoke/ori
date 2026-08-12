"""Small local-only HTTP server for the North episode visualizer."""
from __future__ import annotations

import json
import mimetypes
import pathlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
from typing import Any


def make_handler(controller, static_root: pathlib.Path, robot_assets_root: pathlib.Path, video_files: dict[str, pathlib.Path] | None = None):
    static_root = static_root.resolve()
    robot_assets_root = robot_assets_root.resolve()
    video_files = {k: v.resolve() for k, v in (video_files or {}).items()}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    return self._send_file(static_root, "index.html", {".html"})
                if path.startswith("/static/"):
                    return self._send_file(static_root, path[len("/static/"):], {".js", ".css", ".txt", ".map"})
                if path == "/api/status":
                    return self._send_json(controller.status())
                if path == "/api/trajectory":
                    return self._send_json(controller.trajectory())
                if path == "/api/observation":
                    return self._send_json(controller.observation())
                if path == "/api/robot/config":
                    return self._send_json(controller.robot_config())
                if path == "/api/logs":
                    return self._send_json(controller.logs())
                if path.startswith("/robot-assets/"):
                    return self._send_file(
                        robot_assets_root,
                        unquote(path[len("/robot-assets/"):]),
                        {".urdf", ".stl", ".dae", ".obj", ".mtl"},
                    )
                if path.startswith("/video/"):
                    key = unquote(path[len("/video/"):])
                    video = video_files.get(key)
                    if video is None:
                        return self._send_json({"ok": False, "error": "video not configured"}, HTTPStatus.NOT_FOUND)
                    return self._send_video(video)
                return self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                return self._send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _send_file(self, root: pathlib.Path, relative: str, allowed_suffixes: set[str]) -> None:
            rel = pathlib.PurePosixPath(relative)
            if rel.is_absolute() or ".." in rel.parts:
                return self._send_json({"ok": False, "error": "invalid asset path"}, HTTPStatus.BAD_REQUEST)
            target = (root / pathlib.Path(*rel.parts)).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                return self._send_json({"ok": False, "error": "invalid asset path"}, HTTPStatus.BAD_REQUEST)
            if target.suffix.lower() not in allowed_suffixes or not target.is_file():
                return self._send_json({"ok": False, "error": f"asset not found: {relative}"}, HTTPStatus.NOT_FOUND)
            return self._send(HTTPStatus.OK, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

        def _send_video(self, target: pathlib.Path) -> None:
            # Browser video playback benefits from byte-range requests. Support
            # them so scrubbing remains responsive even for large dataset MP4s.
            size = target.stat().st_size
            range_header = self.headers.get("Range")
            start, end = 0, size - 1
            status = HTTPStatus.OK
            if range_header and range_header.startswith("bytes="):
                spec = range_header[6:].split(",", 1)[0].strip()
                if "-" in spec:
                    left, right = spec.split("-", 1)
                    if left:
                        start = int(left)
                    if right:
                        end = int(right)
                    else:
                        end = size - 1
                    if start < 0 or start >= size or end < start:
                        return self._send_json({"ok": False, "error": "invalid byte range"}, HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    end = min(end, size - 1)
                    status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            with open(target, "rb") as f:
                f.seek(start)
                body = f.read(length)
            self.send_response(status.value)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            return self._send(status, body, "application/json; charset=utf-8")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            try:
                self.send_response(status.value)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            return self._send_json({"ok": False, "error": "read-only visualizer"}, HTTPStatus.METHOD_NOT_ALLOWED)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def create_server(controller, *, host: str, port: int, static_root: pathlib.Path, robot_assets_root: pathlib.Path, video_files: dict[str, pathlib.Path] | None = None):
    return ThreadingHTTPServer((host, port), make_handler(controller, static_root, robot_assets_root, video_files))
