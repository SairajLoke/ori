"""Small local-only HTTP server for the North prediction visualizer."""

from __future__ import annotations

import json
import mimetypes
import pathlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
from typing import Any


def make_handler(controller, static_root: pathlib.Path, robot_assets_root: pathlib.Path):
    static_root = static_root.resolve()
    robot_assets_root = robot_assets_root.resolve()

    class Handler(BaseHTTPRequestHandler):

        def do_GET(self) -> None:
            path = urlsplit(self.path).path

            try:
                # Main HTML page
                if path == "/":
                    return self._send_file(
                        static_root,
                        "index.html",
                        {".html"},
                    )

                # JS / CSS / vendor files
                if path.startswith("/static/"):
                    relative = path[len("/static/"):]

                    return self._send_file(
                        static_root,
                        relative,
                        {
                            ".js",
                            ".css",
                            ".txt",
                            ".map",
                        },
                    )

                # Prediction/status APIs
                if path == "/api/status":
                    return self._send_json(
                        controller.status()
                    )

                if path == "/api/trajectory":
                    return self._send_json(
                        controller.trajectory()
                    )

                if path == "/api/observation":
                    return self._send_json(
                        controller.observation()
                    )

                if path == "/api/robot/config":
                    return self._send_json(
                        controller.robot_config()
                    )

                if path == "/api/logs":
                    return self._send_json(
                        controller.logs()
                    )

                # North URDF / STL assets
                if path.startswith("/robot-assets/"):
                    relative = unquote(
                        path[len("/robot-assets/"):]
                    )

                    return self._send_file(
                        robot_assets_root,
                        relative,
                        {
                            ".urdf",
                            ".stl",
                            ".dae",
                            ".obj",
                            ".mtl",
                        },
                    )

                return self._send_json(
                    {"ok": False, "error": "not found"},
                    HTTPStatus.NOT_FOUND,
                )

            except Exception as error:
                return self._send_json(
                    {
                        "ok": False,
                        "error": str(error),
                    },
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _send_file(
            self,
            root: pathlib.Path,
            relative: str,
            allowed_suffixes: set[str],
        ) -> None:

            rel = pathlib.PurePosixPath(relative)

            # Prevent ../ traversal.
            if rel.is_absolute() or ".." in rel.parts:
                return self._send_json(
                    {"ok": False, "error": "invalid asset path"},
                    HTTPStatus.BAD_REQUEST,
                )

            target = (
                root / pathlib.Path(*rel.parts)
            ).resolve()

            try:
                target.relative_to(root)
            except ValueError:
                return self._send_json(
                    {"ok": False, "error": "invalid asset path"},
                    HTTPStatus.BAD_REQUEST,
                )

            if (
                target.suffix.lower() not in allowed_suffixes
                or not target.is_file()
            ):
                return self._send_json(
                    {
                        "ok": False,
                        "error": f"asset not found: {relative}",
                    },
                    HTTPStatus.NOT_FOUND,
                )

            content_type = (
                mimetypes.guess_type(target.name)[0]
                or "application/octet-stream"
            )

            return self._send(
                HTTPStatus.OK,
                target.read_bytes(),
                content_type,
            )

        def _send_json(
            self,
            value: Any,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:

            body = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()

            return self._send(
                status,
                body,
                "application/json; charset=utf-8",
            )

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
        ) -> None:

            try:
                self.send_response(status.value)

                self.send_header(
                    "Content-Type",
                    content_type,
                )

                self.send_header(
                    "Content-Length",
                    str(len(body)),
                )

                self.send_header(
                    "Cache-Control",
                    "no-store",
                )

                self.send_header(
                    "X-Content-Type-Options",
                    "nosniff",
                )

                self.send_header(
                    "Content-Security-Policy",
                    (
                        "default-src 'self'; "
                        "script-src 'self'; "
                        "style-src 'self'; "
                        "connect-src 'self'; "
                        "frame-ancestors 'none'"
                    ),
                )

                self.end_headers()
                self.wfile.write(body)

            except (
                BrokenPipeError,
                ConnectionResetError,
            ):
                pass

        def do_POST(self) -> None:
            return self._send_json(
                {
                    "ok": False,
                    "error": "read-only visualizer",
                },
                HTTPStatus.METHOD_NOT_ALLOWED,
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def create_server(
    controller,
    *,
    host: str,
    port: int,
    static_root: pathlib.Path,
    robot_assets_root: pathlib.Path,
):

    return ThreadingHTTPServer(
        (host, port),
        make_handler(
            controller,
            static_root,
            robot_assets_root,
        ),
    )