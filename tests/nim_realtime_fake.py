"""In-process NIM transcription WebSocket. No token is retained."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer


class FakeNimRealtime:
    """Serve the public mint and websocket paths on one test server.

    ``authorization`` is recorded only as a boolean. An ephemeral token
    returned by the mint is the constant the test asked for, and it is
    not copied onto this object.
    """

    def __init__(
        self,
        mode: str = "final",
        *,
        transcript: str = "HELLO",
        delta: str = "HE",
        ephemeral: str = "",
    ) -> None:
        self.mode = mode
        self.transcript = transcript
        self.delta = delta
        self._ephemeral = ephemeral
        self.events: list[str] = []
        self.append_at: list[float] = []
        self.query = ""
        self.protocols = ""
        self.session: dict[str, Any] | None = None
        self.saw_authorization = False
        self.saw_cancelled = False
        self.ready = asyncio.Event()

    def app(self) -> web.Application:
        application = web.Application()
        application.router.add_post(
            "/v1/realtime/transcription_sessions", self._mint
        )
        application.router.add_get("/v1/realtime", self._socket)
        return application

    async def _mint(self, request: web.Request) -> web.Response:
        header = request.headers.get("Authorization", "")
        token = ""
        if header.startswith("Bearer "):
            self.saw_authorization = True
            token = header[len("Bearer ") :]
        if self.mode == "unauth":
            return web.Response(status=401, text="rejected")
        if self.mode == "denied":
            return web.Response(status=403, text="rejected")
        if self.mode == "echo-auth":
            body = f"Bearer {token} {token}"
            del token
            return web.Response(status=401, text=body)
        del token
        payload: dict[str, Any] = {
            "object": "realtime.transcription_session",
            "id": "sess-test",
            "client_secret": None,
        }
        if self._ephemeral:
            payload["client_secret"] = {
                "value": self._ephemeral,
                "expires_at": 0,
            }
        return web.json_response(payload)

    async def _socket(self, request: web.Request) -> web.StreamResponse:
        self.query = request.query_string
        self.protocols = request.headers.get("Sec-WebSocket-Protocol", "")
        socket = web.WebSocketResponse(protocols=("realtime",))
        await socket.prepare(request)
        if (
            request.query.get("intent") != "transcription"
            or self.mode == "close"
        ):
            await socket.close(code=1008, message=b"intent")
            return socket
        async for message in socket:
            if message.type is not web.WSMsgType.TEXT:
                continue
            event = json.loads(message.data)
            kind = str(event.get("type"))
            self.events.append(kind)
            if kind == "transcription_session.update":
                session = event.get("session")
                self.session = session if isinstance(session, dict) else None
                if self.mode == "silent":
                    continue
                await socket.send_str(
                    json.dumps(
                        {
                            "type": "transcription_session.updated",
                            "session": session,
                        }
                    )
                )
                self.ready.set()
                if self.mode == "block":
                    # Reading is what observes the client close. A poll of
                    # socket.closed does not, because the close frame sits
                    # unread.
                    while True:
                        incoming = await socket.receive()
                        if incoming.type in {
                            web.WSMsgType.CLOSE,
                            web.WSMsgType.CLOSING,
                            web.WSMsgType.CLOSED,
                            web.WSMsgType.ERROR,
                        }:
                            self.saw_cancelled = True
                            return socket
                if self.mode == "error":
                    await socket.send_str(
                        json.dumps(
                            {
                                "type": "error",
                                "error": {
                                    "code": "invalid_event",
                                    "message": "bad audio",
                                },
                            }
                        )
                    )
                    await socket.close()
                    return socket
            elif kind == "input_audio_buffer.append":
                self.append_at.append(time.monotonic())
            elif kind == "input_audio_buffer.done":
                if self.mode == "hang":
                    continue
                await self._results(socket)
                await socket.close()
                return socket
        return socket

    async def _results(self, socket: web.WebSocketResponse) -> None:
        if self.mode == "final-only":
            await socket.send_str(
                json.dumps(
                    {
                        "type": (
                            "conversation.item.input_audio_transcription.completed"
                        ),
                        "transcript": self.transcript,
                        "is_last_result": True,
                    }
                )
            )
            return
        if self.mode == "partial":
            await socket.send_str(
                json.dumps(
                    {
                        "type": (
                            "conversation.item.input_audio_transcription.completed"
                        ),
                        "transcript": "PAR",
                        "is_last_result": False,
                    }
                )
            )
        else:
            await socket.send_str(
                json.dumps(
                    {
                        "type": (
                            "conversation.item.input_audio_transcription.delta"
                        ),
                        "delta": self.delta,
                    }
                )
            )
        await socket.send_str(
            json.dumps(
                {
                    "type": (
                        "conversation.item.input_audio_transcription.completed"
                    ),
                    "transcript": self.transcript,
                    "is_last_result": True,
                }
            )
        )


class RunningNim:
    """Bind a fake to 127.0.0.1:0 for one test."""

    def __init__(self, fake: FakeNimRealtime) -> None:
        self.fake = fake
        self.endpoint = ""
        self._server: TestServer | None = None

    async def start(self) -> str:
        server = TestServer(self.fake.app())
        await server.start_server()
        self._server = server
        assert server.port is not None
        self.endpoint = f"127.0.0.1:{server.port}"
        return self.endpoint

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.close()
