"""In-process RivaSpeechRecognition server for the default test tier.

The servicer records request order and lifecycle. It does not store
authorization values.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures
from typing import Any

import grpc
from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc


class FakeRiva(riva_asr_pb2_grpc.RivaSpeechRecognitionServicer):  # type: ignore[misc]
    """A scripted recognition service on one process.

    The generated base is untyped, so the subclass suppression is local
    to that missing stub. Offline ``Recognize`` stays the base
    implementation.
    """

    def __init__(self, mode: str = "final", transcript: str = "HELLO") -> None:
        self.mode = mode
        self.transcript = transcript
        self.streams: list[list[tuple[str, float]]] = []
        self.configs: list[Any] = []
        self.saw_authorization = False
        self.function_id: str | None = None
        self.saw_cancelled = False
        self._lock = threading.Lock()

    def _note_metadata(self, context: Any) -> None:
        for key, value in context.invocation_metadata():
            if key == "function-id":
                self.function_id = value
            elif key == "authorization" and value.startswith("Bearer "):
                self.saw_authorization = True

    def GetRivaSpeechRecognitionConfig(self, request: Any, context: Any) -> Any:
        self._note_metadata(context)
        if self.mode == "config-denied":
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "nope")
        if self.mode == "config-unimplemented":
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "no config")
        config = riva_asr_pb2.RivaSpeechRecognitionConfigResponse.Config(
            model_name=request.model_name or "fake"
        )
        config.parameters["kind"] = "fake"
        return riva_asr_pb2.RivaSpeechRecognitionConfigResponse(
            model_config=[config]
        )

    def StreamingRecognize(self, request_iterator: Any, context: Any) -> Any:
        self._note_metadata(context)
        if self.mode == "unauth":
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "authorization Bearer SENTINEL-VALUE rejected",
            )
        if self.mode == "denied":
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "denied")
        if self.mode == "invalid":
            # Consume the config so the RPC is established, then name the
            # status. abort() before any read is sometimes reported as
            # INTERNAL by this grpc build.
            for _request in request_iterator:
                break
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("bad audio")
            return
        if self.mode == "echo-auth":
            echoed = ""
            for key, value in context.invocation_metadata():
                if key == "authorization" and value.startswith("Bearer "):
                    echoed = value[len("Bearer ") :]
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "rejected " + echoed)
        sequence: list[tuple[str, float]] = []
        with self._lock:
            self.streams.append(sequence)
        for request in request_iterator:
            kind = request.WhichOneof("streaming_request") or ""
            sequence.append((kind, time.monotonic()))
            if kind == "streaming_config":
                self.configs.append(request.streaming_config)
            if self.mode == "block":
                self._wait_until_inactive(context)
                return
            if self.mode == "interim" and kind == "audio_content":
                audio_seen = [
                    item for item in sequence if item[0] == "audio_content"
                ]
                if len(audio_seen) == 1:
                    yield _response("PARTIAL", final=False)
            if self.mode == "empty-interim" and kind == "audio_content":
                audio_seen = [
                    item for item in sequence if item[0] == "audio_content"
                ]
                if len(audio_seen) == 1:
                    yield _response("", final=False)
        if self.mode == "hang":
            self._wait_until_inactive(context)
            return
        if self.mode in {
            "final",
            "interim",
            "empty-interim",
            "config-unimplemented",
        }:
            yield _response(self.transcript, final=True)
        elif self.mode == "two-finals":
            yield _response("AB", final=True)
            yield _response("C", final=True)
        elif self.mode == "empty-final":
            yield _response("", final=True)

    def _wait_until_inactive(self, context: Any) -> None:
        """Block until the client cancels or five seconds pass."""
        deadline = time.monotonic() + 5
        while context.is_active() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.saw_cancelled = not context.is_active()


def _response(text: str, *, final: bool) -> Any:
    alternative = riva_asr_pb2.SpeechRecognitionAlternative(transcript=text)
    result = riva_asr_pb2.StreamingRecognitionResult(
        alternatives=[alternative],
        is_final=final,
    )
    return riva_asr_pb2.StreamingRecognizeResponse(results=[result])


class RunningFake:
    """One grpc.server bound to 127.0.0.1 on an ephemeral port."""

    def __init__(self, servicer: FakeRiva) -> None:
        self.servicer = servicer
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        riva_asr_pb2_grpc.add_RivaSpeechRecognitionServicer_to_server(
            servicer, self.server
        )
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.endpoint = f"127.0.0.1:{port}"
        self.server.start()

    def stop(self) -> None:
        self.server.stop(0)
