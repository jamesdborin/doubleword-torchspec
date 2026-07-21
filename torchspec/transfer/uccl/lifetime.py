"""Producer buffer lifetime tracking and acknowledgement transport."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Lease:
    value: Any
    release: Callable[[], None]
    remaining: int
    expires_at: float


class LifetimeRegistry:
    """Retain producer registrations until ACK count or TTL permits release."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._leases: dict[str, _Lease] = {}
        self._lock = threading.Lock()

    def register(
        self,
        token: str,
        value: Any,
        release: Callable[[], None],
        expected_consumers: int,
        ttl_seconds: float,
    ) -> None:
        if expected_consumers < 1:
            raise ValueError("expected_consumers must be at least one")
        with self._lock:
            if token in self._leases:
                raise ValueError(f"duplicate lifetime token: {token}")
            self._leases[token] = _Lease(
                value=value,
                release=release,
                remaining=expected_consumers,
                expires_at=self._clock() + ttl_seconds,
            )

    def ack(self, token: str) -> bool:
        release = None
        with self._lock:
            lease = self._leases.get(token)
            if lease is None:
                return False
            lease.remaining -= 1
            if lease.remaining <= 0:
                release = self._leases.pop(token).release
        if release is not None:
            release()
        return True

    def release(self, token: str) -> bool:
        with self._lock:
            lease = self._leases.pop(token, None)
        if lease is None:
            return False
        lease.release()
        return True

    def cleanup_expired(self) -> int:
        now = self._clock()
        with self._lock:
            expired = [token for token, lease in self._leases.items() if lease.expires_at <= now]
            leases = [self._leases.pop(token) for token in expired]
        for lease in leases:
            lease.release()
        return len(leases)

    def close(self) -> None:
        with self._lock:
            leases = list(self._leases.values())
            self._leases.clear()
        for lease in leases:
            lease.release()

    def __len__(self) -> int:
        with self._lock:
            return len(self._leases)


class ZmqAckServer:
    """Small REP listener that forwards consumer ACK tokens to a registry."""

    def __init__(
        self,
        registry: LifetimeRegistry,
        bind_host: str,
        advertise_host: str,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required by the UCCL producer ACK service") from exc

        self._zmq = zmq
        self._registry = registry
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        port = self._socket.bind_to_random_port(f"tcp://{bind_host}")
        self.address = f"tcp://{advertise_host}:{port}"
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="uccl-ack-registry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._socket, self._zmq.POLLIN)
        while not self._closed.is_set():
            events = dict(poller.poll(100))
            if self._socket in events:
                try:
                    request = json.loads(self._socket.recv().decode("utf-8"))
                    accepted = self._registry.ack(request["token"])
                    self._socket.send_json({"accepted": accepted})
                except Exception as exc:
                    self._socket.send_json({"accepted": False, "error": str(exc)})
            self._registry.cleanup_expired()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._thread.join(timeout=2.0)
        self._socket.close(linger=0)
        self._context.term()


def send_zmq_ack(address: str, token: str, timeout_seconds: float) -> bool:
    """Send one acknowledgement without leaving a persistent socket behind."""

    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("pyzmq is required to acknowledge a UCCL transfer") from exc

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    timeout_ms = max(1, int(timeout_seconds * 1000))
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        socket.connect(address)
        socket.send_json({"token": token})
        return bool(socket.recv_json().get("accepted"))
    finally:
        socket.close(linger=0)
        context.term()
