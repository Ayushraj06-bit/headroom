"""Tests for serving the proxy on a Unix domain socket (`headroom proxy --uds`).

The socket transport is a plain alternative to a TCP port — see
`headroom/proxy/uds.py` for the rationale and for why it does not restore
Claude Code's Remote Control (GH #1779).
"""

from __future__ import annotations

import socket
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.proxy import proxy as proxy_cmd
from headroom.proxy.uds import (
    UDS_SUPPORTED,
    UdsError,
    max_uds_path_length,
    prepare_uds_path,
    remove_uds_path,
    require_uds_support,
)

requires_uds = pytest.mark.skipif(
    not UDS_SUPPORTED, reason="platform has no socket.AF_UNIX (Windows)"
)

try:  # `headroom.proxy.server` pulls in the compiled Rust core.
    import headroom._core  # noqa: F401

    _CORE_BUILT = True
except ImportError:  # pragma: no cover - depends on the local build
    _CORE_BUILT = False

requires_core = pytest.mark.skipif(
    not _CORE_BUILT, reason="headroom._core is not built in this environment"
)


# --------------------------------------------------------------------------
# Platform capability — runs everywhere, since the platform is a parameter.
# --------------------------------------------------------------------------


def test_require_uds_support_rejects_windows() -> None:
    """Windows has neither socket.AF_UNIX nor an asyncio UDS transport."""
    with pytest.raises(UdsError, match="unavailable on this platform"):
        require_uds_support(platform="win32")


def test_require_uds_support_accepts_posix() -> None:
    if not UDS_SUPPORTED:
        pytest.skip("AF_UNIX missing; the platform argument cannot override that")
    require_uds_support(platform="linux")


def test_sun_path_limit_is_platform_specific() -> None:
    """Linux allows 108 bytes, the BSDs and macOS 104. Guessing high truncates."""
    assert max_uds_path_length("linux") == 108
    assert max_uds_path_length("darwin") == 104


def test_cli_rejects_uds_on_windows() -> None:
    """The CLI fails fast with a readable error, not a bind-time OSError."""
    with patch("headroom.proxy.uds.UDS_SUPPORTED", False):
        result = CliRunner().invoke(proxy_cmd, ["--uds", "/tmp/headroom-test.sock"])

    assert result.exit_code != 0
    assert "Unix domain sockets" in result.output
    assert "--port instead" in result.output


# --------------------------------------------------------------------------
# Path preparation — needs a real AF_UNIX platform.
# --------------------------------------------------------------------------


@requires_uds
def test_prepare_creates_parent_owner_only(tmp_path: Path) -> None:
    """The directory mode is the access-control boundary for the socket."""
    target = tmp_path / "run" / "headroom.sock"

    resolved = prepare_uds_path(target)

    assert resolved == target
    assert target.parent.is_dir()
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


@requires_uds
def test_prepare_clears_a_stale_socket(tmp_path: Path) -> None:
    """A crashed proxy leaves an inode behind; a restart must not trip on it."""
    target = tmp_path / "stale.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(target))
    dead.close()  # closing without unlinking is exactly the crash case
    assert target.exists()

    prepare_uds_path(target)

    assert not target.exists()


@requires_uds
def test_prepare_refuses_a_live_socket(tmp_path: Path) -> None:
    """Two proxies on one socket would silently steal each other's traffic."""
    target = tmp_path / "live.sock"
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(target))
    live.listen(1)
    try:
        with pytest.raises(UdsError, match="already listening"):
            prepare_uds_path(target)
        assert target.exists(), "the live socket must survive the refusal"
    finally:
        live.close()
        target.unlink(missing_ok=True)


@requires_uds
def test_prepare_never_deletes_a_regular_file(tmp_path: Path) -> None:
    """A typo'd --uds pointing at real data must not destroy it."""
    target = tmp_path / "notes.txt"
    target.write_text("important", encoding="utf-8")

    with pytest.raises(UdsError, match="is not a socket"):
        prepare_uds_path(target)

    assert target.read_text(encoding="utf-8") == "important"


@requires_uds
def test_prepare_rejects_an_oversized_path(tmp_path: Path) -> None:
    """Past sun_path, bind() fails with an ENAMETOOLONG that names nothing."""
    target = tmp_path / ("d" * 120) / "headroom.sock"

    with pytest.raises(UdsError, match="sun_path limit"):
        prepare_uds_path(target)


@requires_uds
def test_remove_uds_path_is_socket_only(tmp_path: Path) -> None:
    """Cleanup runs in a finally block, so it must be narrow and never raise."""
    sock_path = tmp_path / "gone.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(sock_path))
    sock.close()
    regular = tmp_path / "keep.txt"
    regular.write_text("keep", encoding="utf-8")

    remove_uds_path(sock_path)
    remove_uds_path(regular)
    remove_uds_path(tmp_path / "does-not-exist.sock")

    assert not sock_path.exists()
    assert regular.exists()


# --------------------------------------------------------------------------
# Server wiring — uvicorn is mocked, so this runs on every platform.
# --------------------------------------------------------------------------


def _bind_kwargs_for(**config_kwargs: object) -> dict[str, object]:
    """Run run_server far enough to capture what it would bind to."""
    from headroom.proxy.server import ProxyConfig, run_server

    captured: dict[str, object] = {}

    def fake_run_uvicorn(  # noqa: ANN202
        app_target,  # noqa: ANN001
        bind_kwargs,  # noqa: ANN001
        workers,  # noqa: ANN001
        limit_concurrency,  # noqa: ANN001
        log_level,  # noqa: ANN001
        uvicorn_kwargs,  # noqa: ANN001
    ):
        captured.update(bind_kwargs)

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=fake_run_uvicorn),
        patch("headroom.proxy.server.create_app"),
    ):
        run_server(ProxyConfig(**config_kwargs), print_banner=False)  # type: ignore[arg-type]

    return captured


@requires_core
def test_run_server_binds_host_and_port_by_default() -> None:
    bind = _bind_kwargs_for(host="127.0.0.1", port=9123)

    assert bind == {"host": "127.0.0.1", "port": 9123}


@requires_uds
@requires_core
def test_run_server_binds_the_socket_instead_of_a_port(tmp_path: Path) -> None:
    """uvicorn treats uds and host/port as alternatives; passing both is an error."""
    target = tmp_path / "headroom.sock"

    bind = _bind_kwargs_for(host="127.0.0.1", port=9123, uds=str(target))

    assert bind == {"uds": str(target)}
    assert "host" not in bind and "port" not in bind


@requires_uds
@requires_core
def test_run_server_removes_the_socket_on_exit(tmp_path: Path) -> None:
    """A crash inside uvicorn must not leave an inode that blocks the restart."""
    from headroom.proxy.server import ProxyConfig, run_server

    target = tmp_path / "headroom.sock"

    def bind_then_fail(  # noqa: ANN202
        app_target,  # noqa: ANN001
        bind_kwargs,  # noqa: ANN001
        workers,  # noqa: ANN001
        limit_concurrency,  # noqa: ANN001
        log_level,  # noqa: ANN001
        uvicorn_kwargs,  # noqa: ANN001
    ):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(bind_kwargs["uds"])
        sock.close()
        raise KeyboardInterrupt

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=bind_then_fail),
        patch("headroom.proxy.server.create_app"),
        pytest.raises(KeyboardInterrupt),
    ):
        run_server(ProxyConfig(uds=str(target)), print_banner=False)

    assert not target.exists()
