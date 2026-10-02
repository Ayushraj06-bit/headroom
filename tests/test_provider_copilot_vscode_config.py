from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from headroom.providers.copilot.vscode import (
    configure_vscode_proxy_settings,
    remove_vscode_proxy_settings,
    unrouted_vscode_profiles,
    vscode_proxy_url,
    vscode_settings_path,
    vscode_user_dir,
)


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        (
            "darwin",
            {"HOME": "/Users/a"},
            "/Users/a/Library/Application Support/Code/User/settings.json",
        ),
        (
            "win32",
            {"APPDATA": "C:/Users/A/AppData/Roaming"},
            "C:/Users/A/AppData/Roaming/Code/User/settings.json",
        ),
        ("linux", {"HOME": "/home/a"}, "/home/a/.config/Code/User/settings.json"),
        ("linux", {"HOME": "/home/a", "XDG_CONFIG_HOME": "/cfg"}, "/cfg/Code/User/settings.json"),
    ],
)
def test_vscode_settings_path_is_cross_platform(
    platform: str, env: dict[str, str], expected: str
) -> None:
    assert str(vscode_settings_path(platform=platform, environ=env)).replace("\\", "/") == expected


def test_proxy_url_carries_project_without_changing_model() -> None:
    assert vscode_proxy_url(8787, "my project") == "http://127.0.0.1:8787/p/my%20project"


def test_configure_update_and_remove_preserve_jsonc_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = '{\n\t// user comment\n\t"editor.fontSize": 15,\n}\n'
    path.write_text(original, encoding="utf-8")

    assert configure_vscode_proxy_settings(path, "http://127.0.0.1:8787") == "added"
    configured = path.read_text(encoding="utf-8")
    assert "editor.fontSize" in configured
    assert "user comment" in configured
    assert '"github.copilot.advanced.debug.overrideProxyUrl"' in configured
    assert '"github.copilot.advanced.debug.overrideCapiUrl"' in configured
    # `overrideAuthType` is deliberately absent: no such setting exists in the
    # modern Copilot Chat extension, and writing one made VS Code flag an
    # unknown key while doing nothing (#3076).
    assert "overrideAuthType" not in configured

    assert configure_vscode_proxy_settings(path, "http://127.0.0.1:9999") == "updated"
    assert "9999" in path.read_text(encoding="utf-8")
    assert "8787" not in path.read_text(encoding="utf-8")

    assert remove_vscode_proxy_settings(path) is True
    assert path.read_text(encoding="utf-8") == original
    assert remove_vscode_proxy_settings(path) is False


def test_configure_refuses_malformed_and_unmanaged_override(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    for original in (
        "{broken",
        '{"github.copilot.advanced.debug.overrideProxyUrl":"http://other"}',
        '{"github.copilot.advanced.debug.overrideCapiUrl":"http://other"}',
    ):
        path.write_text(original, encoding="utf-8")
        with pytest.raises(click.ClickException, match="did not overwrite|refusing"):
            configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
        assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_configure_and_remove_preserve_windows_line_endings_and_bom(
    tmp_path: Path, prefix: bytes
) -> None:
    path = tmp_path / "settings.json"
    original = prefix + b'{\r\n\t"editor.fontSize": 15\r\n}\r\n'
    path.write_bytes(original)

    configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
    assert path.read_bytes().startswith(prefix + b"{\r\n")
    assert remove_vscode_proxy_settings(path) is True
    assert path.read_bytes() == original


def test_configure_refuses_duplicate_managed_markers(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = (
        "{\n"
        "// --- Headroom Copilot proxy ---\n"
        "// --- end Headroom Copilot proxy ---\n"
        "// --- Headroom Copilot proxy ---\n"
        "// --- end Headroom Copilot proxy ---\n"
        "}\n"
    )
    path.write_text(original, encoding="utf-8")

    with pytest.raises(click.ClickException, match="marker block"):
        configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
    assert path.read_text(encoding="utf-8") == original


def _write_profiles(tmp_path: Path, state: object) -> tuple[dict[str, str], Path]:
    env = {"APPDATA": str(tmp_path)}
    user_dir = vscode_user_dir(platform="win32", environ=env)
    storage = user_dir / "globalStorage" / "storage.json"
    storage.parent.mkdir(parents=True)
    storage.write_text(state if isinstance(state, str) else json.dumps(state), encoding="utf-8")
    return env, user_dir


def test_unrouted_profiles_lists_only_profiles_with_their_own_settings(tmp_path: Path) -> None:
    env, user_dir = _write_profiles(
        tmp_path,
        {
            "userDataProfiles": [
                {"location": "6c87cdb4", "name": "Work"},
                {"location": "-1f2e3d", "name": "Negative hash"},
                {"location": "aa11", "name": "Shared", "useDefaultFlags": {"settings": True}},
                {"location": "../escape", "name": "Traversal"},
                {"location": {"scheme": "vscode-remote", "path": "/p"}, "name": "Remote"},
                "not-a-profile",
            ]
        },
    )

    assert unrouted_vscode_profiles(platform="win32", environ=env) == [
        ("Work", user_dir / "profiles" / "6c87cdb4" / "settings.json"),
        ("Negative hash", user_dir / "profiles" / "-1f2e3d" / "settings.json"),
    ]


def test_unrouted_profiles_skips_a_profile_already_configured(tmp_path: Path) -> None:
    env, user_dir = _write_profiles(
        tmp_path, {"userDataProfiles": [{"location": "6c87cdb4", "name": "Work"}]}
    )
    profile_settings = user_dir / "profiles" / "6c87cdb4" / "settings.json"
    profile_settings.parent.mkdir(parents=True)
    configure_vscode_proxy_settings(profile_settings, "http://127.0.0.1:8787")

    assert unrouted_vscode_profiles(platform="win32", environ=env) == []


@pytest.mark.parametrize("state", ["{not json", [], {"userDataProfiles": {"a": 1}}, {}])
def test_unrouted_profiles_tolerates_unexpected_vscode_state(tmp_path: Path, state: object) -> None:
    env, _ = _write_profiles(tmp_path, state)

    assert unrouted_vscode_profiles(platform="win32", environ=env) == []


def test_unrouted_profiles_without_vscode_state(tmp_path: Path) -> None:
    assert unrouted_vscode_profiles(platform="win32", environ={"APPDATA": str(tmp_path)}) == []
