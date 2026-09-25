# SPDX-License-Identifier: Apache-2.0
"""GWM advice must never append a trailing system turn.

Qwen3.x chat templates reject a system message that is not the first turn and
fail the request with ``System message must be at the beginning``, which
silently disables the whole advice arm for those policies.
"""

import pathlib
import typing

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src/vllm_gwm/serving.py"


def _load():
    """Exec just the helper; importing serving.py pulls in heavy vLLM deps."""
    src = _SRC.read_text()
    start = src.index("def _with_guidance")
    end = src.index("\nclass ", start)
    # The helper is annotated with names imported at module scope; seed them
    # rather than importing serving.py, which pulls in heavy vLLM deps.
    ns: dict = {"Any": typing.Any}
    exec(src[start:end], ns)  # noqa: S102
    return ns["_with_guidance"]


@pytest.fixture(scope="module")
def with_guidance():
    return _load()


def test_merges_into_leading_system_message(with_guidance):
    out = with_guidance(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], "G")
    assert out[0] == {"role": "system", "content": "S\n\nG"}
    assert out[1] == {"role": "user", "content": "U"}


def test_inserts_system_first_when_absent(with_guidance):
    out = with_guidance([{"role": "user", "content": "U"}], "G")
    assert out[0] == {"role": "system", "content": "G"}
    assert out[1]["role"] == "user"


def test_empty_conversation(with_guidance):
    assert with_guidance([], "G") == [{"role": "system", "content": "G"}]


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
    [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}],
    [],
])
def test_system_turn_is_always_first_and_unique(with_guidance, messages):
    out = with_guidance(messages, "G")
    roles = [m["role"] for m in out]
    assert roles.count("system") <= 1
    if "system" in roles:
        assert roles.index("system") == 0


def test_does_not_mutate_caller_messages(with_guidance):
    original = [{"role": "system", "content": "S"}]
    with_guidance(original, "G")
    assert original == [{"role": "system", "content": "S"}]
