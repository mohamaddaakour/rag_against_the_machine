"""Unit tests for identifier-aware tokenisation."""

from src.analysis import analyze, path_tokens, split_identifier


def test_split_identifier_handles_snake_and_camel() -> None:
    assert split_identifier("fused_batched_moe") == ["fused", "batched", "moe"]
    assert split_identifier("getModelConfig") == ["get", "Model", "Config"]
    assert split_identifier("HTTPServerError") == ["HTTP", "Server", "Error"]
    assert split_identifier("plain") == ["plain"]


def test_analyze_keeps_the_identifier_and_its_parts() -> None:
    tokens = list(analyze("fused_batched_moe"))
    assert "fused_batched_moe" in tokens
    assert {"fused", "batched", "moe"} <= set(tokens)


def test_analyze_lowercases_and_drops_single_characters() -> None:
    assert list(analyze("A b CD")) == ["cd"]


def test_analyze_matches_a_question_phrasing_to_an_identifier() -> None:
    question = set(analyze("What does the fused batched MoE layer return?"))
    code = set(analyze("def fused_batched_moe(x):"))
    assert {"fused", "batched", "moe"} <= question & code


def test_path_tokens_expose_module_names() -> None:
    tokens = set(
        path_tokens("data/raw/vllm-0.10.1/vllm/attention/triton_flash.py")
        .split()
    )
    assert {"attention", "triton", "flash", "vllm", "py"} <= tokens
