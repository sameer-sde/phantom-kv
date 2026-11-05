"""first_divergence: where two run traces split (used by `phantomkv vllm-diff`)."""

from phantomkv.cli import first_divergence


def t(*keys):
    return [(k, "detail") for k in keys]


def test_identical_traces():
    assert first_divergence(t("a", "b"), t("a", "b")) is None


def test_details_are_ignored():
    assert first_divergence([("a", "blocks=(1,)")], [("a", "blocks=(7,)")]) is None


def test_first_differing_key():
    assert first_divergence(t("a", "b", "c"), t("a", "x", "c")) == 1


def test_one_trace_is_a_prefix_of_the_other():
    assert first_divergence(t("a", "b"), t("a", "b", "c")) == 2
