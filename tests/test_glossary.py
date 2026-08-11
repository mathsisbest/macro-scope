"""Unit tests for dashboard/components/glossary.py — the educational tooltip helpers.

Primary concern: the glossary is the single source of truth for domain-concept
explanations, so (1) every entry must be present, non-empty, and reasonably concise,
(2) lookups must be case/separator-insensitive so "OOS R²" and "oos_r2" resolve to the
same definition, (3) typos must fail loudly (KeyError, not a silent empty tooltip), and
(4) the pure render helpers must produce well-formed, HTML-escaped markdown — since the
tooltip text is injected into an HTML title attribute, an unescaped quote would break it.
"""

from __future__ import annotations

import html

import pytest
from dashboard.components.glossary import (
    GLOSSARY,
    TITLES,
    definition,
    glossary_markdown,
    known_terms,
    title,
    tooltip_markdown,
)

# ---------------------------------------------------------------------------
# Glossary content contract
# ---------------------------------------------------------------------------


def test_glossary_is_non_empty_and_sorted() -> None:
    assert len(GLOSSARY) >= 10
    assert known_terms() == sorted(GLOSSARY)  # slugs in sorted order, no duplicates


@pytest.mark.parametrize("slug", GLOSSARY)
def test_every_entry_has_title_and_concise_definition(slug: str) -> None:
    body = definition(slug)
    assert body.strip(), f"{slug} has an empty definition"
    assert 40 <= len(body) <= 600, f"{slug} definition is not a short 2-3 sentence blurb"
    assert slug in TITLES, f"{slug} is missing a human-readable TITLES entry"
    assert title(slug).strip(), f"{slug} has an empty title"


def test_definitions_cover_the_curated_topic_set() -> None:
    # The roadmap's suggested concepts must all be explainable via the shared glossary.
    for term in ["oos_r2", "cape", "vol_regime", "bootstrap_ci", "risk_parity", "vol_rich_plus"]:
        assert term in GLOSSARY


def test_definitions_are_honestly_framed() -> None:
    # Honesty rule: definitions are plain-language blurbs that never promise future
    # results. Any "guarantee" claim must be explicitly negated ("does not guarantee").
    for slug, body in GLOSSARY.items():
        assert "will beat" not in body and "always " not in body, f"{slug} overpromises"
        assert "guarantee" not in body or "does not guarantee" in body or "not guarantee" in body, (
            f"{slug} reads like a promise: {body!r}"
        )
        assert "<" not in body and "**" not in body, f"{slug} contains markdown/HTML artifacts"


# ---------------------------------------------------------------------------
# Lookup: case/separator-insensitive, loud on unknowns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    ["oos_r2", "OOS R²", "oos r2", "Oos-R2", "  OOS R²  ", "OOS.R2"],
)
def test_definition_is_case_and_separator_insensitive(query: str) -> None:
    assert definition(query) == GLOSSARY["oos_r2"]


@pytest.mark.parametrize("bad", ["", "  ", "not_a_term", "sharpe_ratio_xyz", "OOS R²!!"])
def test_unknown_term_raises_keyerror_with_helpful_message(bad: str) -> None:
    with pytest.raises(KeyError, match="Unknown glossary term"):
        definition(bad)


def test_title_falls_back_to_raw_slug() -> None:
    assert title("oos_r2") == "OOS R²"
    assert title("no_such_term") == "no_such_term"


# ---------------------------------------------------------------------------
# Pure render helpers
# ---------------------------------------------------------------------------


def test_tooltip_markdown_binds_definition_to_title_attribute() -> None:
    out = tooltip_markdown("oos_r2")
    assert "**OOS R²**" in out
    assert "class='glossary-q'" in out
    assert ">?</span>" in out
    assert html.escape(GLOSSARY["oos_r2"], quote=True) in out


def test_tooltip_markdown_label_override() -> None:
    out = tooltip_markdown("cape", label="CAPE ratio")
    assert "**CAPE ratio**" in out
    assert "**Shiller CAPE**" not in out


def test_tooltip_markdown_escapes_quotes_in_definition() -> None:
    # The definition lands inside an HTML title="…" attribute: an unescaped quote would
    # truncate the tooltip. "ic" contains an apostrophe ("the model's forecast") which
    # must be HTML-escaped, and no raw quote may appear inside the attribute value.
    out = tooltip_markdown("ic")
    attr = out.split('title="', 1)[1].rsplit('">', 1)[0]
    assert '"' not in attr
    assert html.escape(GLOSSARY["ic"], quote=True) == attr


def test_glossary_markdown_lists_every_term() -> None:
    out = glossary_markdown()
    for slug in known_terms():
        assert f"**{title(slug)}**" in out
        assert GLOSSARY[slug][:30] in out  # definition body is present
    assert len(out.splitlines()) == len(GLOSSARY)


def test_glossary_markdown_filters_and_validates() -> None:
    out = glossary_markdown(["cape", "oos R2"])
    assert out.count("**") == 4  # two bold terms → two "**" pairs
    assert "Shiller CAPE" in out and "OOS R²" in out
    with pytest.raises(KeyError, match="Unknown glossary term"):
        glossary_markdown(["cape", "typo_term"])
