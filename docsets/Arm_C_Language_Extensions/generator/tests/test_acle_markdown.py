from __future__ import annotations

from pathlib import Path

from arm_acle_docset.sources.acle_markdown import (
    _macro_expression,
    parse_acle_markdown,
    parse_acle_markdown_file,
    to_enrichment_records,
    to_ir_records,
)


FIXTURES = Path(__file__).parent / "fixtures" / "acle"
SOURCE_COMMIT = "62d9cbd68abb6d18dd8f06980da7758d9dbe0560"


def _records(name: str) -> list[dict]:
    parsed = parse_acle_markdown_file(
        FIXTURES / name,
        source_commit=SOURCE_COMMIT,
    )
    return parsed["records"]


def _record(records: list[dict], name: str) -> dict:
    return next(item for item in records if item["names"]["explicit"] == name)


def _requirement_macros(expression: dict) -> set[str]:
    macros: set[str] = set()
    if "macro" in expression:
        macros.add(expression["macro"])
    for child in expression.get("args", []):
        macros.update(_requirement_macros(child))
    return macros


def test_general_intrinsics_inherit_release_and_structured_availability() -> None:
    records = _records("general_crc.md")

    assert {item["names"]["explicit"] for item in records} == {
        "__crc32b",
        "__crc32d",
    }
    record = _record(records, "__crc32b")
    assert record["family"] == ["general"]
    assert record["maturity"]["support_level"] == "release"
    assert record["maturity"]["status"] == "inherited"
    assert record["header"] == [{"name": "arm_acle.h", "status": "explicit"}]
    assert _requirement_macros(record["availability"]["expression"]) == {
        "__ARM_FEATURE_CRC32"
    }
    assert record["availability"]["minimum_architecture"] == [
        {"op": "architecture_min", "version": "8", "profile": "A"}
    ]
    assert record["availability"]["execution_states"] == ["AArch32", "AArch64"]
    assert "Performs a CRC-32 checksum" in record["semantics"]


def test_sve_pattern_expansion_maturity_and_variant_diagnostic() -> None:
    record = _record(_records("sve_sme.md"), "svluti6_s8_x2")

    assert record["family"] == ["sve2.3"]
    assert record["names"]["overloaded"] == ["svluti6"]
    assert record["maturity"]["support_level"] == "alpha"
    assert record["header"] == [{"name": "arm_sve.h", "status": "explicit"}]
    assert _requirement_macros(record["availability"]["expression"]) == {
        "__ARM_FEATURE_SVE2p3"
    }
    assert record["variant_origin"] == "unresolved"
    assert record["diagnostics"][0]["code"] == "unexpanded_variant_prose"


def test_variant_only_requirements_do_not_pollute_exemplar_availability() -> None:
    parsed = parse_acle_markdown(
        """
# SME2 intrinsics

## Multi-vector min/max

``` c
// Only if __ARM_FEATURE_SME_TMOP != 0
// Variants are also available for:
//   _bf16_x2 (only if __ARM_FEATURE_SVE_B16B16 != 0)
//   and also _s16_x2 if __ARM_FEATURE_SVE2p3 != 0
svfloat16x2_t svmax[_f16_x2](svfloat16x2_t zdn, svfloat16x2_t zm)
  __arm_streaming;
```
""",
        source_commit=SOURCE_COMMIT,
    )

    record = _record(parsed["records"], "svmax_f16_x2")

    assert _requirement_macros(record["availability"]["expression"]) == {
        "__ARM_FEATURE_SME_TMOP"
    }
    assert record["variant_origin"] == "unresolved"
    assert any(
        diagnostic["code"] == "unexpanded_variant_prose"
        for diagnostic in record["diagnostics"]
    )


def test_pinned_variant_intro_spellings_start_a_variant_only_suffix() -> None:
    parsed = parse_acle_markdown(
        """
# SME2 intrinsics

## Variant spellings

``` c
// Only if __ARM_FEATURE_SME2 != 0
// Variant also available for _f16 if __ARM_FEATURE_SME_F16F16 != 0.
svbfloat16_t svvariant_one[_bf16](svbfloat16_t zn);

// Only if __ARM_FEATURE_SME2 != 0
// Variant for _u8 is available if __ARM_FEATURE_SVE2p3 != 0.
svint8_t svvariant_two[_s8](svint8_t zn);

// Only if __ARM_FEATURE_SME2 != 0
// Available variants are: _za16 if __ARM_FEATURE_SME_F8F16 != 0
//                         _za32 if __ARM_FEATURE_SME_F8F32 != 0
void svvariant_three[_za32](svfloat32_t zn);
```
""",
        source_commit=SOURCE_COMMIT,
    )

    for name in (
        "svvariant_one_bf16",
        "svvariant_two_s8",
        "svvariant_three_za32",
    ):
        record = _record(parsed["records"], name)
        assert _requirement_macros(record["availability"]["expression"]) == {
            "__ARM_FEATURE_SME2"
        }
        assert record["variant_origin"] == "unresolved"


def test_sme_attributes_semantics_and_instruction_group_are_preserved() -> None:
    records = _records("sve_sme.md")
    support = _record(records, "__arm_za_disable")
    intrinsic = _record(records, "svmopa_za32_f32_m")

    assert support["kind"] == "support_function"
    assert "turns ZA off" in support["semantics"]
    assert intrinsic["family"] == ["sme"]
    assert intrinsic["maturity"]["support_level"] == "beta"
    assert intrinsic["names"]["overloaded"] == ["svmopa_za32_m"]
    assert intrinsic["state"] == [
        {"state": "za", "mode": "inout"},
        {"state": "zt0", "mode": "preserves"},
    ]
    assert {item["mnemonics"][0] for item in intrinsic["instructions"]} >= {"BFMOPA"}
    assert intrinsic["availability"]["expression"] == {
        "op": "calling_context",
        "values": ["streaming"],
    }


def test_mode_specific_availability_remains_separate() -> None:
    record = _record(_records("sve_sme.md"), "svclamp_s32")

    assert record["family"] == ["sve2.1", "sme2"]
    assert _requirement_macros(record["availability"]["by_mode"]["non_streaming"]) == {
        "__ARM_FEATURE_SVE2p1"
    }
    assert _requirement_macros(record["availability"]["by_mode"]["streaming"]) == {
        "__ARM_FEATURE_SME2"
    }
    assert {item["name"] for item in record["header"]} == {
        "arm_sve.h",
        "arm_sme.h",
    }
    assert record["availability"]["expression"]["op"] == "any"


def test_sve_mapping_rows_defer_isa_family_to_tablegen_guards() -> None:
    parsed = parse_acle_markdown(
        """
# SVE language extensions and intrinsics

### Mapping of SVE instructions to intrinsics

| **Instruction** | **Intrinsic** |
| --------------- | ------------- |
| SADDLB | [`svaddlb`](https://example.invalid/?q=svaddlb) |
""",
        source_commit=SOURCE_COMMIT,
    )

    patch = next(
        item
        for item in to_enrichment_records(parsed)
        if item["match"]["base_names"] == ["svaddlb"]
    )

    assert patch["family"] == ["sve"]


def test_macro_expression_preserves_mixed_boolean_precedence() -> None:
    expression, diagnostic = _macro_expression(
        "Available when `(__ARM_FEATURE_SVE2 && __ARM_FEATURE_FP8DOT2) || "
        "__ARM_FEATURE_SSVE_FP8DOT2`.",
        (
            "__ARM_FEATURE_SVE2",
            "__ARM_FEATURE_FP8DOT2",
            "__ARM_FEATURE_SSVE_FP8DOT2",
        ),
    )

    assert diagnostic is None
    assert expression["op"] == "any"
    assert any(child["op"] == "all" for child in expression["args"])

    expression, diagnostic = _macro_expression(
        "Available when `__ARM_FEATURE_FP8 && (__ARM_FEATURE_SVE2 || "
        "__ARM_FEATURE_SME2)`.",
        (
            "__ARM_FEATURE_FP8",
            "__ARM_FEATURE_SVE2",
            "__ARM_FEATURE_SME2",
        ),
    )

    assert diagnostic is None
    assert expression["op"] == "all"
    assert any(child["op"] == "any" for child in expression["args"])


def test_ambiguous_macro_prose_is_retained_as_raw_with_diagnostic() -> None:
    sentence = "Available when __ARM_FEATURE_SVE2 and/or __ARM_FEATURE_SME2 is defined."
    expression, diagnostic = _macro_expression(
        sentence,
        ("__ARM_FEATURE_SVE2", "__ARM_FEATURE_SME2"),
    )

    assert expression == {"op": "raw", "text": sentence}
    assert diagnostic is not None


def test_pinned_fp8_prose_groups_sve2_or_sme2_as_one_clause() -> None:
    expression, diagnostic = _macro_expression(
        "Available when __ARM_FEATURE_FP8 is defined, and "
        "__ARM_FEATURE_SVE2 or __ARM_FEATURE_SME2 is defined.",
        (
            "__ARM_FEATURE_FP8",
            "__ARM_FEATURE_SVE2",
            "__ARM_FEATURE_SME2",
        ),
    )

    assert diagnostic is None
    assert expression == {
        "op": "all",
        "args": [
            {"op": "defined", "macro": "__ARM_FEATURE_FP8"},
            {
                "op": "any",
                "args": [
                    {"op": "defined", "macro": "__ARM_FEATURE_SVE2"},
                    {"op": "defined", "macro": "__ARM_FEATURE_SME2"},
                ],
            },
        ],
    }


def test_pinned_faminmax_prose_groups_either_clause_before_and() -> None:
    expression, diagnostic = _macro_expression(
        "Available when either __ARM_FEATURE_SVE2 or __ARM_FEATURE_SME2 is "
        "defined, and __ARM_FEATURE_FAMINMAX is defined.",
        (
            "__ARM_FEATURE_SVE2",
            "__ARM_FEATURE_SME2",
            "__ARM_FEATURE_FAMINMAX",
        ),
    )

    assert diagnostic is None
    assert expression == {
        "op": "all",
        "args": [
            {
                "op": "any",
                "args": [
                    {"op": "defined", "macro": "__ARM_FEATURE_SVE2"},
                    {"op": "defined", "macro": "__ARM_FEATURE_SME2"},
                ],
            },
            {"op": "defined", "macro": "__ARM_FEATURE_FAMINMAX"},
        ],
    }


def test_examples_are_ignored_and_missing_default_is_unspecified() -> None:
    records = _records("edge_cases.md")

    assert {item["names"]["explicit"] for item in records} == {
        "__arm_fpm_init",
        "__unnamed_parameters",
        "svsample_s32",
    }
    record = _record(records, "svsample_s32")
    assert record["maturity"] == {
        "support_level": "unspecified",
        "status": "unresolved",
        "source_line": None,
    }
    assert record["diagnostics"][0]["code"] == "unexpanded_variant_prose"

    unnamed = _record(records, "__unnamed_parameters")
    assert unnamed["signature"]["parameters"] == [
        {"name": None, "type": "uint64_t", "constraints": []},
        {"name": None, "type": "unsigned int", "constraints": []},
    ]


def test_conversion_boundary_can_parse_text_directly() -> None:
    markdown = (FIXTURES / "general_crc.md").read_text(encoding="utf-8")
    records = to_ir_records(markdown, source_commit=SOURCE_COMMIT)

    record = next(item for item in records if item.name == "__crc32b")
    assert record.sources[0].commit == SOURCE_COMMIT
    assert record.compilation.feature_macros == ("__ARM_FEATURE_CRC32",)
    assert record.compilation.architecture_min == "Armv8-A"
    assert record.compilation.execution_states == ("AArch32", "AArch64")


def test_sve_mapping_without_signature_is_an_enrichment_not_a_fake_callable() -> None:
    parsed = parse_acle_markdown_file(
        FIXTURES / "sve_sme.md",
        source_commit=SOURCE_COMMIT,
    )
    patches = to_enrichment_records(parsed)
    add = next(item for item in patches if item["match"]["base_names"] == ["svadd"])
    asr = next(item for item in patches if item["match"]["base_names"] == ["svasr"])

    assert add["source_signature"] is None
    assert (
        add["diagnostics"][0]["code"] == "signature_missing_use_declaration_inventory"
    )
    assert add["instructions"][0]["relation"] == "group"
    assert asr["instructions"][0]["relation"] == "optimizer_candidate"


def test_conversion_boundary_requires_commit_for_text() -> None:
    try:
        to_ir_records("# Empty")
    except ValueError as error:
        assert "source_commit" in str(error)
    else:
        raise AssertionError("expected source_commit validation")


def test_function_multiversion_attributes_are_not_parsed_as_acle_callables() -> None:
    parsed = parse_acle_markdown(
        """
# Function Multi Versioning

```cpp
int __attribute__((target_version("simd"))) f(int value);
int __attribute__((target_version("default"))) f(int value);
```
""",
        source_commit=SOURCE_COMMIT,
    )

    assert parsed["records"] == []
