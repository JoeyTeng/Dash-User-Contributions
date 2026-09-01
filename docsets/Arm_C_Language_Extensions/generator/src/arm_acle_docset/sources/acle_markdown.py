"""Parse callable metadata from the Arm ACLE Markdown specification.

The ACLE main specification is intentionally human-authored Markdown rather
than a callable database.  This adapter therefore builds a small structural
tree first and only then extracts facts from bounded section contexts.  It
does not expand natural-language variant lists: such lists are retained as
diagnostics for a separate, audited expansion layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Sequence, cast

from ..model import AvailabilityExpr, AvailabilityOp, ComparisonOperator
from ..normalize import parse_availability_guard


ACLE_REPOSITORY = "ARM-software/acle"
ACLE_MARKDOWN_LICENSE = "CC-BY-SA-4.0 AND Apache-Patent-License"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})\s*(?P<info>[^`]*)$")
_CALLABLE_NAME_RE = re.compile(
    r"(?P<name>(?:__arm_|__|sv)[A-Za-z0-9_]*(?:\[[^\]\n]+\][A-Za-z0-9_]*)*)\s*\("
)
_BRACKET_SEGMENT_RE = re.compile(r"\[([^\]]+)\]")
_ATTRIBUTE_RE = re.compile(
    r"__arm_(?:agnostic|in|inout|locally_streaming|new|out|preserves|"
    r"streaming|streaming_compatible)\b(?:\s*\([^)]*\))?"
)
_ATTRIBUTE_NAMES = {
    "__attribute__",
    "__arm_agnostic",
    "__arm_in",
    "__arm_inout",
    "__arm_locally_streaming",
    "__arm_new",
    "__arm_out",
    "__arm_preserves",
    "__arm_streaming",
    "__arm_streaming_compatible",
}
_STATE_ATTRIBUTE_RE = re.compile(
    r"__arm_(?P<mode>in|inout|new|out|preserves)\s*\((?P<states>[^)]*)\)"
)
_MACRO_RE = re.compile(r"\b__ARM(?:_FEATURE)?_[A-Za-z0-9_]+\b|\b__ARM_FP\b")
_ARCH_RE = re.compile(r"\bArmv(?P<version>\d+(?:\.\d+)?)(?:-(?P<profile>[AMR]))?\b")
_EXECUTION_STATE_RE = re.compile(r"\bAArch(?:32|64)\b")
_FEAT_RE = re.compile(r"\bFEAT_[A-Za-z0-9_]+\b")
_HEADER_RE = re.compile(r"<(?P<header>arm_[a-z0-9_]+\.h)>")
_MATURITY_RE = re.compile(
    r"(?:"
    r"\[\*\*(Release|Beta|Alpha)\*\*\s+(?:quality level|state)\]\([^)]*\)"
    r"|\[\*\*(Release|Beta|Alpha)\*\*\]\([^)]*\)\s+(?:quality level|state)"
    r"|\*\*(Release|Beta|Alpha)\*\*\s+(?:quality level|state)"
    r")",
    re.IGNORECASE,
)
_VARIANT_PROSE_RE = re.compile(
    r"\b(?:"
    r"variants?\s+(?:(?:are|is)\s+)?(?:also\s+)?available"
    r"|variants?\s+for\b.+?\b(?:are|is)\s+available"
    r"|available\s+variants?\s+are"
    r"|and similarly for"
    r"|replac(?:e|ing) .+? with"
    r"|the same (?:prototype|intrinsic).+?for"
    r")\b",
    re.IGNORECASE,
)
_REQUIREMENT_CUE_RE = re.compile(
    r"\b(?:available|availability|defined|only if|only when|provided|"
    r"requires?|introduced in|execution state|target|header file|includ(?:e|ed|ing))\b",
    re.IGNORECASE,
)
_EXAMPLE_HEADING_RE = re.compile(r"\bexamples?\b", re.IGNORECASE)


@dataclass(slots=True)
class _TextBlock:
    lines: list[str]
    start_line: int
    end_line: int

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass(slots=True)
class _CodeBlock:
    lines: list[str]
    info: str
    start_line: int
    end_line: int


_Block = _TextBlock | _CodeBlock


@dataclass(slots=True)
class _Section:
    title: str
    level: int
    start_line: int
    end_line: int
    parent: _Section | None = None
    blocks: list[_Block] = field(default_factory=list)
    children: list[_Section] = field(default_factory=list)

    def ancestors(self, *, include_self: bool = True) -> list[_Section]:
        result: list[_Section] = []
        current: _Section | None = self if include_self else self.parent
        while current is not None:
            result.append(current)
            current = current.parent
        result.reverse()
        return result


def parse_acle_markdown_file(
    path: str | Path,
    *,
    source_commit: str,
    source_path: str = "main/acle.md",
) -> dict[str, Any]:
    """Parse an ACLE Markdown file and return records plus diagnostics."""

    markdown = Path(path).read_text(encoding="utf-8")
    return parse_acle_markdown(
        markdown,
        source_commit=source_commit,
        source_path=source_path,
    )


def parse_acle_markdown(
    markdown: str,
    *,
    source_commit: str,
    source_path: str = "main/acle.md",
) -> dict[str, Any]:
    """Parse supported callable facts from ACLE Markdown.

    The returned mapping deliberately stays adapter-local.  ``to_ir_records``
    is the single conversion boundary used by the generator's canonical IR.
    """

    root = _parse_markdown_tree(markdown)
    default_maturity = _document_default_maturity(root)
    named_semantics = _collect_named_semantics(root)
    records: list[dict[str, Any]] = []
    document_diagnostics: list[dict[str, Any]] = []

    for section in _walk_sections(root):
        if section is root or _is_example_section(section):
            continue
        code_blocks = [
            block for block in section.blocks if isinstance(block, _CodeBlock)
        ]
        if not code_blocks:
            continue

        first_code_index = next(
            index
            for index, block in enumerate(section.blocks)
            if isinstance(block, _CodeBlock)
        )
        section_intro = [
            block
            for block in section.blocks[:first_code_index]
            if isinstance(block, _TextBlock)
        ]

        for block in code_blocks:
            if block.info and block.info.lower() not in {"c", "cpp", "c++", ""}:
                continue
            following_text = _following_text_blocks(section.blocks, block)
            declarations = _extract_declarations(block)
            if not declarations:
                continue

            for declaration in declarations:
                context_blocks: list[_TextBlock] = [*section_intro, *following_text]
                record = _record_from_declaration(
                    declaration,
                    section=section,
                    context_blocks=context_blocks,
                    default_maturity=default_maturity,
                    named_semantics=named_semantics,
                    source_commit=source_commit,
                    source_path=source_path,
                )
                records.append(record)

    records, duplicate_diagnostics = _deduplicate_records(records)
    document_diagnostics.extend(duplicate_diagnostics)
    enrichments = _parse_instruction_mapping_enrichments(
        root,
        default_maturity=default_maturity,
        source_commit=source_commit,
        source_path=source_path,
    )
    return {
        "records": records,
        "enrichments": enrichments,
        "diagnostics": document_diagnostics,
        "source": {
            "repository": ACLE_REPOSITORY,
            "commit": source_commit,
            "path": source_path,
            "license": ACLE_MARKDOWN_LICENSE,
        },
    }


def to_ir_records(
    parsed_or_markdown: dict[str, Any] | str,
    *,
    source_commit: str | None = None,
    source_path: str = "main/acle.md",
    families: Iterable[str] = ("general",),
) -> list[Any]:
    """Convert explicit Markdown declarations to canonical callables.

    The default deliberately converts only general ACLE declarations.  SVE
    and SME declaration identity comes from the pinned LLVM inventory; use
    :func:`to_enrichment_records` to merge Markdown semantics and metadata
    into those callables.  Callers may opt additional families in when they
    need standalone source comparison records.
    """

    if isinstance(parsed_or_markdown, str):
        if not source_commit:
            raise ValueError("source_commit is required when parsing Markdown text")
        parsed = parse_acle_markdown(
            parsed_or_markdown,
            source_commit=source_commit,
            source_path=source_path,
        )
    else:
        parsed = parsed_or_markdown

    payloads = parsed.get("records")
    if not isinstance(payloads, list):
        raise TypeError("parsed ACLE Markdown must contain a records list")
    included = set(families)
    records = []
    for payload in payloads:
        blocker = _canonical_signature_blocker(payload)
        if blocker:
            diagnostic = {
                "code": "canonical_signature_unresolved",
                "message": blocker,
                "source": payload["provenance"]["source"],
            }
            if diagnostic not in parsed.setdefault("diagnostics", []):
                parsed["diagnostics"].append(diagnostic)
            continue
        payload_families = payload["family"]
        selected = [family for family in payload_families if family in included]
        for family in selected:
            records.append(_payload_to_concrete_callable(payload, family=family))
    return records


def to_enrichment_records(
    parsed_or_markdown: dict[str, Any] | str,
    *,
    source_commit: str | None = None,
    source_path: str = "main/acle.md",
) -> list[dict[str, Any]]:
    """Return name-matchable metadata patches for the declaration inventory."""

    if isinstance(parsed_or_markdown, str):
        if not source_commit:
            raise ValueError("source_commit is required when parsing Markdown text")
        parsed = parse_acle_markdown(
            parsed_or_markdown,
            source_commit=source_commit,
            source_path=source_path,
        )
    else:
        parsed = parsed_or_markdown

    patches: list[dict[str, Any]] = list(parsed.get("enrichments", []))
    for payload in parsed.get("records", []):
        if payload["family"] == ["general"]:
            continue
        names = [payload["names"]["explicit"], *payload["names"]["overloaded"]]
        patches.append(
            {
                "match": {"names": names, "base_names": []},
                "family": payload["family"],
                "header": payload["header"],
                "availability": payload["availability"],
                "maturity": payload["maturity"],
                "semantics": payload["semantics"],
                "instructions": payload["instructions"],
                "state": payload["state"],
                "taxonomy_path": payload["taxonomy_path"],
                "source_signature": payload["signature"],
                "provenance": payload["provenance"],
                "diagnostics": payload["diagnostics"],
            }
        )
    return patches


def _parse_markdown_tree(markdown: str) -> _Section:
    lines = markdown.splitlines()
    root = _Section(title="", level=0, start_line=1, end_line=max(1, len(lines)))
    stack: list[_Section] = [root]
    text_lines: list[str] = []
    text_start = 1

    def flush_text(end_line: int) -> None:
        nonlocal text_lines, text_start
        if text_lines and any(line.strip() for line in text_lines):
            stack[-1].blocks.append(
                _TextBlock(text_lines, text_start, max(text_start, end_line))
            )
        text_lines = []

    index = 0
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        fence_match = _FENCE_RE.match(_strip_blockquote_prefix(line))
        if fence_match:
            flush_text(line_number - 1)
            fence = fence_match.group("fence")
            info_text = fence_match.group("info").strip()
            info = info_text.split(maxsplit=1)[0] if info_text else ""
            code_lines: list[str] = []
            start_line = line_number
            index += 1
            while index < len(lines):
                candidate = _strip_blockquote_prefix(lines[index])
                if re.match(
                    rf"^\s*{re.escape(fence[0])}{{{len(fence)},}}\s*$", candidate
                ):
                    break
                code_lines.append(candidate)
                index += 1
            end_line = min(len(lines), index + 1)
            stack[-1].blocks.append(
                _CodeBlock(code_lines, info.lower(), start_line, end_line)
            )
            index += 1
            text_start = index + 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_text(line_number - 1)
            level = len(heading_match.group(1))
            while stack[-1].level >= level:
                stack[-1].end_line = line_number - 1
                stack.pop()
            section = _Section(
                title=_plain_heading(heading_match.group(2)),
                level=level,
                start_line=line_number,
                end_line=max(line_number, len(lines)),
                parent=stack[-1],
            )
            stack[-1].children.append(section)
            stack.append(section)
            index += 1
            text_start = index + 1
            continue

        if not text_lines:
            text_start = line_number
        text_lines.append(line)
        index += 1

    flush_text(len(lines))
    while len(stack) > 1:
        stack[-1].end_line = len(lines)
        stack.pop()
    return root


def _walk_sections(root: _Section) -> Iterator[_Section]:
    yield root
    for child in root.children:
        yield from _walk_sections(child)


def _strip_blockquote_prefix(line: str) -> str:
    stripped = line
    while re.match(r"^\s*>\s?", stripped):
        stripped = re.sub(r"^\s*>\s?", "", stripped, count=1)
    return stripped


def _plain_heading(title: str) -> str:
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[`*_]", "", title)
    return re.sub(r"\s+", " ", title).strip()


def _document_default_maturity(root: _Section) -> tuple[str, int | None]:
    for section in _walk_sections(root):
        if section.title.lower() != "support levels":
            continue
        text = _blocks_text(
            block for block in section.blocks if isinstance(block, _TextBlock)
        )
        match = re.search(
            r"All content in this document is at the \*\*(Release|Beta|Alpha)\*\* "
            r"quality level",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), section.start_line
    return "unspecified", None


def _explicit_section_maturity(section: _Section) -> tuple[str, int] | None:
    if section.title.lower() == "support levels":
        return None
    intro = _section_intro_blocks(section)
    for block in intro:
        match = _MATURITY_RE.search(block.text)
        if match:
            level = next(group for group in match.groups() if group is not None)
            return level.lower(), block.start_line
    return None


def _resolved_maturity(
    section: _Section,
    default: tuple[str, int | None],
) -> dict[str, Any]:
    level, line = default
    status = "inherited" if line is not None else "unresolved"
    for ancestor in section.ancestors():
        explicit = _explicit_section_maturity(ancestor)
        if explicit is not None:
            level, line = explicit
            status = "explicit" if ancestor is section else "inherited"
    return {
        "support_level": level,
        "status": status,
        "source_line": line,
    }


def _section_intro_blocks(section: _Section) -> list[_TextBlock]:
    result: list[_TextBlock] = []
    for block in section.blocks:
        if isinstance(block, _CodeBlock):
            break
        if isinstance(block, _TextBlock):
            result.append(block)
    return result


def _following_text_blocks(
    blocks: Sequence[_Block], code: _CodeBlock
) -> list[_TextBlock]:
    result: list[_TextBlock] = []
    found = False
    for block in blocks:
        if block is code:
            found = True
            continue
        if not found:
            continue
        if isinstance(block, _CodeBlock):
            break
        result.append(block)
    return result


def _section_wide_requirement_blocks(section: _Section) -> list[_TextBlock]:
    """Select late prose that explicitly scopes requirements to the section."""

    result: list[_TextBlock] = []
    scope_cue = re.compile(
        r"\b(?:these intrinsics|intrinsics in this section|to access (?:these|the) intrinsics)\b",
        re.IGNORECASE,
    )
    for block in section.blocks:
        if not isinstance(block, _TextBlock):
            continue
        selected = [
            sentence
            for sentence in _sentences(block.text)
            if scope_cue.search(sentence) and _REQUIREMENT_CUE_RE.search(sentence)
        ]
        if selected:
            result.append(_TextBlock(selected, block.start_line, block.end_line))
    return result


def _is_example_section(section: _Section) -> bool:
    return any(_EXAMPLE_HEADING_RE.search(item.title) for item in section.ancestors())


def _extract_declarations(block: _CodeBlock) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    pending_comments: list[tuple[int, str]] = []
    statement_lines: list[tuple[int, str]] = []
    in_block_comment = False

    def flush_statement() -> None:
        nonlocal statement_lines, pending_comments
        if not statement_lines:
            return
        raw = "\n".join(line for _, line in statement_lines).strip()
        line_start = statement_lines[0][0]
        line_end = statement_lines[-1][0]
        statement_lines = []
        parsed = _parse_declaration(raw)
        if parsed is None:
            return
        parsed["line_start"] = line_start
        parsed["line_end"] = line_end
        parsed["leading_comments"] = list(pending_comments)
        declarations.append(parsed)
        pending_comments = []

    for offset, original in enumerate(block.lines, start=block.start_line + 1):
        stripped = original.strip()
        if in_block_comment:
            pending_comments.append((offset, stripped))
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            pending_comments.append((offset, stripped))
            if "*/" not in stripped:
                in_block_comment = True
            continue
        if stripped.startswith("//"):
            pending_comments.append((offset, stripped[2:].strip()))
            continue
        if not stripped or stripped.startswith("#"):
            if statement_lines:
                statement_lines.append((offset, original))
            continue
        statement_lines.append((offset, original))
        if ";" in _remove_c_comments(original):
            flush_statement()

    flush_statement()
    return declarations


def _parse_declaration(raw: str) -> dict[str, Any] | None:
    normalized = _normalize_space(raw)
    if not normalized.endswith(";"):
        return None
    if (
        "{" in normalized
        or "}" in normalized
        or normalized.startswith(("typedef ", "enum "))
    ):
        return None
    code = _normalize_space(_remove_c_comments(normalized))
    matches = list(_CALLABLE_NAME_RE.finditer(code))
    if not matches:
        return None

    match = next(
        (item for item in matches if item.group("name") not in _ATTRIBUTE_NAMES),
        None,
    )
    if match is None:
        return None
    name = match.group("name")
    prefix = code[: match.start("name")].strip()
    if not prefix or prefix.startswith(("return ", "case ")) or "=" in prefix:
        return None
    if "(" in prefix or ")" in prefix:
        return None

    open_paren = code.find("(", match.start("name"))
    close_paren = _matching_paren(code, open_paren)
    if close_paren is None:
        return None
    suffix = code[close_paren + 1 :].rstrip(";").strip()
    attributes = _ATTRIBUTE_RE.findall(code)
    return_type = _ATTRIBUTE_RE.sub("", prefix).strip()
    if not return_type:
        return None

    parameters_raw = code[open_paren + 1 : close_paren].strip()
    parameters = [_parse_parameter(item) for item in _split_parameters(parameters_raw)]
    return {
        "name_pattern": name,
        "raw": normalized,
        "return_type": return_type,
        "parameters": parameters,
        "attributes": attributes,
        "suffix": suffix,
    }


def _matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_parameters(parameters: str) -> list[str]:
    if not parameters or parameters == "void":
        return []
    result: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(parameters):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    result.append(parameters[start:].strip())
    return [item for item in result if item]


def _parse_parameter(raw: str) -> dict[str, Any]:
    cleaned = _normalize_space(_remove_c_comments(raw))
    constraint = "constant_expression" if "/*constant*/" in raw else None
    if cleaned == "...":
        return {"name": None, "type": "...", "constraints": []}
    match = re.search(r"\b([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$", cleaned)
    candidate_is_type = False
    if match:
        candidate = match.group(1)
        prefix_words = cleaned[: match.start(1)].strip().split()
        type_words = {
            "_Bool",
            "char",
            "const",
            "double",
            "float",
            "int",
            "long",
            "short",
            "signed",
            "unsigned",
            "void",
            "volatile",
        }
        candidate_is_type = (
            not prefix_words
            or candidate in type_words
            and all(word in type_words for word in prefix_words)
            or candidate.endswith("_t")
            and all(
                word in {"const", "volatile", "signed", "unsigned"}
                for word in prefix_words
            )
        )
    if match and not candidate_is_type:
        name = match.group(1)
        parameter_type = (cleaned[: match.start(1)] + cleaned[match.end(1) :]).strip()
    else:
        name = None
        parameter_type = cleaned
    constraints = []
    if constraint:
        constraints.append({"kind": constraint, "raw": "/*constant*/"})
    return {"name": name, "type": parameter_type, "constraints": constraints}


def _canonical_signature_blocker(payload: dict[str, Any]) -> str | None:
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        return "The Markdown record has no structured signature."
    if not str(signature.get("return_type", "")).strip():
        return "The Markdown declaration has no parseable return type."
    for index, parameter in enumerate(signature.get("parameters", []), start=1):
        if not str(parameter.get("type", "")).strip():
            return (
                f"Parameter {index} of {payload['names']['explicit']} has no "
                "parseable type; no placeholder type was invented."
            )
    return None


def _remove_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//.*?(?=\n|$)", " ", text)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _record_from_declaration(
    declaration: dict[str, Any],
    *,
    section: _Section,
    context_blocks: Sequence[_TextBlock],
    default_maturity: tuple[str, int | None],
    named_semantics: dict[str, str],
    source_commit: str,
    source_path: str,
) -> dict[str, Any]:
    path = [item.title for item in section.ancestors() if item.title]
    families = _families_from_path(path)
    name_pattern = declaration["name_pattern"]
    explicit_name, overloaded_name = _expand_lockstep_name(name_pattern)
    signature_raw = declaration["raw"].replace(name_pattern, explicit_name, 1)
    ancestor_intro_blocks: list[_TextBlock] = []
    for ancestor in section.ancestors(include_self=False):
        ancestor_intro_blocks.extend(_section_intro_blocks(ancestor))
    all_requirement_blocks = [
        *ancestor_intro_blocks,
        *_section_wide_requirement_blocks(section),
        *context_blocks,
    ]
    comments = declaration["leading_comments"]
    comment_blocks = [
        _TextBlock([text], line, line) for line, text in comments if text.strip()
    ]
    requirement_comment_blocks = _requirement_comment_blocks(comments)
    requirements = _extract_requirements(
        [*all_requirement_blocks, *requirement_comment_blocks],
        families=families,
        attributes=declaration["attributes"],
    )
    variant_hints = _variant_hints([*context_blocks, *comment_blocks])
    diagnostics = [
        {
            "code": "unexpanded_variant_prose",
            "line": hint["line"],
            "message": hint["raw"],
        }
        for hint in variant_hints
    ]
    diagnostics.extend(requirements.pop("diagnostics"))
    semantics = named_semantics.get(explicit_name)
    if semantics is None and overloaded_name:
        semantics = named_semantics.get(overloaded_name)
    if semantics is None:
        semantics = _semantic_context(context_blocks)

    state = _state_access(declaration["attributes"])
    instructions = _instruction_relations(section, context_blocks)
    kind = _callable_kind(section, declaration)
    line_start = declaration["line_start"]
    line_end = declaration["line_end"]
    source = {
        "repository": ACLE_REPOSITORY,
        "commit": source_commit,
        "path": source_path,
        "start_line": line_start,
        "end_line": line_end,
        "license": ACLE_MARKDOWN_LICENSE,
    }
    return {
        "kind": kind,
        "family": families,
        "names": {
            "pattern": name_pattern,
            "explicit": explicit_name,
            "overloaded": [overloaded_name] if overloaded_name else [],
        },
        "signature": {
            "raw": signature_raw,
            "return_type": declaration["return_type"],
            "parameters": declaration["parameters"],
            "attributes": declaration["attributes"],
            "is_concrete": True,
        },
        "header": requirements.pop("headers"),
        "availability": requirements,
        "maturity": _resolved_maturity(section, default_maturity),
        "semantics": semantics,
        "instructions": instructions,
        "state": state,
        "variant_origin": "unresolved"
        if variant_hints
        else ("expanded_from_pattern" if overloaded_name else "explicit"),
        "variant_hints": variant_hints,
        "taxonomy_path": path,
        "provenance": {
            "source": source,
            "fields": {
                "signature": "explicit",
                "names": "expanded" if overloaded_name else "explicit",
                "maturity": _resolved_maturity(section, default_maturity)["status"],
                "availability": "inherited_or_explicit",
                "semantics": "explicit" if semantics else "unresolved",
            },
        },
        "diagnostics": diagnostics,
    }


def _requirement_comment_blocks(
    comments: Sequence[tuple[int, str]],
) -> list[_TextBlock]:
    """Keep the pre-variant declaration guard separate from variant conditions.

    The pinned ACLE source places declaration-wide guards before a variant-list
    introduction. The entire remaining comment suffix belongs to that list,
    including wrapped entries that start with prose such as ``and also``.
    """

    result: list[_TextBlock] = []
    for line, text in comments:
        normalized = _normalize_space(text.lstrip("/ "))
        if not normalized:
            continue
        if _VARIANT_PROSE_RE.search(normalized):
            break
        result.append(_TextBlock([text], line, line))
    return result


def _expand_lockstep_name(pattern: str) -> tuple[str, str | None]:
    if "[" not in pattern:
        return pattern, None
    explicit = _BRACKET_SEGMENT_RE.sub(lambda match: match.group(1), pattern)
    overloaded = _BRACKET_SEGMENT_RE.sub("", pattern)
    overloaded = re.sub(r"__+", "_", overloaded)
    return explicit, overloaded


def _families_from_path(path: Sequence[str]) -> list[str]:
    joined = " / ".join(path)
    matches: list[tuple[int, str]] = []
    for prefix in ("SVE", "SME"):
        for match in re.finditer(
            rf"\b{prefix}2(?:[._]([123]))?\b|\b{prefix}\b", joined
        ):
            token = match.group(0).lower().replace("_", ".")
            matches.append((match.start(), token))
    if not matches:
        return ["general"]
    result: list[str] = []
    for _, family in sorted(matches):
        if family not in result:
            result.append(family)
    # A versioned family is more precise than its unversioned ancestor.
    for base in ("sve", "sme"):
        if any(item.startswith(base + "2") for item in result):
            result = [item for item in result if item != base]
    return result


def _extract_requirements(
    blocks: Sequence[_TextBlock],
    *,
    families: Sequence[str],
    attributes: Sequence[str],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    by_mode: dict[str, list[dict[str, Any]]] = {}
    headers: list[dict[str, Any]] = []
    architecture: list[dict[str, Any]] = []
    execution_states: list[str] = []
    features: list[str] = []
    extensions: list[str] = []
    diagnostics: list[dict[str, Any]] = []

    for block in blocks:
        for sentence in _sentences(block.text):
            headers_in_sentence = _HEADER_RE.findall(sentence)
            for header in headers_in_sentence:
                _append_unique(headers, {"name": header, "status": "explicit"})
            for match in _ARCH_RE.finditer(sentence):
                item = {
                    "op": "architecture_min",
                    "version": match.group("version"),
                }
                if match.group("profile"):
                    item["profile"] = match.group("profile")
                _append_unique(architecture, item)
            for state in _EXECUTION_STATE_RE.findall(sentence):
                if state not in execution_states:
                    execution_states.append(state)
            for feature in _FEAT_RE.findall(sentence):
                if feature not in features:
                    features.append(feature)
            for extension in re.findall(
                r"\b([A-Z][A-Z0-9_.+-]{2,}) extensions?\b", sentence
            ):
                if extension not in extensions:
                    extensions.append(extension)

            mode = _calling_mode(sentence)
            macros = _MACRO_RE.findall(sentence)
            if macros and _REQUIREMENT_CUE_RE.search(sentence):
                expression, diagnostic = _macro_expression(sentence, macros)
                if diagnostic:
                    diagnostics.append(
                        {
                            "code": "acle.availability_expression_unparsed",
                            "line": block.start_line,
                            "message": diagnostic,
                        }
                    )
                if mode:
                    by_mode.setdefault(mode, []).append(expression)
                else:
                    nodes.append(expression)
                raw.append(
                    {"line": block.start_line, "text": _normalize_space(sentence)}
                )
            elif mode == "streaming" and re.search(
                r"appropriate SME feature macro", sentence, re.IGNORECASE
            ):
                inherited_macros = sorted(
                    macro
                    for node in nodes
                    for macro in _requirement_macros_from_payload(node)
                    if macro.startswith("__ARM_FEATURE_SME")
                )
                if inherited_macros:
                    by_mode.setdefault(mode, []).append(
                        _combine_requirements(
                            [
                                {"op": "defined", "macro": macro}
                                for macro in inherited_macros
                            ]
                        )
                    )
                    raw.append(
                        {
                            "line": block.start_line,
                            "text": _normalize_space(sentence),
                        }
                    )
            elif mode == "streaming_compatible" and re.search(
                r"called from both non-streaming code and streaming code",
                sentence,
                re.IGNORECASE,
            ):
                inherited_modes = [
                    _combine_requirements(by_mode[required_mode])
                    for required_mode in ("non_streaming", "streaming")
                    if required_mode in by_mode
                ]
                if len(inherited_modes) == 2:
                    by_mode.setdefault(mode, []).append(
                        {"op": "all", "args": inherited_modes}
                    )
                    raw.append(
                        {
                            "line": block.start_line,
                            "text": _normalize_space(sentence),
                        }
                    )
            elif _REQUIREMENT_CUE_RE.search(sentence) and (
                _ARCH_RE.search(sentence) or _EXECUTION_STATE_RE.search(sentence)
            ):
                raw.append(
                    {"line": block.start_line, "text": _normalize_space(sentence)}
                )

    for attribute in attributes:
        if attribute.startswith("__arm_streaming_compatible"):
            nodes.append({"op": "calling_context", "values": ["streaming_compatible"]})
        elif attribute.startswith("__arm_streaming"):
            nodes.append({"op": "calling_context", "values": ["streaming"]})
        elif attribute.startswith("__arm_locally_streaming"):
            nodes.append({"op": "calling_context", "values": ["locally_streaming"]})

    if not headers:
        defaults = {
            "general": "arm_acle.h",
            "sve": "arm_sve.h",
            "sve2": "arm_sve.h",
            "sve2.1": "arm_sve.h",
            "sve2.2": "arm_sve.h",
            "sve2.3": "arm_sve.h",
            "sme": "arm_sme.h",
            "sme2": "arm_sme.h",
            "sme2.1": "arm_sme.h",
            "sme2.2": "arm_sme.h",
            "sme2.3": "arm_sme.h",
        }
        for family in families:
            if family in defaults:
                _append_unique(headers, {"name": defaults[family], "status": "derived"})

    expression = _combine_requirements(nodes)
    return {
        "headers": headers,
        "expression": expression,
        "by_mode": {
            mode: _combine_requirements(mode_nodes)
            for mode, mode_nodes in sorted(by_mode.items())
        },
        "minimum_architecture": architecture,
        "execution_states": execution_states,
        "features": features,
        "extensions": extensions,
        "raw": raw,
        "diagnostics": diagnostics,
    }


def _sentences(text: str) -> Iterator[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = _normalize_space(paragraph)
        if not paragraph:
            continue
        yield from (item for item in re.split(r"(?<=[.!?])\s+", paragraph) if item)


def _macro_expression(
    sentence: str,
    macros: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
    """Parse one source condition without flattening mixed boolean operators."""

    macro_matches = list(_MACRO_RE.finditer(sentence))
    if not macro_matches:
        diagnostic = "The requirement sentence contains no parseable feature macro."
        return {"op": "raw", "text": _normalize_space(sentence)}, diagnostic

    code_candidate = _code_like_macro_span(sentence, macro_matches)
    if code_candidate:
        expression, diagnostic = parse_availability_guard(code_candidate)
        if diagnostic is None:
            return _availability_payload(expression), None

    atom_texts: list[str] = []
    connectors: list[tuple[str, bool]] = []
    for index, match in enumerate(macro_matches):
        if index:
            between = sentence[macro_matches[index - 1].end() : match.start()]
            connector, clause_boundary, connector_diagnostic = _prose_boolean_connector(
                between
            )
            if connector_diagnostic:
                return (
                    {"op": "raw", "text": _normalize_space(sentence)},
                    connector_diagnostic,
                )
            connectors.append((connector, clause_boundary))
        following_end = (
            macro_matches[index + 1].start()
            if index + 1 < len(macro_matches)
            else len(sentence)
        )
        following = sentence[match.end() : following_end]
        macro = match.group(0)
        if re.search(r"^\s*!=\s*0|\bnon-zero\b", following, re.IGNORECASE):
            atom_texts.append(f"{macro} != 0")
        elif re.search(r"^\s*==\s*1|\bdefined to\s+1\b", following, re.IGNORECASE):
            atom_texts.append(f"{macro} == 1")
        else:
            atom_texts.append(macro)

    expression, diagnostic = _parse_prose_boolean_expression(atom_texts, connectors)
    if diagnostic is not None:
        return (
            {"op": "raw", "text": _normalize_space(sentence)},
            f"Could not preserve availability expression: {diagnostic}",
        )
    return _availability_payload(expression), None


def _code_like_macro_span(
    sentence: str,
    macro_matches: Sequence[re.Match[str]],
) -> str | None:
    unquoted = sentence.replace("`", "")
    first_macro = macro_matches[0].group(0)
    last_macro = macro_matches[-1].group(0)
    start = unquoted.find(first_macro)
    end = unquoted.rfind(last_macro) + len(last_macro)
    if start < 0 or end <= start:
        return None
    while start > 0 and unquoted[start - 1] in " (!":
        start -= 1
    comparison = re.match(r"\s*(?:==|!=)\s*\d+", unquoted[end:])
    if comparison:
        end += comparison.end()
    while end < len(unquoted) and unquoted[end] in " )":
        end += 1
    candidate = unquoted[start:end].strip()
    return candidate if re.search(r"&&|\|\||[|!,()]", candidate) else None


def _prose_boolean_connector(value: str) -> tuple[str, bool, str | None]:
    if re.search(r"\band\s*/\s*or\b", value, re.IGNORECASE):
        return (
            "",
            False,
            (
                "The source uses the ambiguous prose connector 'and/or': "
                f"{_normalize_space(value)!r}."
            ),
        )
    has_or = bool(re.search(r"\|\||(?<!\|)\|(?!\|)|\bor\b", value, re.IGNORECASE))
    has_and = bool(re.search(r"&&|\band\b", value, re.IGNORECASE))
    if has_or and has_and:
        return (
            "",
            False,
            (
                "A macro connector contains both AND and OR without source-backed "
                f"grouping: {_normalize_space(value)!r}."
            ),
        )
    clause_boundary = "," in value
    if has_or:
        return "||", clause_boundary, None
    if has_and or clause_boundary:
        return "&&", clause_boundary, None
    return (
        "",
        False,
        (
            "No source-backed boolean connector was found between adjacent macros: "
            f"{_normalize_space(value)!r}."
        ),
    )


def _parse_prose_boolean_expression(
    atom_texts: Sequence[str],
    connectors: Sequence[tuple[str, bool]],
) -> tuple[AvailabilityExpr, str | None]:
    atoms: list[AvailabilityExpr] = []
    for atom_text in atom_texts:
        atom, diagnostic = parse_availability_guard(atom_text)
        if diagnostic is not None:
            return AvailabilityExpr.raw(" ".join(atom_texts)), diagnostic
        atoms.append(atom)

    clauses: list[list[AvailabilityExpr]] = [[atoms[0]]]
    clause_connectors: list[list[str]] = [[]]
    outer_connectors: list[str] = []
    for connector_index, (operator, boundary) in enumerate(connectors):
        if boundary:
            outer_connectors.append(operator)
            clauses.append([atoms[connector_index + 1]])
            clause_connectors.append([])
        else:
            clause_connectors[-1].append(operator)
            clauses[-1].append(atoms[connector_index + 1])

    clause_values: list[AvailabilityExpr] = []
    for clause_atoms, operators in zip(clauses, clause_connectors, strict=True):
        clause, diagnostic = _combine_prose_operands(clause_atoms, operators)
        if diagnostic is not None:
            return AvailabilityExpr.raw(" ".join(atom_texts)), diagnostic
        clause_values.append(clause)
    return _combine_prose_operands(clause_values, outer_connectors)


def _combine_prose_operands(
    operands: Sequence[AvailabilityExpr],
    operators: Sequence[str],
) -> tuple[AvailabilityExpr, str | None]:
    if len(operands) == 1:
        return operands[0], None
    unique_operators = set(operators)
    if len(unique_operators) != 1:
        return AvailabilityExpr.raw("mixed prose boolean expression"), (
            "Mixed prose boolean operators lack a source-backed clause boundary."
        )
    operator = next(iter(unique_operators))
    if operator == "&&":
        return AvailabilityExpr.all(*operands), None
    if operator == "||":
        return AvailabilityExpr.any(*operands), None
    return AvailabilityExpr.raw("unknown prose boolean operator"), (
        f"Unknown prose boolean operator: {operator!r}."
    )


def _availability_payload(expression: AvailabilityExpr) -> dict[str, Any]:
    if expression.op in {AvailabilityOp.ALL, AvailabilityOp.ANY, AvailabilityOp.NOT}:
        return {
            "op": expression.op.value,
            "args": [_availability_payload(item) for item in expression.arguments],
        }
    if expression.op is AvailabilityOp.DEFINED:
        return {"op": "defined", "macro": expression.key}
    if expression.op is AvailabilityOp.COMPARE:
        return {
            "op": "compare",
            "macro": expression.key,
            "comparator": cast(ComparisonOperator, expression.comparator).value,
            "value": expression.value,
        }
    if expression.op is AvailabilityOp.RAW:
        return {"op": "raw", "text": expression.text}
    return {"op": expression.op.value}


def _calling_mode(sentence: str) -> str | None:
    lowered = sentence.lower()
    if "streaming-compatible" in lowered:
        return "streaming_compatible"
    if "non-streaming" in lowered:
        return "non_streaming"
    if "streaming code" in lowered or "streaming mode" in lowered:
        return "streaming"
    return None


def _combine_requirements(nodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        key = json.dumps(node, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(node)
    if not unique:
        return {"op": "always"}
    if len(unique) == 1:
        return unique[0]
    return {"op": "all", "args": unique}


def _variant_hints(blocks: Sequence[_TextBlock]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for block in blocks:
        for offset, line in enumerate(block.lines):
            normalized = _normalize_space(line.lstrip("/ "))
            if normalized and _VARIANT_PROSE_RE.search(normalized):
                hints.append({"line": block.start_line + offset, "raw": normalized})
    return hints


def _state_access(attributes: Sequence[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for attribute in attributes:
        match = _STATE_ATTRIBUTE_RE.search(attribute)
        if match:
            states = re.findall(r'"([^"]+)"', match.group("states"))
            for state in states:
                _append_unique(result, {"state": state, "mode": match.group("mode")})
        elif attribute.startswith("__arm_agnostic"):
            _append_unique(result, {"state": "sme_za_state", "mode": "agnostic"})
    return result


def _instruction_relations(
    section: _Section,
    context_blocks: Sequence[_TextBlock],
) -> list[dict[str, Any]]:
    title = section.title
    result: list[dict[str, Any]] = []
    if not re.search(
        r"\b(?:intrinsics?|functions?|prototypes?|semantics?)\b", title, re.IGNORECASE
    ):
        heading_prefix = re.split(r"\s*\(", title, maxsplit=1)[0]
        tokens = [token.strip() for token in heading_prefix.split(",")]
        if tokens and all(re.fullmatch(r"[A-Z][A-Z0-9.]*", token) for token in tokens):
            result.append(
                {
                    "relation": "group",
                    "mnemonics": tokens,
                    "heading": title,
                    "guaranteed_emission": False,
                }
            )

    text = _blocks_text(context_blocks)
    generate_match = re.search(
        r"\bGenerates? (?:an? )?(?P<mnemonic>[A-Z][A-Z0-9]+)\b(?P<tail>[^.]*\.)",
        text,
    )
    if generate_match:
        tail = generate_match.group("tail")
        relation = (
            "semantic_equivalent"
            if re.search(r"\bequivalent\b|\bor nothing\b", tail)
            else "direct_access"
        )
        item = {
            "relation": relation,
            "mnemonics": [generate_match.group("mnemonic")],
            "raw": _normalize_space(generate_match.group(0)),
            "guaranteed_emission": False,
        }
        if item not in result:
            result.append(item)
    return result


def _callable_kind(section: _Section, declaration: dict[str, Any]) -> str:
    comments = " ".join(text for _, text in declaration["leading_comments"])
    path = " / ".join(item.title for item in section.ancestors())
    if re.search(r"external linkage", comments, re.IGNORECASE):
        return "support_function"
    if "Streaming-compatible versions of standard routines" in path:
        return "support_function"
    if declaration["name_pattern"].startswith("__arm_") and re.search(
        r"\bPSTATE functions\b", path, re.IGNORECASE
    ):
        return "support_function"
    return "intrinsic"


def _semantic_context(blocks: Sequence[_TextBlock]) -> str | None:
    text = _blocks_text(blocks)
    return text or None


def _collect_named_semantics(root: _Section) -> dict[str, str]:
    result: dict[str, str] = {}
    marker = re.compile(r"\*\*`(?P<name>[A-Za-z_]\w*)\(\)`\*\*")
    for section in _walk_sections(root):
        if section.title.lower() != "semantics":
            continue
        text = _blocks_text(
            block for block in section.blocks if isinstance(block, _TextBlock)
        )
        matches = list(marker.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            semantics = text[match.end() : end].strip()
            if semantics:
                result[match.group("name")] = semantics
    return result


def _parse_instruction_mapping_enrichments(
    root: _Section,
    *,
    default_maturity: tuple[str, int | None],
    source_commit: str,
    source_path: str,
) -> list[dict[str, Any]]:
    """Parse signature-less SVE instruction-to-family mapping table rows."""

    enrichments: list[dict[str, Any]] = []
    link_name = re.compile(r"\[`(?P<name>sv[A-Za-z0-9_]+)`\]\(")
    for section in _walk_sections(root):
        path = [item.title for item in section.ancestors() if item.title]
        if "Mapping of SVE instructions to intrinsics" not in path:
            continue
        for block in section.blocks:
            if not isinstance(block, _TextBlock):
                continue
            for offset, line in enumerate(block.lines):
                if not line.lstrip().startswith("|"):
                    continue
                columns = [item.strip() for item in line.strip().strip("|").split("|")]
                if len(columns) != 2 or columns[0].startswith(("**", "--")):
                    continue
                names = list(dict.fromkeys(link_name.findall(columns[1])))
                if not names:
                    continue
                instruction_text = re.sub(r"[`*_]", "", columns[0]).strip()
                mnemonic_match = re.match(
                    r"(?P<mnemonic>[A-Z][A-Z0-9.]*)", instruction_text
                )
                if mnemonic_match is None:
                    continue
                mnemonic = mnemonic_match.group("mnemonic")
                relation = (
                    "optimizer_candidate"
                    if "optimization" in columns[1].lower()
                    else "group"
                )
                source_line = block.start_line + offset
                for name in names:
                    enrichments.append(
                        {
                            "match": {"names": [], "base_names": [name]},
                            # The mapping table has no ISA column. The pinned
                            # LLVM TableGen guard adapter supplies the family.
                            "family": ["sve"],
                            "header": [{"name": "arm_sve.h", "status": "derived"}],
                            "availability": {
                                "expression": {"op": "always"},
                                "by_mode": {},
                                "minimum_architecture": [],
                                "execution_states": ["AArch64"],
                                "features": [],
                                "extensions": [],
                                "raw": [],
                            },
                            "maturity": _resolved_maturity(section, default_maturity),
                            "semantics": None,
                            "instructions": [
                                {
                                    "relation": relation,
                                    "mnemonics": [mnemonic],
                                    "form": instruction_text,
                                    "guaranteed_emission": False,
                                }
                            ],
                            "state": [],
                            "taxonomy_path": path,
                            "source_signature": None,
                            "provenance": {
                                "source": {
                                    "repository": ACLE_REPOSITORY,
                                    "commit": source_commit,
                                    "path": source_path,
                                    "start_line": source_line,
                                    "end_line": source_line,
                                    "license": ACLE_MARKDOWN_LICENSE,
                                },
                                "fields": {
                                    "signature": "unresolved",
                                    "instructions": "explicit",
                                    "maturity": _resolved_maturity(
                                        section, default_maturity
                                    )["status"],
                                },
                            },
                            "diagnostics": [
                                {
                                    "code": "signature_missing_use_declaration_inventory",
                                    "line": source_line,
                                    "message": (
                                        "The ACLE mapping table names an intrinsic family "
                                        "but does not provide a callable signature."
                                    ),
                                }
                            ],
                        }
                    )
    return enrichments


def _payload_to_concrete_callable(payload: dict[str, Any], *, family: str) -> Any:
    """Build one canonical model callable from an explicit source declaration."""

    from ..model import (
        Alias,
        AvailabilityExpr,
        AvailabilityOp,
        CallableKind,
        ComparisonOperator,
        CompilationRequirements,
        ConcreteCallable,
        Constraint,
        ConstraintKind,
        Diagnostic,
        FieldProvenance,
        InstructionMapping,
        InstructionRelationKind,
        Maturity,
        ModeAvailability,
        NameRole,
        Parameter,
        Provenance,
        ProvenanceKind,
        Semantics,
        Signature,
        SourceRef,
        StateAccess,
        StateAccessMode,
    )

    del AvailabilityExpr, AvailabilityOp, ComparisonOperator  # Used by helper imports.
    source_data = payload["provenance"]["source"]
    source = SourceRef(
        id=(
            f"acle:{source_data['path']}:"
            f"{source_data['start_line']}-{source_data['end_line']}"
        ),
        repository=source_data["repository"],
        commit=source_data["commit"],
        path=source_data["path"],
        start_line=source_data["start_line"],
        end_line=source_data["end_line"],
        license_id=source_data["license"],
        url=(
            f"https://github.com/{source_data['repository']}/blob/"
            f"{source_data['commit']}/{source_data['path']}"
        ),
    )
    explicit = Provenance(ProvenanceKind.EXPLICIT, (source,))
    expanded = Provenance(
        ProvenanceKind.EXPANDED,
        (source,),
        rule="ACLE lockstep bracket expansion",
    )
    availability = _availability_to_model(payload["availability"]["expression"])
    parameters = []
    for parameter_payload in payload["signature"]["parameters"]:
        constraints = tuple(
            Constraint(
                kind=ConstraintKind(item["kind"]),
                text=item["raw"],
                parameter=parameter_payload["name"],
                provenance=explicit,
            )
            for item in parameter_payload["constraints"]
        )
        parameters.append(
            Parameter(
                name=parameter_payload["name"],
                type_name=parameter_payload["type"],
                constraints=constraints,
            )
        )
    signature = Signature(
        return_type=payload["signature"]["return_type"],
        parameters=tuple(parameters),
        attributes=tuple(payload["signature"]["attributes"]),
        raw=payload["signature"]["raw"],
    )
    aliases = tuple(
        Alias(
            name=name,
            role=NameRole.OVERLOADED,
            availability=availability,
            provenance=expanded,
        )
        for name in payload["names"]["overloaded"]
    )
    instructions = tuple(
        InstructionMapping(
            relation=InstructionRelationKind(item["relation"]),
            mnemonic=mnemonic,
            form=item.get("form") or item.get("heading") or item.get("raw"),
            guaranteed_emission=item.get("guaranteed_emission", False),
            provenance=explicit,
        )
        for item in payload["instructions"]
        for mnemonic in item.get("mnemonics", [None])
    )
    states = tuple(
        StateAccess(
            state=item["state"],
            mode=StateAccessMode(item["mode"]),
            provenance=explicit,
        )
        for item in payload["state"]
    )
    headers = tuple(item["name"] for item in payload["header"])
    feature_macros = tuple(
        sorted(_requirement_macros_from_payload(payload["availability"]["expression"]))
    )
    mode_availability = tuple(
        ModeAvailability(
            mode=mode,
            availability=_availability_to_model(expression),
            provenance=explicit,
        )
        for mode, expression in sorted(payload["availability"]["by_mode"].items())
    )
    architecture_min, profiles = _compilation_architecture(payload["availability"])
    compilation = CompilationRequirements(
        architecture_min=architecture_min,
        profiles=profiles,
        extensions=tuple(payload["availability"]["extensions"]),
        feature_macros=feature_macros,
        headers=headers,
        execution_states=tuple(payload["availability"]["execution_states"]),
        availability=availability,
        availability_by_mode=mode_availability,
        provenance=explicit,
    )
    maturity = Maturity(payload["maturity"]["support_level"])
    semantics_text = payload["semantics"]
    semantics = Semantics(
        description=semantics_text,
        provenance=(
            explicit
            if semantics_text
            else Provenance.unresolved("No prose was associated with this declaration.")
        ),
    )
    diagnostics = tuple(
        Diagnostic(
            code=item["code"],
            message=item["message"],
            sources=(source,),
        )
        for item in payload["diagnostics"]
    )
    field_provenance = (
        FieldProvenance("signature", explicit),
        FieldProvenance("names", expanded if aliases else explicit),
        FieldProvenance(
            "maturity",
            Provenance(
                ProvenanceKind(payload["maturity"]["status"]),
                (source,),
            ),
        ),
        FieldProvenance("availability", explicit),
        FieldProvenance(
            "semantics",
            explicit
            if semantics_text
            else Provenance.unresolved("No associated prose."),
        ),
    )
    return ConcreteCallable(
        family=family,
        name=payload["names"]["explicit"],
        signature=signature,
        kind=CallableKind(payload["kind"]),
        name_role=NameRole.TYPED,
        aliases=aliases,
        availability=availability,
        maturity=maturity,
        semantics=semantics,
        instructions=instructions,
        state_access=states,
        compilation=compilation,
        headers=headers,
        taxonomy=(tuple(payload["taxonomy_path"]),),
        sources=(source,),
        field_provenance=field_provenance,
        diagnostics=diagnostics,
    )


def _availability_to_model(node: dict[str, Any]) -> Any:
    from ..model import AvailabilityExpr, AvailabilityOp, ComparisonOperator

    op = AvailabilityOp(node["op"])
    if op in {AvailabilityOp.ALL, AvailabilityOp.ANY, AvailabilityOp.NOT}:
        return AvailabilityExpr(
            op,
            arguments=tuple(
                _availability_to_model(child) for child in node.get("args", [])
            ),
        )
    if op is AvailabilityOp.DEFINED:
        return AvailabilityExpr(op, key=node["macro"])
    if op is AvailabilityOp.COMPARE:
        return AvailabilityExpr(
            op,
            key=node["macro"],
            comparator=ComparisonOperator(node["comparator"]),
            value=node["value"],
        )
    if op in {
        AvailabilityOp.CALLING_CONTEXT,
        AvailabilityOp.EXECUTION_STATE,
        AvailabilityOp.PROFILE,
        AvailabilityOp.ARCHITECTURE_MIN,
    }:
        return AvailabilityExpr(
            op,
            key=node.get("key"),
            value=tuple(node.get("values", [])) or node.get("value"),
        )
    if op is AvailabilityOp.RAW:
        return AvailabilityExpr.raw(node["text"])
    return AvailabilityExpr(op)


def _requirement_macros_from_payload(node: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    macro = node.get("macro")
    if macro:
        result.add(macro)
    for child in node.get("args", []):
        result.update(_requirement_macros_from_payload(child))
    return result


def _compilation_architecture(
    availability: dict[str, Any],
) -> tuple[str | None, tuple[str, ...]]:
    candidates = availability["minimum_architecture"]
    if not candidates:
        return None, ()

    def version_key(item: dict[str, Any]) -> tuple[int, ...]:
        return tuple(int(part) for part in item["version"].split("."))

    selected = max(candidates, key=version_key)
    profile = selected.get("profile")
    label = f"Armv{selected['version']}" + (f"-{profile}" if profile else "")
    profiles = tuple(
        dict.fromkeys(item["profile"] for item in candidates if item.get("profile"))
    )
    return label, profiles


def _blocks_text(blocks: Iterable[_TextBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def _deduplicate_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            ",".join(record["family"]),
            record["names"]["explicit"],
            record["signature"]["raw"],
        )
        if key in seen:
            diagnostics.append(
                {
                    "code": "duplicate_markdown_declaration",
                    "message": record["names"]["explicit"],
                    "first_line": seen[key]["provenance"]["source"]["start_line"],
                    "duplicate_line": record["provenance"]["source"]["start_line"],
                }
            )
            continue
        seen[key] = record
        result.append(record)
    return result, diagnostics
