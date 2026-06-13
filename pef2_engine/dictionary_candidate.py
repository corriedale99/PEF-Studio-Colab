from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pef2_engine.gemini_dictionary_review import MAX_REVIEW_TERMS


ASCII_PATTERN = re.compile(r"[A-Za-z]")
ASCII_ONLY_PATTERN = re.compile(r"^[A-Za-z]+$")
KANJI_PATTERN = re.compile(r"[一-龯々〆ヶ]")
HIRAGANA_ONLY_PATTERN = re.compile(r"^[ぁ-ん]+$")
KATAKANA_ONLY_PATTERN = re.compile(r"^[ァ-ヶー・]+$")
DIGIT_ONLY_PATTERN = re.compile(r"^\d+$")
NUMBER_SYMBOL_ONLY_PATTERN = re.compile(r"^[\d\s\W_]+$")
SENTENCE_PATTERN = re.compile(r"[^。！？!?]+[。！？!?]+[」』]*|[^。！？!?]+[」』]+|[^。！？!?]+")
ENGLISH_PHRASE_PATTERN = re.compile(r"\b[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})+\b")
ENGLISH_WORD_PATTERN = re.compile(r"\b[A-Za-z]{2,}\b")
ENGLISH_CANDIDATE_SURFACE_PATTERN = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)*$")
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
MERGE_STOP_POS = {"助詞", "助動詞", "補助記号"}
ENGLISH_STOPWORDS = {"the", "and", "of", "in", "to", "for", "with", "is", "are"}


@dataclass
class RawSentence:
    text: str
    paragraph_index: int
    line_index: int
    sentence_index: int


@dataclass
class CandidateRecord:
    index: int
    paragraph_index: int
    line_index: int
    sentence_index: int
    tokens: dict = field(default_factory=dict)
    dictionary_matches: list[dict] = field(default_factory=list)
    suspicious_terms: list[dict] = field(default_factory=list)


def split_into_sentences(text: str) -> list[RawSentence]:
    sentences: list[RawSentence] = []
    paragraph_index = 0
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        paragraph for paragraph in re.split(r"\n\s*\n", normalized) if paragraph.strip()
    ]
    for paragraph in paragraphs:
        line_index = 0
        for line in paragraph.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue
            sentence_index = 0
            for sentence_text in _split_line(stripped_line):
                sentences.append(
                    RawSentence(
                        text=sentence_text,
                        paragraph_index=paragraph_index,
                        line_index=line_index,
                        sentence_index=sentence_index,
                    )
                )
                sentence_index += 1
            line_index += 1
        paragraph_index += 1
    return sentences


def tokenize_for_dictionary_candidates(text: str) -> tuple[list[dict], list[str]]:
    try:
        return _tokenize_sudachi(text), []
    except ImportError:
        return _tokenize_fallback(text), ["sudachipy_unavailable_fallback_tokenizer_used"]


def extract_suspicious_terms(
    sudachi_tokens: list[dict],
    dictionary_matches: list[dict],
) -> list[dict]:
    known_words = {match.get("word", "") for match in dictionary_matches}
    suspicious_terms: list[dict] = []
    seen_surfaces: set[str] = set()

    for token in sudachi_tokens:
        surface = str(token.get("surface", "")).strip()
        if not surface or surface in seen_surfaces or surface in known_words:
            continue

        pos = token.get("pos", [])
        if _is_punctuation(pos) or _is_noise_surface(surface):
            continue

        reading = str(token.get("reading", ""))
        reasons = _collect_reasons(surface, reading, pos)
        if not reasons:
            continue

        suspicious_terms.append(
            {
                "index": len(suspicious_terms),
                "surface": surface,
                "reading": reading,
                "pos": pos,
                "reasons": reasons,
                "priority": _get_priority(surface, reasons),
                "source": token.get("source", "sudachi"),
            }
        )
        seen_surfaces.add(surface)
    return suspicious_terms


def extract_english_terms(
    text: str,
    dictionary_matches: list[dict],
) -> list[dict]:
    known_words = {match.get("word", "") for match in dictionary_matches}
    suspicious_terms: list[dict] = []
    seen_surfaces: set[str] = set()
    phrase_ranges: list[tuple[int, int]] = []

    for match in ENGLISH_PHRASE_PATTERN.finditer(text):
        words = [word_match.group(0) for word_match in ENGLISH_WORD_PATTERN.finditer(match.group(0))]
        words = _trim_english_stopwords(words)
        if len(words) < 2:
            continue
        surface = " ".join(words)
        if surface in known_words or surface in seen_surfaces:
            continue
        suspicious_terms.append(
            _english_term(
                surface=surface,
                reasons=["english_phrase"],
                count_index=len(suspicious_terms),
            )
        )
        seen_surfaces.add(surface)
        phrase_ranges.append(match.span())

    for match in ENGLISH_WORD_PATTERN.finditer(text):
        surface = match.group(0)
        if (
            _is_english_stopword(surface)
            or surface in known_words
            or surface in seen_surfaces
            or _overlaps_range(match.start(), match.end(), phrase_ranges)
        ):
            continue
        suspicious_terms.append(
            _english_term(
                surface=surface,
                reasons=_english_word_reasons(surface),
                count_index=len(suspicious_terms),
            )
        )
        seen_surfaces.add(surface)

    return suspicious_terms


def aggregate_suspicious_terms(records: list[CandidateRecord]) -> list[dict]:
    aggregated: dict[str, dict] = {}

    for record in records:
        seen_in_record: set[str] = set()
        for term in record.suspicious_terms:
            surface = term.get("surface", "")
            if not surface or surface in seen_in_record:
                continue
            seen_in_record.add(surface)

            if surface not in aggregated:
                aggregated[surface] = {
                    "surface": surface,
                    "reading": term.get("reading", ""),
                    "pos": term.get("pos", []),
                    "reasons": [],
                    "priority": term.get("priority", "low"),
                    "count": 0,
                    "occurrences": [],
                    "source": term.get("source", "sudachi"),
                }

            item = aggregated[surface]
            item["count"] += 1
            item["priority"] = _higher_priority(
                item["priority"],
                term.get("priority", "low"),
            )
            item["reasons"] = _merge_reasons(item["reasons"], term.get("reasons", []))
            item["occurrences"].append(
                {
                    "record_index": record.index,
                    "paragraph_index": record.paragraph_index,
                    "line_index": record.line_index,
                    "sentence_index": record.sentence_index,
                }
            )

    results = sorted(
        aggregated.values(),
        key=lambda item: (
            PRIORITY_RANK.get(item["priority"], PRIORITY_RANK["low"]),
            -item["count"],
            item["surface"],
        ),
    )
    for index, item in enumerate(results):
        item["index"] = index
    return results


def build_ai_review_terms(
    aggregated_terms: list[dict],
    records: list[CandidateRecord] | None = None,
    max_terms: int = MAX_REVIEW_TERMS,
) -> dict:
    source_records = records or []
    merged_terms = _build_merged_review_terms(source_records)
    misread_person_surfaces = _misread_person_name_surfaces(source_records)
    excluded_surfaces = {
        surface
        for term in merged_terms
        for surface in term.get("source_terms", [])
    } | misread_person_surfaces
    review_terms = [
        term for term in aggregated_terms if term.get("surface") not in excluded_surfaces
    ] + merged_terms
    candidates = [term for term in review_terms if _is_ai_review_candidate(term)]
    candidates.sort(
        key=lambda term: (
            -term.get("count", 0),
            "proper_noun" not in term.get("reasons", []),
            term.get("surface", ""),
        )
    )
    selected = candidates[:max_terms]

    return {
        "schema_version": "step5b-1",
        "purpose": "gemini_reading_review_candidates",
        "dictionary_scope": "work",
        "gemini_api_called": False,
        "selection_policy": {
            "max_terms": max_terms,
            "rules": [
                "priority high or medium",
                "surface contains kanji",
                "single kanji surface excluded",
                "proper_noun or suspicious_reading or short_kanji_word or kanji_noun",
                "english_phrase or english_acronym or english_term",
                "single english stopwords excluded",
            ],
        },
        "total_candidates": len(review_terms),
        "selected_count": len(selected),
        "items": [_to_ai_review_item(index, term) for index, term in enumerate(selected)],
    }


def build_gemini_prompt_preview(ai_review_terms: dict) -> dict:
    candidates = [_to_prompt_candidate(item) for item in ai_review_terms.get("items", [])]
    return {
        "schema_version": "step5b-3",
        "purpose": "gemini_prompt_preview",
        "gemini_api_called": False,
        "dictionary_scope": "work",
        "target_dictionary": "work",
        "promote_to_user_dictionary": False,
        "candidate_count": len(candidates),
        "model_policy": {
            "api_call": "disabled",
            "future_use": "reading_suggestion_for_work_dictionary_draft",
        },
        "system_instruction": _build_gemini_system_instruction(),
        "user_prompt": _build_gemini_user_prompt(candidates),
        "candidates": candidates,
        "expected_response_schema": [
            {
                "単語原文": "犍陀多",
                "読み": "カンダタ",
                "意味": "作品固有語・人名などの簡潔な説明",
                "難易度": 5,
                "confidence": "high",
                "target_dictionary": "work",
            }
        ],
    }


def _tokenize_sudachi(text: str) -> list[dict]:
    from sudachipy import dictionary
    from sudachipy import tokenizer

    sudachi_tokenizer = dictionary.Dictionary().create()
    tokens = sudachi_tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.C)
    return [
        {
            "index": index,
            "surface": token.surface(),
            "reading": token.reading_form(),
            "pos": list(token.part_of_speech()),
            "source": "sudachi",
        }
        for index, token in enumerate(tokens)
    ]


def _tokenize_fallback(text: str) -> list[dict]:
    tokens: list[dict] = []
    for match in re.finditer(r"[一-龯々〆ヶ]{1,5}|[A-Za-z][A-Za-z0-9_-]*", text):
        surface = match.group(0)
        tokens.append(
            {
                "index": len(tokens),
                "surface": surface,
                "reading": "",
                "pos": ["名詞", "固有名詞" if len(surface) >= 2 else "普通名詞"],
                "source": "fallback",
            }
        )
    return tokens


def _split_line(line: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in SENTENCE_PATTERN.finditer(line)
        if match.group(0).strip()
    ]


def _collect_reasons(surface: str, reading: str, pos: list) -> list[str]:
    reasons: list[str] = []
    has_kanji = bool(KANJI_PATTERN.search(surface))

    if ASCII_PATTERN.search(surface):
        reasons.append("ascii_word")
    if "固有名詞" in pos:
        reasons.append("proper_noun")
    if not reading or (reading == surface and not _is_katakana_only(surface)):
        reasons.append("suspicious_reading")
    if has_kanji and pos and pos[0] == "名詞":
        reasons.append("kanji_noun")
    if has_kanji and 1 <= len(surface) <= 2:
        reasons.append("short_kanji_word")
    return reasons


def _get_priority(surface: str, reasons: list[str]) -> str:
    if (
        "proper_noun" in reasons
        or "ascii_word" in reasons
        or ("suspicious_reading" in reasons and KANJI_PATTERN.search(surface))
    ):
        return "high"
    if "kanji_noun" in reasons or "short_kanji_word" in reasons:
        return "medium"
    return "low"


def _is_ai_review_candidate(term: dict) -> bool:
    surface = term.get("surface", "")
    reasons = term.get("reasons", [])
    if _is_english_review_candidate(surface, reasons):
        return True
    if not bool(KANJI_PATTERN.search(surface)) or _is_single_kanji_surface(surface):
        return False
    if term.get("priority") == "medium":
        return "kanji_noun" in reasons or "short_kanji_word" in reasons
    return (
        term.get("priority") == "high"
        and (
            "proper_noun" in reasons
            or "suspicious_reading" in reasons
            or "short_kanji_word" in reasons
            or "honorific_name_pattern" in reasons
        )
    )


def _is_single_kanji_surface(surface: str) -> bool:
    return len(surface) == 1 and bool(KANJI_PATTERN.fullmatch(surface))


def _is_english_review_candidate(surface: str, reasons: list[str]) -> bool:
    return (
        bool(ENGLISH_CANDIDATE_SURFACE_PATTERN.fullmatch(surface))
        and not _is_english_stopword(surface)
        and (
            "english_phrase" in reasons
            or "english_acronym" in reasons
            or "english_term" in reasons
        )
    )


def _to_ai_review_item(index: int, term: dict) -> dict:
    item = {
        "index": index,
        "surface": term.get("surface", ""),
        "current_reading": term.get("reading", ""),
        "suggested_reading": "",
        "target_dictionary": "work",
        "promote_to_user_dictionary": False,
        "priority": term.get("priority", "low"),
        "reasons": term.get("reasons", []),
        "count": term.get("count", 0),
        "occurrences": term.get("occurrences", []),
        "decision": "pending",
        "notes": "",
        "source": term.get("source", "sudachi"),
    }
    if "source_terms" in term:
        item["source_terms"] = term["source_terms"]
    return item


def _to_prompt_candidate(item: dict) -> dict:
    candidate = {
        "surface": item.get("surface", ""),
        "current_reading": item.get("current_reading", ""),
        "reasons": item.get("reasons", []),
        "count": item.get("count", 0),
    }
    if "source_terms" in item:
        candidate["source_terms"] = item["source_terms"]
    return candidate


def _build_gemini_system_instruction() -> str:
    return (
        "あなたは日本語TTS向けの読み辞書候補を作る辞書編纂者です。"
        "対象は作品辞書であり、全作品共通のユーザ辞書ではありません。"
        "辞書は短いほどよいです。100%読み間違えそうな語、初見で読みが止まりそうな語、"
        "作品固有語、人名、地名、当て字、難読語を優先してください。"
        "普通に読める一般語は返さないでください。"
        "候補語を勝手に分解しないでください。入力候補にない語を新規追加せず、"
        "入力候補と同じ表記の語だけを単語原文に入れてください。"
        "複合語や英語フレーズが候補に含まれる場合もむやみに分解しないでください。"
        "読みはカタカナを基本にし、スペースを入れないでください。"
        "JSON配列だけを返し、Markdownコードフェンスや説明文を返さないでください。"
        "返答は1個のJSON配列だけにしてください。空の場合は[]だけを1回だけ返してください。"
        "前置き、後書き、説明文、複数のJSON配列は禁止です。"
        "指定キー以外の余計なキーは返さないでください。"
        "target_dictionaryは返す場合でもworkにしてください。"
    )


def _build_gemini_user_prompt(candidates: list[dict]) -> str:
    return (
        "以下の候補語について、日本語TTS向けの読みを確認してください。"
        "作品辞書ドラフトに入れる前提で、必要な候補だけをJSON配列で返してください。\n"
        "100%読み間違えそうな語、初見で読みが止まりそうな語を優先してください。\n"
        "普通に読める一般語は返さないでください。辞書は短いほどよいです。\n"
        "候補語を分解しないでください。入力候補と同じ表記だけ返してください。\n"
        "読みにはスペースを入れないでください。JSON配列だけ返してください。\n"
        "返答は1個のJSON配列だけです。空の場合は[]だけを1回だけ返してください。\n"
        "前置き、後書き、説明文、複数のJSON配列は禁止です。\n"
        "期待する形式:\n"
        "[{\"単語原文\":\"犍陀多\",\"読み\":\"カンダタ\",\"意味\":\"作品固有の人名\","
        "\"難易度\":5,\"confidence\":\"high\",\"target_dictionary\":\"work\"}]\n"
        f"候補:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


def _build_merged_review_terms(records: list[CandidateRecord]) -> list[dict]:
    merged: dict[str, dict] = {}

    for record in records:
        known_words = {match.get("word", "") for match in record.dictionary_matches}
        terms_by_surface = {
            term.get("surface", ""): term for term in record.suspicious_terms
        }
        tokens = record.tokens.get("sudachi", [])

        for token_index, _token in enumerate(tokens):
            _collect_honorific_candidate(merged, record, tokens, token_index)
            _collect_kanji_sequence_candidate(
                merged,
                record,
                tokens,
                token_index,
                terms_by_surface,
                known_words,
            )

    results = sorted(merged.values(), key=lambda item: (-item["count"], item["surface"]))
    for index, item in enumerate(results):
        item["index"] = index
    return results


def _collect_kanji_sequence_candidate(
    merged: dict[str, dict],
    record: CandidateRecord,
    tokens: list[dict],
    token_index: int,
    terms_by_surface: dict[str, dict],
    known_words: set[str],
) -> None:
    token = tokens[token_index]
    surface = token.get("surface", "")
    seed_term = terms_by_surface.get(surface)
    if not seed_term or not _is_merge_seed(seed_term):
        return
    if _has_previous_kanji_sequence_token(tokens, token_index):
        return

    source_terms = [surface]
    readings = [token.get("reading", "")]
    merged_surface = surface

    for next_index, next_token in enumerate(tokens[token_index + 1:], start=token_index + 1):
        next_surface = next_token.get("surface", "")
        if _is_merge_stop_token(next_token):
            break
        if len(merged_surface + next_surface) > 5:
            break

        merged_surface += next_surface
        source_terms.append(next_surface)
        readings.append(next_token.get("reading", ""))

        if (
            len(merged_surface) >= 2
            and len(merged_surface) > len(surface)
            and KANJI_PATTERN.search(merged_surface)
            and merged_surface not in known_words
            and not _has_following_kanji_sequence_token(tokens, next_index)
        ):
            reasons = _merge_reasons(
                seed_term.get("reasons", []),
                ["merged_kanji_sequence"],
            )
            _upsert_merged_candidate(
                merged,
                record,
                merged_surface,
                "".join(readings),
                seed_term.get("pos", []),
                reasons,
                source_terms,
            )


def _collect_honorific_candidate(
    merged: dict[str, dict],
    record: CandidateRecord,
    tokens: list[dict],
    token_index: int,
) -> None:
    token = tokens[token_index]
    surface = token.get("surface", "")

    if surface == "御" and token_index + 2 < len(tokens):
        middle = tokens[token_index + 1]
        last = tokens[token_index + 2]
        if _is_kanji_noun(middle) and last.get("surface") == "様":
            _add_honorific_candidate(merged, record, [token, middle, last])

    if surface.startswith("御") and KANJI_PATTERN.search(surface) and token_index + 1 < len(tokens):
        last = tokens[token_index + 1]
        if last.get("surface") == "様":
            _add_honorific_candidate(merged, record, [token, last])


def _add_honorific_candidate(
    merged: dict[str, dict],
    record: CandidateRecord,
    source_tokens: list[dict],
) -> None:
    surface = "".join(token.get("surface", "") for token in source_tokens)
    if not (3 <= len(surface) <= 5) or not KANJI_PATTERN.search(surface):
        return

    _upsert_merged_candidate(
        merged,
        record,
        surface,
        "".join(token.get("reading", "") for token in source_tokens),
        source_tokens[0].get("pos", []),
        ["honorific_name_pattern", "merged_kanji_sequence"],
        [token.get("surface", "") for token in source_tokens],
    )


def _upsert_merged_candidate(
    merged: dict[str, dict],
    record: CandidateRecord,
    surface: str,
    reading: str,
    pos: list,
    reasons: list[str],
    source_terms: list[str],
) -> None:
    if surface not in merged:
        merged[surface] = {
            "surface": surface,
            "reading": reading,
            "pos": pos,
            "reasons": reasons,
            "priority": "high",
            "count": 0,
            "occurrences": [],
            "source": "sudachi",
            "source_terms": source_terms[:],
        }

    item = merged[surface]
    if not any(occurrence["record_index"] == record.index for occurrence in item["occurrences"]):
        item["count"] += 1
        item["occurrences"].append(
            {
                "record_index": record.index,
                "paragraph_index": record.paragraph_index,
                "line_index": record.line_index,
                "sentence_index": record.sentence_index,
            }
        )


def _is_merge_seed(term: dict) -> bool:
    surface = term.get("surface", "")
    reasons = term.get("reasons", [])
    return (
        bool(KANJI_PATTERN.search(surface))
        and not HIRAGANA_ONLY_PATTERN.fullmatch(surface)
        and not KATAKANA_ONLY_PATTERN.fullmatch(surface)
        and not ASCII_ONLY_PATTERN.fullmatch(surface)
        and not DIGIT_ONLY_PATTERN.fullmatch(surface)
        and ("proper_noun" in reasons or "suspicious_reading" in reasons)
    )


def _is_merge_stop_token(token: dict) -> bool:
    surface = token.get("surface", "")
    pos = token.get("pos", [])
    return (
        not surface
        or (pos and pos[0] in MERGE_STOP_POS)
        or (pos and pos[0] == "接尾辞")
        or not KANJI_PATTERN.search(surface)
        or bool(HIRAGANA_ONLY_PATTERN.fullmatch(surface))
        or _is_noise_surface(surface)
    )


def _has_previous_kanji_sequence_token(tokens: list[dict], token_index: int) -> bool:
    return token_index > 0 and not _is_merge_stop_token(tokens[token_index - 1])


def _has_following_kanji_sequence_token(tokens: list[dict], token_index: int) -> bool:
    next_index = token_index + 1
    return next_index < len(tokens) and not _is_merge_stop_token(tokens[next_index])


def _misread_person_name_surfaces(records: list[CandidateRecord]) -> set[str]:
    surfaces: set[str] = set()
    for record in records:
        run: list[dict] = []
        for token in record.tokens.get("sudachi", []):
            if _is_merge_stop_token(token):
                _collect_mixed_person_name_surfaces(run, surfaces)
                run = []
            else:
                run.append(token)
        _collect_mixed_person_name_surfaces(run, surfaces)
    return surfaces


def _collect_mixed_person_name_surfaces(run: list[dict], surfaces: set[str]) -> None:
    if len(run) < 2:
        return
    has_person = any(_is_person_name_token(token) for token in run)
    has_non_person = any(not _is_person_name_token(token) for token in run)
    if not (has_person and has_non_person):
        return
    for token in run:
        if _is_person_name_token(token):
            surfaces.add(str(token.get("surface") or ""))


def _is_person_name_token(token: dict) -> bool:
    pos = token.get("pos", [])
    return len(pos) >= 3 and pos[0] == "名詞" and pos[1] == "固有名詞" and pos[2] == "人名"


def _is_kanji_noun(token: dict) -> bool:
    surface = token.get("surface", "")
    pos = token.get("pos", [])
    return bool(KANJI_PATTERN.search(surface)) and bool(pos) and pos[0] == "名詞"


def _higher_priority(current: str, candidate: str) -> str:
    if PRIORITY_RANK.get(candidate, PRIORITY_RANK["low"]) < PRIORITY_RANK.get(
        current, PRIORITY_RANK["low"]
    ):
        return candidate
    return current


def _merge_reasons(current: list[str], new_reasons: list[str]) -> list[str]:
    merged = current[:]
    for reason in new_reasons:
        if reason not in merged:
            merged.append(reason)
    return merged


def _is_punctuation(pos: list) -> bool:
    return bool(pos) and pos[0] == "補助記号"


def _is_noise_surface(surface: str) -> bool:
    return (
        bool(HIRAGANA_ONLY_PATTERN.fullmatch(surface))
        or bool(NUMBER_SYMBOL_ONLY_PATTERN.fullmatch(surface))
    )


def _is_katakana_only(surface: str) -> bool:
    return bool(KATAKANA_ONLY_PATTERN.fullmatch(surface))


def _english_term(surface: str, reasons: list[str], count_index: int) -> dict:
    return {
        "index": count_index,
        "surface": surface,
        "reading": "",
        "pos": ["名詞", "固有名詞", "一般", "*", "*", "*"],
        "reasons": reasons,
        "priority": "high",
        "source": "english_regex",
    }


def _english_word_reasons(surface: str) -> list[str]:
    reasons = ["ascii_word", "english_term"]
    if _is_english_acronym(surface):
        reasons.append("english_acronym")
    if _is_english_term(surface):
        reasons.append("english_term")
    return reasons


def _is_english_acronym(surface: str) -> bool:
    return len(surface) >= 2 and surface.isupper()


def _is_english_term(surface: str) -> bool:
    return any(char.isupper() for char in surface[1:]) or _is_english_acronym(surface)


def _is_english_stopword(surface: str) -> bool:
    return surface.lower() in ENGLISH_STOPWORDS


def _trim_english_stopwords(words: list[str]) -> list[str]:
    start = 0
    end = len(words)
    while start < end and _is_english_stopword(words[start]):
        start += 1
    while end > start and _is_english_stopword(words[end - 1]):
        end -= 1
    return words[start:end]


def _overlaps_range(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in ranges)
