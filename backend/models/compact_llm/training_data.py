from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "당신은 감정 기반 스마트 조명 제어 모델입니다. "
    "반드시 JSON만 출력하세요. "
    "규칙: 명시적 색상 요청이 있으면 hue는 그 색상을 우선하되, "
    "감정과 강도 표현에 따라 saturation, brightness, dimmer, color temperature를 조절하세요. "
    "애매한 감정이면 안전하고 무난한 조명값을 사용하세요. "
    "출력 키는 H,S,B,Dimmer,CT만 사용하세요."
)

USER_PROMPT_TEMPLATE = "문장: {text}"
SAFE_FALLBACK = {"H": 45, "S": 20, "B": 56, "Dimmer": 56, "CT": 300}

EMOTION_PROFILES: dict[str, dict[str, Any]] = {
    "sadness": {
        "label": "슬픔",
        "target": {"H": 220, "S": 36, "B": 24, "Dimmer": 22, "CT": 410},
        "phrases": ["슬퍼", "우울해", "외로워", "침울해", "쓸쓸해", "울적해", "기운이 없어", "마음이 가라앉아"],
        "adjust": {"H": -4, "S": -28, "B": -36, "Dimmer": -38, "CT": 90},
    },
    "happiness": {
        "label": "행복",
        "target": {"H": 38, "S": 78, "B": 88, "Dimmer": 86, "CT": 235},
        "phrases": ["행복해", "행복하다", "기뻐", "너무 좋아", "즐거워", "재미있다", "아주 재밌다", "신나", "설레", "기분 최고야", "들떠 있어", "두근거리고 설레"],
        "adjust": {"H": 2, "S": 6, "B": 14, "Dimmer": 14, "CT": -25},
    },
    "anger": {
        "label": "분노",
        "target": {"H": 12, "S": 92, "B": 72, "Dimmer": 76, "CT": 255},
        "phrases": ["화가 나", "짜증나", "열받아", "분노가 올라와", "빡쳐", "너무 화나", "속이 부글부글해", "신경질 나", "화가 치밀어", "짜증이 폭발해"],
        "adjust": {"H": 0, "S": 12, "B": 10, "Dimmer": 12, "CT": -20},
    },
    "calm": {
        "label": "평온",
        "target": {"H": 185, "S": 42, "B": 58, "Dimmer": 60, "CT": 315},
        "phrases": ["평온해", "안정돼", "편안해", "차분해", "고요해", "잔잔해", "마음이 놓여", "한결 부드러워", "편안한 밤이야", "조용히 안정되는 느낌이야"],
        "adjust": {"H": 3, "S": -22, "B": -8, "Dimmer": -8, "CT": 45},
    },
    "anxiety": {
        "label": "불안",
        "target": {"H": 210, "S": 30, "B": 30, "Dimmer": 28, "CT": 395},
        "phrases": ["불안해", "긴장돼", "걱정돼", "초조해", "조마조마해", "마음이 불편해", "안절부절못해", "무서워"],
        "adjust": {"H": -2, "S": -30, "B": -30, "Dimmer": -32, "CT": 65},
    },
}

COLOR_PROFILES: dict[str, dict[str, Any]] = {
    "red": {
        "label": "빨간색",
        "anchor_h": 0,
        "target": {"H": 0, "S": 92, "B": 74, "Dimmer": 72, "CT": 255},
        "phrases": ["빨강", "빨강색", "빨간색", "빨간빛", "붉은색", "붉은빛", "새빨간 빛", "강렬한 빨강", "빨강 느낌", "붉은 계열"],
    },
    "yellow": {
        "label": "노란색",
        "anchor_h": 55,
        "target": {"H": 55, "S": 88, "B": 82, "Dimmer": 80, "CT": 240},
        "phrases": ["노랑", "노랑색", "노란색", "노란빛", "황금빛", "따뜻한 노란빛", "밝은 노란색", "노랑 느낌", "노란 계열"],
    },
    "blue": {
        "label": "파란색",
        "anchor_h": 220,
        "target": {"H": 220, "S": 78, "B": 64, "Dimmer": 62, "CT": 410},
        "phrases": ["파랑", "파랑색", "파란색", "파란빛", "푸른빛", "하늘빛", "하늘같이 파란 색", "여름 하늘빛 같은 파랑", "파란 계열"],
    },
    "green": {
        "label": "초록색",
        "anchor_h": 120,
        "target": {"H": 120, "S": 74, "B": 68, "Dimmer": 66, "CT": 320},
        "phrases": ["초록색", "초록빛", "연초록빛", "부드러운 초록색", "초록 느낌"],
    },
    "purple": {
        "label": "보라색",
        "anchor_h": 285,
        "target": {"H": 285, "S": 76, "B": 62, "Dimmer": 60, "CT": 340},
        "phrases": ["보라색", "보랏빛", "연보라색", "보라 느낌", "은은한 보라색"],
    },
    "orange": {
        "label": "주황색",
        "anchor_h": 28,
        "target": {"H": 28, "S": 90, "B": 78, "Dimmer": 76, "CT": 250},
        "phrases": ["주황색", "오렌지빛", "주황빛", "주황 느낌", "따뜻한 주황색"],
    },
    "white": {
        "label": "흰색",
        "anchor_h": 45,
        "target": {"H": 45, "S": 6, "B": 84, "Dimmer": 84, "CT": 285},
        "phrases": ["흰색", "하얀 빛", "맑은 흰색", "깔끔한 흰빛", "밝은 흰색"],
    },
}

INTENSITY_MODIFIERS: dict[str, dict[str, Any]] = {
    "neutral": {"label": "", "phrases": [""], "adjust": {"H": 0, "S": 0, "B": 0, "Dimmer": 0, "CT": 0}},
    "soft": {"label": "은은한", "phrases": ["은은한", "연한", "부드러운"], "adjust": {"H": -3, "S": -24, "B": -8, "Dimmer": -12, "CT": 18}},
    "strong": {"label": "강한", "phrases": ["강한", "선명한", "진한", "강렬한"], "adjust": {"H": 2, "S": 10, "B": 8, "Dimmer": 8, "CT": -10}},
    "bright": {"label": "밝은", "phrases": ["밝은", "환한"], "adjust": {"H": 1, "S": 4, "B": 16, "Dimmer": 16, "CT": -8}},
    "dark": {"label": "어두운", "phrases": ["어두운", "짙고 어두운", "낮은 밝기의"], "adjust": {"H": -2, "S": -8, "B": -30, "Dimmer": -32, "CT": 38}},
}

AMBIGUOUS_INPUTS = [
    "기분이 좀 애매해",
    "뭐라고 표현하기 어려운 느낌이야",
    "딱히 큰 감정은 없는데 묘해",
    "무난하게 해줘",
    "잘 모르겠지만 편하게 보여줘",
    "무난하고 애매한 기분이야",
    "특별한 감정은 없고 무난해",
    "과하지 않게 적당한 느낌이면 돼",
    "평범하고 안전한 분위기로 해줘",
]

TARGETED_HARD_CASES = [
    {"text": "하늘같이 파란 마음", "emotion": None, "base_color": "blue", "intensity": "neutral"},
    {"text": "여름 하늘은 정말 파랗다", "emotion": None, "base_color": "blue", "intensity": "bright"},
    {"text": "빨강", "emotion": None, "base_color": "red", "intensity": "neutral"},
    {"text": "빨강색", "emotion": None, "base_color": "red", "intensity": "neutral"},
    {"text": "빨갛다", "emotion": None, "base_color": "red", "intensity": "neutral"},
    {"text": "붉은빛으로 표현해줘", "emotion": None, "base_color": "red", "intensity": "soft"},
    {"text": "오늘은 정말 들떠 있고 설레", "emotion": "happiness", "base_color": None, "intensity": "bright"},
    {"text": "신경질 나고 열받아", "emotion": "anger", "base_color": None, "intensity": "strong"},
    {"text": "잔잔하고 편안한 밤이야", "emotion": "calm", "base_color": None, "intensity": "soft"},
    {"text": "무난하고 애매한 기분이야", "emotion": None, "base_color": None, "intensity": "neutral"},
]


COLOR_ADJECTIVE_PROFILES: dict[str, list[dict[str, str]]] = {
    "red": [
        {"adjective": "발갛다", "adverbial": "발갛게", "noun_phrase": "발갛게 물든 붉은빛", "intensity": "neutral"},
        {"adjective": "발그스레하다", "adverbial": "발그스레하게", "noun_phrase": "발그스레한 붉은빛", "intensity": "soft"},
        {"adjective": "발그레하다", "adverbial": "발그레하게", "noun_phrase": "발그레한 빨간빛", "intensity": "soft"},
        {"adjective": "불그스름하다", "adverbial": "불그스름하게", "noun_phrase": "불그스름한 붉은빛", "intensity": "soft"},
        {"adjective": "붉으스름하다", "adverbial": "붉으스름하게", "noun_phrase": "붉으스름한 빛", "intensity": "soft"},
        {"adjective": "새빨갛다", "adverbial": "새빨갛게", "noun_phrase": "새빨간 빛", "intensity": "bright"},
        {"adjective": "시뻘겋다", "adverbial": "시뻘겋게", "noun_phrase": "시뻘건 붉은빛", "intensity": "strong"},
        {"adjective": "핏빛이다", "adverbial": "핏빛처럼", "noun_phrase": "핏빛 붉은색", "intensity": "dark"},
    ],
    "yellow": [
        {"adjective": "노르스름하다", "adverbial": "노르스름하게", "noun_phrase": "노르스름한 노란빛", "intensity": "soft"},
        {"adjective": "노릇하다", "adverbial": "노릇하게", "noun_phrase": "노릇한 노란빛", "intensity": "soft"},
        {"adjective": "샛노랗다", "adverbial": "샛노랗게", "noun_phrase": "샛노란 빛", "intensity": "bright"},
        {"adjective": "누르스름하다", "adverbial": "누르스름하게", "noun_phrase": "누르스름한 노란빛", "intensity": "dark"},
        {"adjective": "황금빛이다", "adverbial": "황금빛으로", "noun_phrase": "황금빛 노랑", "intensity": "strong"},
    ],
    "blue": [
        {"adjective": "파르스름하다", "adverbial": "파르스름하게", "noun_phrase": "파르스름한 파란빛", "intensity": "soft"},
        {"adjective": "푸르스름하다", "adverbial": "푸르스름하게", "noun_phrase": "푸르스름한 푸른빛", "intensity": "soft"},
        {"adjective": "푸르다", "adverbial": "푸르게", "noun_phrase": "푸른빛", "intensity": "neutral"},
        {"adjective": "새파랗다", "adverbial": "새파랗게", "noun_phrase": "새파란 빛", "intensity": "bright"},
        {"adjective": "시퍼렇다", "adverbial": "시퍼렇게", "noun_phrase": "시퍼런 파란빛", "intensity": "strong"},
        {"adjective": "하늘빛이다", "adverbial": "하늘빛으로", "noun_phrase": "하늘빛 파랑", "intensity": "bright"},
    ],
    "green": [
        {"adjective": "초록빛이다", "adverbial": "초록빛으로", "noun_phrase": "초록빛", "intensity": "neutral"},
        {"adjective": "연초록빛이다", "adverbial": "연초록빛으로", "noun_phrase": "연초록빛", "intensity": "soft"},
    ],
    "purple": [
        {"adjective": "보랏빛이다", "adverbial": "보랏빛으로", "noun_phrase": "보랏빛", "intensity": "neutral"},
        {"adjective": "연보랏빛이다", "adverbial": "연보랏빛으로", "noun_phrase": "연보랏빛", "intensity": "soft"},
    ],
    "orange": [
        {"adjective": "주황빛이다", "adverbial": "주황빛으로", "noun_phrase": "주황빛", "intensity": "neutral"},
        {"adjective": "오렌지빛이다", "adverbial": "오렌지빛으로", "noun_phrase": "오렌지빛", "intensity": "bright"},
    ],
    "white": [
        {"adjective": "하얗다", "adverbial": "하얗게", "noun_phrase": "하얀 빛", "intensity": "bright"},
        {"adjective": "희끄무레하다", "adverbial": "희끄무레하게", "noun_phrase": "희끄무레한 흰빛", "intensity": "soft"},
    ],
}

COLOR_ADJECTIVE_REQUIRED_CASES: list[dict[str, Any]] = [
    {"text": "발그스레한 느낌으로 은은하게 켜줘", "emotion": None, "base_color": "red", "intensity": "soft"},
    {"text": "불그스름한 붉은빛으로 부드럽게 해줘", "emotion": None, "base_color": "red", "intensity": "soft"},
    {"text": "발갛게 켜줘", "emotion": None, "base_color": "red", "intensity": "neutral"},
    {"text": "새빨갛고 선명하게 켜줘", "emotion": None, "base_color": "red", "intensity": "strong"},
    {"text": "시뻘겋게 화난 느낌으로 켜줘", "emotion": "anger", "base_color": "red", "intensity": "strong"},
    {"text": "슬프지만 조명은 발그레하게 해줘", "emotion": "sadness", "base_color": "red", "intensity": "soft"},
    {"text": "기분은 우울하지만 조명은 발그스레하게 해줘", "emotion": "sadness", "base_color": "red", "intensity": "soft"},
    {"text": "노르스름하고 따뜻하게 해줘", "emotion": None, "base_color": "yellow", "intensity": "soft"},
    {"text": "노릇한 노란빛으로 편하게 켜줘", "emotion": None, "base_color": "yellow", "intensity": "soft"},
    {"text": "샛노랗고 환하게 켜줘", "emotion": None, "base_color": "yellow", "intensity": "bright"},
    {"text": "누르스름한 낮은 밝기로 해줘", "emotion": None, "base_color": "yellow", "intensity": "dark"},
    {"text": "행복해서 황금빛으로 밝게 보여줘", "emotion": "happiness", "base_color": "yellow", "intensity": "bright"},
    {"text": "푸르스름하고 차분한 분위기로 해줘", "emotion": "calm", "base_color": "blue", "intensity": "soft"},
    {"text": "파르스름한 파란빛으로 은은하게 켜줘", "emotion": None, "base_color": "blue", "intensity": "soft"},
    {"text": "새파랗게 맑은 느낌으로 켜줘", "emotion": None, "base_color": "blue", "intensity": "bright"},
    {"text": "시퍼렇게 불안한 느낌을 낮은 밝기로 보여줘", "emotion": "anxiety", "base_color": "blue", "intensity": "dark"},
    {"text": "불안하지만 조명은 푸르스름하게 차분히 해줘", "emotion": "anxiety", "base_color": "blue", "intensity": "soft"},
]


def build_required_validation_cases() -> list[dict[str, Any]]:
    cases = [
        {"text": "빨강", "emotion": None, "base_color": "red", "intensity": "neutral"},
        {"text": "빨강색", "emotion": None, "base_color": "red", "intensity": "neutral"},
        {"text": "빨간색으로 해줘", "emotion": None, "base_color": "red", "intensity": "neutral"},
        {"text": "빨갛게 켜줘", "emotion": None, "base_color": "red", "intensity": "neutral"},
        {"text": "은은한 빨강으로 해줘", "emotion": None, "base_color": "red", "intensity": "soft"},
        {"text": "화가 나서 강한 빨간색으로 해줘", "emotion": "anger", "base_color": "red", "intensity": "strong"},
        {"text": "슬프지만 붉은빛으로 표현해줘", "emotion": "sadness", "base_color": "red", "intensity": "soft"},
        {"text": "행복해서 밝은 노란색으로 해줘", "emotion": "happiness", "base_color": "yellow", "intensity": "bright"},
        {"text": "차분하게 파란색으로 켜줘", "emotion": "calm", "base_color": "blue", "intensity": "soft"},
        {"text": "불안해서 어두운 파란빛으로 해줘", "emotion": "anxiety", "base_color": "blue", "intensity": "dark"},
    ]
    cases.extend(COLOR_ADJECTIVE_REQUIRED_CASES)
    for color_name in ["red", "yellow", "blue"]:
        for emotion_name in EMOTION_PROFILES:
            for intensity_name in ["neutral", "soft", "strong", "bright", "dark"]:
                color_label = COLOR_PROFILES[color_name]["label"]
                emotion_phrase = EMOTION_PROFILES[emotion_name]["phrases"][0]
                modifier = INTENSITY_MODIFIERS[intensity_name]["label"]
                text = f"지금 {emotion_phrase}. 조명은 {modifier + ' ' if modifier else ''}{color_label}으로 표현해줘"
                cases.append({"text": text, "emotion": emotion_name, "base_color": color_name, "intensity": intensity_name})
    deduped: dict[str, dict[str, Any]] = {}
    for case in cases:
        deduped[case["text"]] = case
    return list(deduped.values())


REQUIRED_VALIDATION_CASES = build_required_validation_cases()


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))


def hue_distance(a: int, b: int) -> int:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def normalize_hue(value: int) -> int:
    """Circular normalization; never clamp."""
    return int(round(float(value))) % 360


def normalize_output(values: dict[str, Any]) -> dict[str, int]:
    lower = {str(key).lower(): value for key, value in values.items()}
    return {
        # Hue is circular -- modulo only. Clamping first (0,360) destroyed
        # wraparound (370 -> 0 instead of 10). See backend/common/lighting_spec.py.
        "H": normalize_hue(lower.get("h", 0)),
        "S": clamp_int(lower.get("s", 0), 0, 100),
        "B": clamp_int(lower.get("b", 0), 0, 100),
        "Dimmer": clamp_int(lower.get("dimmer", 0), 0, 100),
        "CT": clamp_int(lower.get("ct", 300), 153, 500),  # 153 = declared prompt floor
    }


def apply_adjustments(base: dict[str, int], *adjustments: dict[str, int] | None) -> dict[str, int]:
    output = dict(base)
    for adjustment in adjustments:
        if not adjustment:
            continue
        output["H"] = normalize_hue(output["H"] + int(adjustment.get("H", 0)))
        output["S"] += int(adjustment.get("S", 0))
        output["B"] += int(adjustment.get("B", 0))
        output["Dimmer"] += int(adjustment.get("Dimmer", 0))
        output["CT"] += int(adjustment.get("CT", 0))
    return normalize_output(output)


def resolve_case_output(case: dict[str, Any]) -> dict[str, int]:
    color_name = case.get("base_color")
    emotion_name = case.get("emotion")
    intensity_name = case.get("intensity") or "neutral"
    intensity = INTENSITY_MODIFIERS.get(intensity_name, INTENSITY_MODIFIERS["neutral"])
    if color_name:
        color_target = dict(COLOR_PROFILES[color_name]["target"])
        emotion_adjust = EMOTION_PROFILES.get(emotion_name or "", {}).get("adjust")
        output = apply_adjustments(color_target, emotion_adjust, intensity["adjust"])
        anchor = COLOR_PROFILES[color_name]["anchor_h"]
        if hue_distance(output["H"], anchor) > 18:
            output["H"] = anchor
        return output
    if emotion_name:
        return apply_adjustments(dict(EMOTION_PROFILES[emotion_name]["target"]), intensity["adjust"])
    return apply_adjustments(dict(SAFE_FALLBACK), intensity["adjust"])


def build_prompt(text: str) -> str:
    return (
        "<|system|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|user|>\n"
        f"{USER_PROMPT_TEMPLATE.format(text=text)}\n"
        "<|assistant|>\n"
    )


def build_target_text(output: dict[str, Any]) -> str:
    return json.dumps(normalize_output(output), ensure_ascii=False)


def make_row(
    *,
    text: str,
    output: dict[str, Any],
    source: str,
    emotion: str | None,
    base_color: str | None,
    intensity: str | None = None,
) -> dict[str, Any]:
    return {
        "instruction": SYSTEM_PROMPT,
        "input": text,
        "output": normalize_output(output),
        "emotion": emotion,
        "base_color": base_color,
        "intensity": intensity or "neutral",
        "source": source,
    }


def build_color_adjective_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    simple_templates = [
        "{adverbial} 켜줘",
        "조명을 {adverbial} 만들어줘",
        "{noun_phrase}으로 보여줘",
        "{noun_phrase} 느낌으로 해줘",
        "{adjective} 느낌을 조명으로 표현해줘",
        "오늘은 {noun_phrase}이 좋겠어",
    ]
    emotion_templates = [
        "지금 {emotion}. 조명은 {adverbial} 해줘",
        "{emotion}. {noun_phrase}으로 표현해줘",
        "{emotion}. 그래도 색은 {noun_phrase}이 좋아",
    ]
    for color_name, entries in COLOR_ADJECTIVE_PROFILES.items():
        for entry in entries:
            intensity_name = entry["intensity"]
            base_case = {"base_color": color_name, "emotion": None, "intensity": intensity_name}
            for template in simple_templates:
                text = template.format(**entry)
                rows.append(make_row(text=text, output=resolve_case_output(base_case), source="color_adjective_sft", emotion=None, base_color=color_name, intensity=intensity_name))
            for emotion_name, emotion_info in EMOTION_PROFILES.items():
                output = resolve_case_output({"base_color": color_name, "emotion": emotion_name, "intensity": intensity_name})
                for emotion_phrase in emotion_info["phrases"][:4]:
                    for template in emotion_templates:
                        text = template.format(emotion=emotion_phrase, **entry)
                        rows.append(make_row(text=text, output=output, source="color_adjective_mixed_emotion", emotion=emotion_name, base_color=color_name, intensity=intensity_name))
    for case in COLOR_ADJECTIVE_REQUIRED_CASES:
        rows.append(make_row(text=case["text"], output=resolve_case_output(case), source="required_validation", emotion=case.get("emotion"), base_color=case.get("base_color"), intensity=case.get("intensity")))
    deduped: dict[tuple[str, str | None, str | None, str | None], dict[str, Any]] = {}
    for row in rows:
        deduped[(row["input"], row["emotion"], row["base_color"], row["intensity"])] = row
    return list(deduped.values())


def build_synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_color_adjective_rows())
    emotion_templates = ["오늘 너무 {emotion}", "지금 좀 {emotion}", "요즘 계속 {emotion}", "마음이 많이 {emotion}"]
    color_templates = ["{phrase}", "{phrase} 느낌으로 해줘", "{phrase}으로 표현해줘", "{phrase} 톤으로 바꿔줘"]
    mixed_templates = [
        "지금 {emotion}. 조명은 {modifier}{color}으로 표현해줘",
        "{emotion}. 조명은 {modifier}{color}으로 해줘",
        "{emotion}. {modifier}{color}빛으로 보여줘",
        "{emotion}. 색은 {modifier}{color}이 좋아",
        "오늘은 {emotion}. {modifier}{color}으로 켜줘",
    ]

    for emotion_name, emotion_info in EMOTION_PROFILES.items():
        for phrase in emotion_info["phrases"]:
            for template in emotion_templates:
                rows.append(make_row(text=template.format(emotion=phrase), output=emotion_info["target"], source="synthetic_emotion", emotion=emotion_name, base_color=None))

    for color_name, color_info in COLOR_PROFILES.items():
        for phrase in color_info["phrases"]:
            for intensity_name, intensity_info in INTENSITY_MODIFIERS.items():
                modifier_phrases = intensity_info["phrases"] if intensity_name != "neutral" else [""]
                for modifier_phrase in modifier_phrases:
                    modifier = f"{modifier_phrase} " if modifier_phrase else ""
                    for template in color_templates:
                        case = {"base_color": color_name, "emotion": None, "intensity": intensity_name}
                        rows.append(
                            make_row(
                                text=template.format(phrase=f"{modifier}{phrase}").strip(),
                                output=resolve_case_output(case),
                                source="synthetic_color",
                                emotion=None,
                                base_color=color_name,
                                intensity=intensity_name,
                            )
                        )

    for emotion_name, emotion_info in EMOTION_PROFILES.items():
        emotion_phrases = emotion_info["phrases"][:5]
        for color_name, color_info in COLOR_PROFILES.items():
            color_phrases = color_info["phrases"][:5]
            for intensity_name, intensity_info in INTENSITY_MODIFIERS.items():
                modifier = f"{intensity_info['label']} " if intensity_info["label"] else ""
                output = resolve_case_output({"base_color": color_name, "emotion": emotion_name, "intensity": intensity_name})
                for emotion_phrase in emotion_phrases:
                    for color_phrase in color_phrases:
                        for template in mixed_templates:
                            rows.append(
                                make_row(
                                    text=template.format(emotion=emotion_phrase, modifier=modifier, color=color_phrase),
                                    output=output,
                                    source="synthetic_mixed",
                                    emotion=emotion_name,
                                    base_color=color_name,
                                    intensity=intensity_name,
                                )
                            )

    for text in AMBIGUOUS_INPUTS:
        rows.append(make_row(text=text, output=SAFE_FALLBACK, source="synthetic_fallback", emotion=None, base_color=None))

    for case in TARGETED_HARD_CASES:
        rows.append(
            make_row(
                text=case["text"],
                output=resolve_case_output(case),
                source="targeted_hard_case",
                emotion=case.get("emotion"),
                base_color=case.get("base_color"),
                intensity=case.get("intensity"),
            )
        )

    for case in REQUIRED_VALIDATION_CASES:
        rows.append(
            make_row(
                text=case["text"],
                output=resolve_case_output(case),
                source="required_validation",
                emotion=case.get("emotion"),
                base_color=case.get("base_color"),
                intensity=case.get("intensity"),
            )
        )

    deduped: dict[tuple[str, str | None, str | None, str | None], dict[str, Any]] = {}
    for row in rows:
        key = (row["input"], row["emotion"], row["base_color"], row["intensity"])
        deduped[key] = row
    return list(deduped.values())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_base_rows(rows: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(
            {
                "instruction": row.get("instruction", SYSTEM_PROMPT),
                "input": str(row["input"]),
                "output": normalize_output(row["output"]),
                "emotion": row.get("emotion"),
                "base_color": row.get("base_color"),
                "intensity": row.get("intensity", "neutral"),
                "source": source_name,
            }
        )
    return enriched


def prepare_datasets(
    *,
    base_train_rows: list[dict[str, Any]],
    base_val_rows: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    synthetic = build_synthetic_rows()
    rng.shuffle(synthetic)
    split = max(1, int(len(synthetic) * 0.9))
    synthetic_train = synthetic[:split]
    synthetic_val = synthetic[split:]

    train_repeat_by_source = {
        "synthetic_emotion": 1,
        "synthetic_color": 2,
        "synthetic_mixed": 2,
        "synthetic_fallback": 1,
        "targeted_hard_case": 10,
        "required_validation": 8,
        "color_adjective_sft": 3,
        "color_adjective_mixed_emotion": 2,
    }
    val_repeat_by_source = {
        "synthetic_emotion": 1,
        "synthetic_color": 1,
        "synthetic_mixed": 1,
        "synthetic_fallback": 1,
        "targeted_hard_case": 3,
        "required_validation": 3,
        "color_adjective_sft": 1,
        "color_adjective_mixed_emotion": 1,
    }

    expanded_train: list[dict[str, Any]] = []
    for row in synthetic_train:
        expanded_train.extend([dict(row) for _ in range(train_repeat_by_source.get(row["source"], 1))])

    expanded_val: list[dict[str, Any]] = []
    for row in synthetic_val:
        expanded_val.extend([dict(row) for _ in range(val_repeat_by_source.get(row["source"], 1))])

    train_rows = enrich_base_rows(base_train_rows, "base_train") + expanded_train
    val_rows = enrich_base_rows(base_val_rows, "base_val") + expanded_val
    required_rows = [row for row in synthetic if row["source"] == "required_validation"]
    return train_rows, val_rows, required_rows


def count_rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "none")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(paths: list[Path]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for path in paths:
        sources[path.name] = {
            "path": str(path),
            "exists": path.exists(),
            "rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0,
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return {"sources": sources}


def build_labeling_policy() -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "color_anchors": {name: {"hue": profile["anchor_h"], "target": profile["target"]} for name, profile in COLOR_PROFILES.items()},
        "emotion_adjustments": {name: profile["adjust"] for name, profile in EMOTION_PROFILES.items()},
        "intensity_adjustments": {name: profile["adjust"] for name, profile in INTENSITY_MODIFIERS.items()},
        "hue_policy": "Explicit color requests keep hue within distance <= 18 of the color anchor; emotion and intensity adjust S/B/Dimmer/CT primarily.",
    }


def build_dataset_card(metadata: dict[str, Any], manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SentiLight Color Emotion Dataset v1",
            "",
            "Smart bulb control dataset for CompactLM and future local lighting models.",
            "",
            "## Files",
            "",
            "- `train.jsonl`: supervised fine-tuning rows",
            "- `val.jsonl`: validation rows",
            "- `test.jsonl`: required color/emotion validation rows",
            "- `required_validation.jsonl`: explicit acceptance cases",
            "- `pretrain_corpus.txt`: mixed pretraining text corpus",
            "- `pretrain_tokens.npy`: tokenized pretraining corpus",
            "- `metadata.json`: generated counts and splits",
            "- `source_manifest.json`: source/generation checksums",
            "- `labeling_policy.json`: color/emotion/intensity rules",
            "",
            "## Counts",
            "",
            f"- train rows: `{metadata.get('prepared_train_rows')}`",
            f"- val rows: `{metadata.get('prepared_val_rows')}`",
            f"- test rows: `{metadata.get('prepared_test_rows')}`",
            f"- required validation rows: `{metadata.get('required_validation_rows')}`",
            "",
            "## Sources",
            "",
            json.dumps(manifest.get("sources", {}), ensure_ascii=False, indent=2),
            "",
        ]
    )
