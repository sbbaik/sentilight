import os
import csv
import time
import random
import re
from typing import List, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tqdm import tqdm

# =========================
# 설정
# =========================
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 GEMINI_API_KEY 를 설정하세요.")

MODEL_NAME = "gemini-2.5-flash-lite"
INPUT_FILE = "KOTE-main/train.tsv"
OUTPUT_FILE = "labeled_kote_results.csv"

# Tier 1 기준 조정
BATCH_SIZE = 12
BASE_WAIT_BETWEEN_CALLS = 0.6
MAX_RETRIES_PER_CALL = 8
LONG_WAIT_ON_PERSISTENT_FAILURE = 8.0
MAX_OUTPUT_TOKENS = 2200

client = genai.Client(api_key=API_KEY)


# =========================
# Structured Output Schema
# =========================
class LabelItem(BaseModel):
    index: int = Field(description="입력 문장의 번호(1부터 시작)")
    h: int = Field(ge=0, le=359)
    s: int = Field(ge=0, le=100)
    b: int = Field(ge=0, le=100)
    dimmer: int = Field(ge=0, le=100)
    ct: int = Field(ge=153, le=500)


class BatchLabelResponse(BaseModel):
    results: List[LabelItem]


# =========================
# 예외
# =========================
class RetryableAPIError(Exception):
    pass


class NonRetryableAPIError(Exception):
    pass


# =========================
# 유틸
# =========================
def tasmota_command_from_values(h: int, s: int, b: int, dimmer: int, ct: int) -> str:
    return f"HSBCOLOR {h},{s},{b};Dimmer {dimmer};CT {ct}"


def sleep_with_jitter(seconds: float):
    time.sleep(seconds + random.uniform(0.0, 0.35))


def extract_retry_delay_seconds(msg: str) -> Optional[float]:
    msg_lower = msg.lower()

    patterns = [
        r"retry in\s*([0-9]+(?:\.[0-9]+)?)s",
        r"retrydelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)s",
        r"'retrydelay':\s*'([0-9]+(?:\.[0-9]+)?)s'",
        r'"retrydelay":\s*"([0-9]+(?:\.[0-9]+)?)s"',
    ]

    for pattern in patterns:
        m = re.search(pattern, msg_lower)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def classify_error(msg: str) -> str:
    msg = msg.lower()

    if "400" in msg or "invalid argument" in msg or "response.parsed" in msg:
        return "parse_or_request"
    if "429" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if "500" in msg or "503" in msg or "internal" in msg or "service unavailable" in msg:
        return "server"
    if "deadline" in msg or "timeout" in msg or "connection reset" in msg or "network" in msg:
        return "network"
    return "unknown"


def backoff_sleep(attempt: int, error_msg: str = ""):
    retry_delay = extract_retry_delay_seconds(error_msg)
    if retry_delay is not None:
        sleep_with_jitter(retry_delay)
        return

    base = min(1.5 * (2 ** attempt), 20.0)
    sleep_with_jitter(base)


def normalize_label_item(item: LabelItem) -> LabelItem:
    h = max(0, min(359, int(item.h)))
    s = max(0, min(100, int(item.s)))
    b = max(0, min(100, int(item.b)))
    dimmer = max(0, min(100, int(item.dimmer)))
    ct = max(153, min(500, int(item.ct)))

    if s == 0:
        s = 12

    if b == 100 and dimmer == 100:
        dimmer = 92

    if s < 10 and ct < 180:
        ct = 260

    return LabelItem(
        index=item.index,
        h=h,
        s=s,
        b=b,
        dimmer=dimmer,
        ct=ct
    )


def create_batch_prompt(sentence_batch: List[str]) -> str:
    lines = [f"{i}: {sentence}" for i, sentence in enumerate(sentence_batch, start=1)]

    return f"""
당신은 한국어 문장의 정서와 뉘앙스를 분석하여 Tasmota 스마트 전구 제어값으로 변환하는 고정밀 라벨링 전문가입니다.

목표는 문장을 읽고 실제 조명 제어에 쓸 수 있는 설득력 있는 h,s,b,dimmer,ct 값을 생성하는 것입니다.
모든 문장에 비슷한 기본값을 반복하는 것은 금지합니다.
문장마다 감정, 태도, 뉘앙스가 다르면 결과값도 분명히 달라져야 합니다.

반드시 아래 규칙을 지키세요.

[출력 규칙]
1. 반드시 JSON만 출력하세요.
2. results 배열 길이는 입력 문장 수와 정확히 같아야 합니다.
3. 각 항목의 index는 입력 번호와 일치해야 합니다.
4. 값 범위:
   - h: 0~359
   - s: 0~100
   - b: 0~100
   - dimmer: 0~100
   - ct: 153~500
5. 설명문, 코드블록, 주석은 절대 출력하지 마세요.

[핵심 원칙]
1. 각 문장을 내부적으로 먼저 해석하세요:
   - 주감정: 분노 / 혐오 / 냉소 / 짜증 / 귀여움 / 애정 / 기쁨 / 자랑 / 흥분 / 평온 / 슬픔 / 우울 / 불안 / 공포 / 중립 등
   - 감정 강도: 약 / 중 / 강
   - 정서 극성: 긍정 / 부정 / 혼합
   - 에너지 수준: 낮음 / 중간 / 높음
   - 따뜻한 느낌인지 차가운 느낌인지
2. 이 내부 해석을 바탕으로 최종 h,s,b,dimmer,ct 값을 결정하세요.
3. 내부 해석 과정은 출력하지 말고 최종 JSON만 출력하세요.
4. 애매하더라도 무조건 기본값으로 도망가지 말고 가장 그럴듯한 정서로 해석하세요.
5. 특별한 이유 없이 동일하거나 거의 동일한 조명값을 반복하지 마세요.
6. 특히 아래와 같은 단조 출력은 금지합니다:
   - s=0 반복
   - b=100 반복
   - dimmer=100 반복
   - ct=153 반복
   - 사실상 흰색 기본등처럼 보이는 결과 반복

[파라미터 의미 규칙]
1. Hue(h): 감정 색상 계열
- 분노 / 공격 / 위협 / 격앙: 0~25
- 혐오 / 독설 / 조롱 / 거친 비난: 330~359 또는 280~330
- 짜증 / 냉소 / 비꼼 / 불쾌감: 260~330
- 경계 / 긴장 / 불안: 35~90
- 성취 / 자신감 / 자랑 / 활력: 20~60
- 유쾌 / 장난 / 가벼운 즐거움: 45~100
- 평온 / 차분 / 담담 / 이성적 중립: 160~240
- 슬픔 / 우울 / 공허: 210~260
- 귀여움 / 애정 / 포근함 / 사랑스러움: 320~350
- 신비 / 몽환 / 비현실감: 260~320

2. Saturation(s): 감정의 선명도
- 강한 감정: 75~100
- 보통 감정: 40~74
- 중립 / 건조 / 무덤덤: 10~39
- 특별한 이유 없이 s=0 금지

3. Brightness(b): 색 자체의 발광감
- 밝고 경쾌 / 귀여움 / 칭찬 / 흥분: 70~100
- 공격적 / 분노 / 혐오 / 위협: 35~75
- 슬픔 / 냉소 / 무기력 / 우울: 20~60
- 평온 / 담담: 45~75

4. Dimmer(dimmer): 전체 조명의 체감 밝기
- 외향적 / 활기 / 자랑 / 들뜸: 70~100
- 보통: 40~69
- 무겁고 우울 / 위협 / 공격적: 15~55
- b와 dimmer를 항상 동일하게 하지 말고 약간 차이를 두세요.

5. Color Temperature(ct)
- 분노 / 공격 / 혐오 / 비난: 180~260
- 귀여움 / 애정 / 포근함: 220~320
- 활기 / 자랑 / 성취 / 자신감: 200~300
- 평온 / 중립 / 담담: 280~380
- 슬픔 / 불안 / 고독 / 냉소 / 거리감: 340~500
- 특별한 이유 없이 극단값 반복 금지

[중요한 해석 규칙]
1. 욕설, 공격적 언사, 저주, 격한 비난이 있으면 거의 항상 강한 부정 감정을 반영해야 합니다.
2. "ㅎㅎ", "헤헤", "귀엽", "구엽", "사랑스럽", "포근" 등은 귀여움/애정/따뜻함을 반영해야 합니다.
3. 자랑, 능력 과시, 자신감은 밝기와 채도를 어느 정도 높이되, 분노 계열과 혼동하지 마세요.
4. 비꼼, 냉소, 짜증은 단순 분노와 다르게 약간 차갑거나 보라/자주 계열도 적극 고려하세요.
5. 공격적 문장에 흰색 기본조명 같은 결과는 금지합니다.
6. 귀엽고 사랑스러운 문장에 차갑고 무채색인 결과는 금지합니다.
7. 중립이 아닌데도 h,s,b,dimmer,ct가 지나치게 평범하거나 일률적이면 안 됩니다.

[출력 다양성 규칙]
1. 같은 배치 안에서 문장들의 감정이 다르면 결과도 확실히 달라야 합니다.
2. 완전히 같은 의미가 아닌데도 같은 숫자 조합을 재사용하지 마세요.
3. 단, 의미가 매우 유사하면 어느 정도 비슷한 범위는 허용됩니다.
4. 숫자는 실제 조명 제어에 쓸 수 있도록 자연스럽고 설득력 있게 선택하세요.
5. 기계적으로 균등분포시킬 필요는 없지만, 의미 차이가 있으면 값 차이도 반영하세요.

[예시적 경향]
- "댓글주작하지마라 드르븐것들아 양심도 없냐"
  -> 공격적 분노, 혐오, 비난
  -> h는 적색/자주 계열, s 높음, b 중간, dimmer 중간 이하, ct는 낮거나 중간

- "유치해 메시지가 너무 노골적이어서 보다가 물린다"
  -> 짜증, 냉소, 불쾌
  -> h는 자주/보라 계열 가능, s 중간 이상, b 중간 이하, dimmer 과도하게 높지 않음, ct는 중간~차가운 편 가능

- "ㅎㅎ 얼마나 천방지축 뛰어다녔으면 저렇게하고 잘까요. 너무 구엽네요."
  -> 귀여움, 애정, 포근함
  -> h는 분홍/자홍 계열, s 중~높음, b 높음, dimmer 밝은 편, ct는 따뜻하거나 중간

- "운전병을 이럴때 써먹는구나... 무사고 운행 가능 헤헤"
  -> 자랑, 능력, 가벼운 유쾌함
  -> h는 주황/노랑 계열 가능, s 중간 이상, b 밝은 편, dimmer 중~높음, ct는 따뜻한 편

[입력]
{chr(10).join(lines)}
""".strip()


def parse_and_validate_response(parsed, batch_size: int) -> List[LabelItem]:
    if not parsed or not hasattr(parsed, "results"):
        raise RetryableAPIError("response.parsed 가 비어 있습니다.")

    results = parsed.results
    if len(results) != batch_size:
        raise RetryableAPIError(
            f"응답 개수 불일치: expected={batch_size}, got={len(results)}"
        )

    results = sorted(results, key=lambda x: x.index)
    expected_indexes = list(range(1, batch_size + 1))
    got_indexes = [x.index for x in results]
    if got_indexes != expected_indexes:
        raise RetryableAPIError(
            f"index 불일치: expected={expected_indexes}, got={got_indexes}"
        )

    return [normalize_label_item(item) for item in results]


def call_gemini_batch_once(batch: List[str]) -> List[LabelItem]:
    prompt = create_batch_prompt(batch)

    for attempt in range(MAX_RETRIES_PER_CALL):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_schema=BatchLabelResponse,
                ),
            )
            return parse_and_validate_response(response.parsed, len(batch))

        except Exception as e:
            msg = str(e)
            error_type = classify_error(msg)

            if error_type in ("rate_limit", "server", "network", "parse_or_request"):
                if attempt < MAX_RETRIES_PER_CALL - 1:
                    backoff_sleep(attempt, msg)
                    continue
                raise RetryableAPIError(msg)

            raise NonRetryableAPIError(msg)


def call_gemini_batch_with_split(batch: List[str]) -> List[LabelItem]:
    """
    한 번의 배치 시도에서 너무 오래 막히면 분할로 내려감.
    """
    try:
        return call_gemini_batch_once(batch)
    except RetryableAPIError:
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        left = call_gemini_batch_with_split(batch[:mid])
        right = call_gemini_batch_with_split(batch[mid:])
        return left + right


def call_gemini_batch_until_success(batch: List[str]) -> List[LabelItem]:
    round_idx = 0
    while True:
        try:
            return call_gemini_batch_with_split(batch)
        except RetryableAPIError as e:
            round_idx += 1
            tqdm.write(
                f"[WARN] 배치 재시도 예정 (size={len(batch)}, round={round_idx}): {e}"
            )
            sleep_with_jitter(LONG_WAIT_ON_PERSISTENT_FAILURE)


def count_existing_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
        if not rows:
            return 0

        header = rows[0]
        if header and header[0] == "sentence":
            return len(rows) - 1
        return len(rows)


def ensure_output_header(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label", "h", "s", "b", "dimmer", "ct"])


def load_sentences(input_file: str) -> List[str]:
    result = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                sentence = row[1].strip()
                if sentence and sentence.lower() != "sentence":
                    result.append(sentence)
    return result


def append_success_rows(path: str, batch: List[str], items: List[LabelItem]):
    if len(batch) != len(items):
        raise ValueError("배치 길이와 결과 길이가 다릅니다.")

    rows = []
    for sentence, item in zip(batch, items):
        label = tasmota_command_from_values(
            item.h, item.s, item.b, item.dimmer, item.ct
        )
        rows.append([sentence, label, item.h, item.s, item.b, item.dimmer, item.ct])

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    sentences = load_sentences(INPUT_FILE)
    ensure_output_header(OUTPUT_FILE)

    done = count_existing_rows(OUTPUT_FILE)
    if done > len(sentences):
        raise RuntimeError("OUTPUT_FILE 행 수가 INPUT_FILE 보다 많습니다. 출력 파일을 확인하세요.")

    remaining = sentences[done:]
    pbar = tqdm(total=len(sentences), initial=done, desc="[Labeling]")

    idx = 0
    while idx < len(remaining):
        batch = remaining[idx: idx + BATCH_SIZE]

        items = call_gemini_batch_until_success(batch)
        append_success_rows(OUTPUT_FILE, batch, items)

        pbar.update(len(batch))
        idx += len(batch)

        sleep_with_jitter(BASE_WAIT_BETWEEN_CALLS)

    pbar.close()
    print("작업이 완료되었습니다.")


if __name__ == "__main__":
    main()