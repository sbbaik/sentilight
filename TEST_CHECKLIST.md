# GUI Text Input & Space Key Fix - Test Checklist

## Implementation Summary
Fixed Space key and text input handling to allow proper text entry while supporting STT recording.

### Key Changes
1. **Focus-Aware Space Key**: Space only triggers STT recording when text input is NOT focused
2. **Enter Key Submission**: Enter submits text to backend; Shift+Enter creates newline
3. **Duplicate Prevention**: Locks UI during recording/STT/analysis

---

## Test Cases

### ✓ Test 1: Text Input with Spaces
**Procedure:**
1. Launch GUI app
2. Click on text input field to focus it
3. Type: "오늘 기분이 좋아요"

**Expected Result:**
- Text appears with proper spacing: "오늘 기분이 좋아요"
- NOT "오늘기분이좋아요" (missing spaces)
- Space bar does NOT trigger recording

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 2: Enter Key Submission
**Procedure:**
1. Focus text input field
2. Type: "안녕하세요"
3. Press Enter key

**Expected Result:**
- Text submits to backend
- Log shows: "입력 문장: 안녕하세요"
- "말하기" button changes to "분석 중..."
- Five model panels update with results

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 3: Shift+Enter Newline
**Procedure:**
1. Focus text input field
2. Type: "첫 번째 문장"
3. Press Shift+Enter
4. Type: "두 번째 문장"

**Expected Result:**
- Text appears on two lines in input field:
  ```
  첫 번째 문장
  두 번째 문장
  ```
- No submission to backend

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 4: Space Long-Press (≥400ms) - Outside Text Input
**Procedure:**
1. Focus on a different widget (e.g., "Settings" button)
2. Hold Space bar for 500ms (longer than 400ms threshold)
3. Release Space bar

**Expected Result:**
- Log shows: "녹음 중입니다. 말씀하세요."
- "말하기" button changes to "녹음 중..."
- Audio recording starts
- Recording continues for 200ms after release
- STT begins after recording stops

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 5: Space Quick-Press - Outside Text Input
**Procedure:**
1. Focus on a different widget
2. Quickly press Space bar for <200ms
3. Release Space bar

**Expected Result:**
- No recording starts
- No audio feedback
- Log remains unchanged

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 6: "말하기" Button Press/Release
**Procedure:**
1. Click "말하기" button and hold for 1 second
2. Release button

**Expected Result:**
- Log shows: "녹음 중입니다. 말씀하세요."
- Recording continues for 200ms after button release
- Log shows: "녹음 종료. Gemini STT 처리 중입니다."
- After STT completes, backend submits and results appear

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 7: Duplicate Execution Prevention - Space During Recording
**Procedure:**
1. Press Space bar to start recording (hold ≥400ms)
2. While recording, press Space bar again
3. Wait for recording to complete

**Expected Result:**
- Second Space bar press is ignored
- Only one recording captured
- No duplicate backend calls

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 8: Duplicate Execution Prevention - Enter During Recording
**Procedure:**
1. Focus text input
2. Type: "테스트"
3. Press Space bar to start STT recording (hold ≥400ms)
4. While recording, press Enter in text input (if possible)

**Expected Result:**
- Enter is blocked/ignored during recording
- Only one backend call from STT

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 9: Space -> Text -> Enter Workflow
**Procedure:**
1. Use Space bar long-press to start STT recording
2. Say: "오늘 기분이 좋아요"
3. Release Space bar
4. Verify transcribed text appears in input with spaces
5. Press Enter to submit

**Expected Result:**
- Space long-press triggered STT
- Text inserted correctly: "오늘 기분이 좋아요"
- Enter submission works
- Backend results appear

**Status:** [ ] Pass [ ] Fail

---

### ✓ Test 10: Text Focus Back to Unfocused
**Procedure:**
1. Focus text input field
2. Type: "첫 번째"
3. Click elsewhere to unfocus
4. Hold Space bar ≥400ms

**Expected Result:**
- Text retained: "첫 번째"
- Space triggers recording (after focus lost)

**Status:** [ ] Pass [ ] Fail

---

## Summary

**Total Tests:** 10  
**Passed:** [ ] / 10  
**Failed:** [ ] / 10  

### Notes
- Test all scenarios before deployment
- Log files for debugging: Check terminal output or app logs
- Config files: `config/frontend.yaml`, `config/stt.yaml`
- Gemini API key must be set in Settings for STT to work
