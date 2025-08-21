# MRDA Corpus Label Analysis: Content vs Non-Content Classification

## Overview
Analysis of dialogue act labels in the MRDA corpus for binary classification into "content" (meaningful dialogue contributions) and "non-content" (conversational mechanics, noise, backchannels).

## Basic DA Labels (5 labels)

### Content Labels
**Short Names:** S, Q  
**Long Names:**
- S: Statement (64,233)
- Q: Question (6,983)

### Non-Content Labels
**Short Names:** B, D, F  
**Long Names:**
- B: BackChannel (14,620)
- D: Disruption (14,548)
- F: FloorGrabber (7,818)

### Basic DA Statistics
- **Content:** S (64,233) + Q (6,983) = **71,216 utterances (65.8%)**
- **Non-Content:** B (14,620) + D (14,548) + F (7,818) = **36,986 utterances (34.2%)**

## General DA Labels (12 labels)

### Content Labels
**Short Names:** s, qy, qw, qr, qo, qh, qrr  
**Long Names:**
- s: Statement (69,873)
- qy: Yes-No-question (4,986)
- qw: Wh-Question (1,707)
- qh: Rhetorical Question (352)
- qrr: Or-Clause (392)
- qr: Or Question (207)
- qo: Open-ended Question (169)

### Non-Content Labels
**Short Names:** b, fh, %, fg, h  
**Long Names:**
- b: Continuer (15,167)
- fh: Floor Holder (8,362)
- %: Interrupted/Abandoned/Uninterpretable (3,103)
- fg: Floor Grabber (3,092)
- h: Hold Before Answer/Agreement (792)

### General DA Statistics
- **Content:** s (69,873) + qy (4,986) + qw (1,707) + qh (352) + qrr (392) + qr (207) + qo (169) = **77,686 utterances (71.8%)**
- **Non-Content:** b (15,167) + fh (8,362) + % (3,103) + fg (3,092) + h (792) = **30,516 utterances (28.2%)**

## Full DA Labels (52 labels)

### Content Labels
**Short Names:** s, df, e, cs, d, qw, co, qy, cc, am, qrr, t, qh, tc, qr, qo  
**Long Names:**
- s: Statement (33,472)
- df: Defending/Explanation (3,724)
- e: Expansions of y/n Answers (3,200)
- cs: Offer (2,662)
- d: Declarative-Question (1,805)
- qw: Wh-Question (951)
- co: Action-directive (674)
- qy: Yes-No-question (669)
- cc: Commit (371)
- am: Maybe (349)
- qrr: Or-Clause (345)
- t: About-task (253)
- qh: Rhetorical-Question (214)
- tc: Topic Change (212)
- qr: Or-Question (127)
- qo: Open-Question (74)

### Non-Content Labels
**Short Names:** b, fh, bk, aa, %, rt, fg, ba, bu, na, ar, 2, no, h, nd, j, bd, ng, fe, m, fa, br, aap, r, t1, t3, bh, bsc, arp, bs, f, ft, g, bc, by, fw  
**Long Names:**
- b: Continuer (15,013)
- fh: Floor Holder (8,362)
- bk: Acknowledge-answer (7,177)
- aa: Accept (5,898)
- %: Interrupted/Abandoned/Uninterpretable (3,103)
- rt: Rising Tone (3,101)
- fg: Floor Grabber (3,092)
- ba: Assessment/Appreciation (2,216)
- bu: Understanding Check (2,091)
- na: Affirmative Non-yes Answers (1,112)
- ar: Reject (908)
- 2: Collaborative Completion (841)
- no: Other Answers (828)
- h: Hold Before Answer/Agreement (792)
- nd: Dispreferred Answers (483)
- j: Humorous Material (463)
- bd: Downplayer (387)
- ng: Negative Non-no Answers (351)
- fe: Exclamation (307)
- m: Mimic Other (293)
- fa: Apology (259)
- br: Signal-non-understanding (236)
- aap: Accept-part (219)
- r: Repeat (208)
- t1: Self-talk (198)
- t3: 3rd-party-talk (165)
- bh: Rhetorical-question Continue (154)
- bsc: Reject-part (150)
- arp: Misspeak Self-Correction (150)
- bs: Reformulate/Summarize (141)
- f: "Follow Me" (128)
- ft: Thanking (119)
- g: Tag-Question (87)
- bc: Correct-misspeaking (51)
- by: Sympathy (11)
- fw: Welcome (6)

### Full DA Statistics
- **Content:** 16 content labels = **49,102 utterances (45.4%)**
- **Non-Content:** 36 non-content labels = **59,100 utterances (54.6%)**

## Summary

| Granularity Level | Content Percentage | Non-Content Percentage | Content Count | Non-Content Count |
|------------------|-------------------|----------------------|---------------|-------------------|
| **Basic DA (5 labels)** | **65.8%** | **34.2%** | 71,216 | 36,986 |
| **General DA (12 labels)** | **71.8%** | **28.2%** | 77,686 | 30,516 |
| **Full DA (52 labels)** | **45.4%** | **54.6%** | 49,102 | 59,100 |

**Recommendation:** Use **General DA labels** for best content/non-content distinction with 71.8% content ratio, or **Basic DA** for simpler model with 65.8% content ratio.
