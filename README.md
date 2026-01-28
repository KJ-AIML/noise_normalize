# Audio QC / EDA for Call Center WAV (`main.py`)

สคริปต์นี้ใช้ทำ “ตรวจสุขภาพไฟล์เสียง” และทำ EDA สำหรับงาน ASR/Call Center โดยสรุปผลเป็น **CSV** (พร้อม optional JSON segments) เพื่อให้คุณ:

* คัดไฟล์เสีย/คุณภาพต่ำก่อนเข้า ASR
* หา outlier เช่น เสียงเบาเกิน, คลิป, dead air, noise สูง, dropouts
* เช็คความพร้อมสำหรับงาน diarization/overlap (โดยเฉพาะถ้าเป็น stereo)
* ทำ dashboard/สถิติ dataset ได้เหมือนทำ EDA กับข้อมูลตาราง

> หมายเหตุ: บาง metric เป็น “proxy” (ตัวแทนเชิงประมาณ) ไม่ใช่ตัววัดสุดท้ายระดับงานวิจัย แต่เพียงพอมากสำหรับ QC และค้นหา outliers

---

## ติดตั้ง

```bash
pip install numpy soundfile scipy
```

* `soundfile` ใช้สำหรับอ่าน wav แบบถูกต้อง
* `scipy` จำเป็นสำหรับบางฟีเจอร์ เช่น LUFS (ประมาณ), Welch PSD, filter

---

## วิธีรัน

### วิเคราะห์โฟลเดอร์

```powershell
uv run .\main.py --input ".\data\calls" --recursive --out_csv ".\qc_report.csv" --out_json_dir ".\qc_json"
```

### วิเคราะห์ไฟล์เดียว

```powershell
uv run .\main.py --input ".\data\calls\sample_90s_test.wav" --out_csv ".\qc_report.csv" --out_json_dir ".\qc_json"
```

### ปิด spectral (เร็วขึ้น)

```powershell
uv run .\main.py --input ".\data\calls" --recursive --no_spectral --out_csv ".\qc_report.csv"
```

---

## API (FastAPI)

### Run server

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 1) Single file (upload)

**Endpoint:** `POST /api/v1/analyze/file`  
**Form field:** `file` (wav)  
**Query params:** `save_json` (bool), `out_dir` (path), `include_segments` (bool), `no_spectral` (bool)

Example:

```powershell
curl -F "file=@.\data\calls\sample_90s_test.wav" "http://127.0.0.1:8000/api/v1/analyze/file?save_json=true&out_dir=qc_results"
```

### 2) Batch (folder)

**Endpoint:** `POST /api/v1/analyze/batch`  
**Body:** JSON

```json
{
  "input_path": ".\\data\\calls",
  "recursive": true,
  "out_dir": "qc_results",
  "include_segments": false,
  "save_json": true,
  "return_results": false
}
```

Results are saved in `out_dir` using the same base filename as the input wav (e.g., `sample_90s_test.wav` → `sample_90s_test.json`).

### Health check

`GET /health`

---

## Output

### 1) `qc_report.csv`

แถวละ 1 ไฟล์ ประกอบด้วย metadata + metrics + `flags`

### 2) `qc_json/xxx.json` (optional)

* speech segments (VAD)
* per-channel speech segments
* ใช้ทำ visualization / debug ได้

---

## คำศัพท์พื้นฐานที่ควรรู้

### dBFS คืออะไร?

* **dBFS (decibels relative to full scale)**: หน่วยเดซิเบลเทียบกับ “ระดับสูงสุดที่ไฟล์ดิจิทัลเก็บได้”
* `0 dBFS` คือ “เต็มสเกล” (ใกล้จุดที่เริ่มคลิป)
* ค่า RMS/Peak มักจะเป็นค่าติดลบ เช่น `-18 dBFS`

### RMS vs Peak

* **Peak**: จุดที่ดังที่สุด (ไวต่อคลิป)
* **RMS**: ความดังเฉลี่ยเชิงพลังงาน (สะท้อนความดังโดยรวมมากกว่า)

### LUFS คืออะไร?

* วัดความดังตามการรับรู้ (perceptual loudness) นิยมในงานเสียงสมัยใหม่
* ในสคริปต์นี้เป็น **approx** (ใช้ K-weighting proxy + gating)

---

## Metrics อธิบายทีละตัว

> แนวคิดหลัก: แยก metric เป็น 5 หมวด
> **(A) Metadata** (B) Level/Loudness (C) Speech/Silence/VAD (D) Noise/Artifacts (E) Stereo/Conversation dynamics (F) Spectral & hum & echo

---

# A) Metadata / Sanity

### `sample_rate`

**ดูทำไม:** โทรศัพท์มัก 8kHz (narrowband) / 16kHz (wideband)
**ผลต่อ ASR:** 8k จะตัดความถี่สูง ทำให้ความชัดลดลง
**แปลผล:** ถ้า dataset ปน sample rate เยอะ → ควร resample ให้เป็นมาตรฐานก่อนเข้าโมเดลเดียวกัน

### `channels`

* mono = 1 ช่อง
* stereo = 2 ช่อง (บางระบบแยก Agent/Caller)

**ดูทำไม:**

* ถ้าแยกคนละ channel จะได้ diarization ง่ายมาก และ QC เรื่อง overlap/crosstalk ทำได้ดีขึ้น

### `duration_sec`

**ดูทำไม:** คัดไฟล์สั้น/ยาวผิดปกติ (pipeline หลุด, ตัดผิดช่วง)

### `format`, `subtype`

**ดูทำไม:** ตรวจ encoding (PCM16/float/ulaw ฯลฯ) เพื่อความเข้ากันกับ pipeline

---

# B) Level / Loudness / Clipping

### `peak_dbfs_max` (+ `peak_dbfs_chX`)

**คือ:** Peak ของสัญญาณใน dBFS
**คิดยังไง:** `peak = max(|x|)` แล้วแปลงเป็น dBFS
**ดูทำไม:** ถ้า peak ใกล้ 0 dBFS มาก ๆ มีโอกาสคลิป/แตก
**แปลผล:** peak สูงมาก + clipping_pct สูง → เสียงแตก ทำให้ ASR เพี้ยนหนัก

---

### `rms_dbfs_mean` (+ `rms_dbfs_chX`)

**คือ:** RMS (พลังงานเฉลี่ย) ใน dBFS
**คิดยังไง:** `rms = sqrt(mean(x^2))` → dB
**ดูทำไม:** ช่วยวัดว่าไฟล์โดยรวม “เบาไป/ดังไป”
**แปลผล:** rms ต่ำมาก → ASR มักหลุดคำ / VAD ตัดผิด

---

### `crest_db_mean`

**คือ:** crest factor (Peak - RMS)
**ดูทำไม:** ถ้า crest สูงมาก = มี transient/จุดพุ่ง (เช่น click/pop) หรือ noise spike
**แปลผล:** crest สูงผิดปกติร่วมกับ hum/echo อาจบ่งบอก artifact

---

### `lufs_i`

**คือ:** Integrated loudness (LUFS) แบบประมาณ
**ดูทำไม:** เป็นตัววัดความดังที่ใกล้ perception มากกว่า RMS
**แปลผล (คร่าว ๆ):**

* ต่ำมาก (เช่น < -40 LUFS) = ไฟล์เงียบ/เบา
* สูงมาก (เช่น > -12 LUFS) = อาจดัง/อัดแน่น/AGCแรง

> ถ้า `lufs_i = NaN` มักเกิดจากไม่มี `scipy` หรือไฟล์สั้นมาก

---

### `clipping_pct_max` (+ `clipping_pct_chX`)

**คือ:** % samples ที่อยู่ใกล้ full scale (>= clip_threshold)
**ดูทำไม:** คลิป = “ข้อมูลเสียงหาย” แก้ยากกว่า noise ทั่วไป
**แปลผล:** ถ้าเกิน ~0.1% (แล้วแต่ระบบ) ควร flag ตรวจ

---

# C) Speech / Silence / VAD (Energy-based)

สคริปต์ใช้ VAD แบบ energy-based:

1. คำนวณพลังงานเป็นเฟรม (frame_ms=30, hop_ms=10)
2. ประมาณ noise floor ด้วย percentile ต่ำ (10th)
3. ถือว่าเป็น speech ถ้า `frame_db >= noise_floor_db + vad_margin_db`
4. ทำ smoothing ให้ผ่าน min_speech/min_silence

> ข้อดี: เร็ว, ไม่ต้องใช้โมเดล
> ข้อจำกัด: ถ้า noise สูงมาก/เพลง/TV อาจเข้าใจผิดว่า “speech”

---

### `speech_ratio_any`

**คือ:** สัดส่วนเวลาที่ “มีเสียงพูด” (รวมทุก channel)
**ดูทำไม:** คัดไฟล์ที่แทบไม่มีบทสนทนา (dead call, hold music, silence)
**แปลผล:** ต่ำมาก (เช่น < 0.15) มักเป็นไฟล์ไม่คุ้มถอด

### `num_speech_segments_any`

**คือ:** จำนวนช่วงพูด (หลัง smoothing)
**ดูทำไม:** ถ้า segments เยอะผิดปกติ = VAD แตก (noise สลับ) หรือเสียงกระตุก

### `avg_speech_segment_s_any`

**คือ:** ความยาวเฉลี่ยของช่วงพูด
**ดูทำไม:** สะท้อน turn-taking หรือปัญหา chunking/VAD

### `max_silence_s_ch0`, `initial_silence_s_ch0`

**ดูทำไม:**

* initial silence สูงมาก = ช่วงต้นสายเงียบ/agent ยังไม่รับ/ไฟล์นำหน้ามากไป
* max silence สูงมาก = dead air / call hold / mic mute

### per-channel (เช่น `speech_ratio_ch0`, `num_speech_segments_ch0`)

**ดูทำไม:** ถ้า stereo แยกคนพูด จะเห็นเลยว่าใครพูดน้อย/ไมค์ดับ/ช่องเงียบ

---

# D) Noise / Dropouts / Artifacts

### `noise_floor_dbfs_mean` (+ `noise_floor_dbfs_chX`)

**คือ:** ระดับพลังงานตอน “เงียบ” (ประมาณจาก 10th percentile ของ frame dB)
**ดูทำไม:** noise floor สูง = noise เยอะ
**แปลผล:** ถ้าสูงกว่าปกติใน dataset มาก ๆ → คุณภาพไม่ดี/ไมค์แย่/เสียงฮัม

---

### `est_snr_db_best` (+ `est_snr_db_chX`)

**คือ:** SNR ประมาณจากเฟรมพูด vs เฟรมไม่พูด (median speech dB - median noise dB)
**ดูทำไม:** SNR ต่ำสัมพันธ์กับ WER สูงใน ASR
**แปลผล (คร่าว ๆ):**

* < 10 dB = น่าห่วง
* 10–20 dB = พอใช้
* > 20 dB = ค่อนข้างดี

> หมายเหตุ: ถ้า VAD แยก speech/noise ผิด SNR จะเพี้ยนได้

---

### `zero_pct_max`

**คือ:** % samples ใกล้ 0 มาก ๆ (near_zero_threshold)
**ดูทำไม:** ตรวจ “เงียบผิดธรรมชาติ” หรือกรณีไฟล์มีช่วงเป็นศูนย์

### `longest_zero_run_ms_max` (+ per channel)

**คือ:** ช่วง near-zero ที่ยาวที่สุด (ms)
**ดูทำไม:** ตรวจ dropouts / เสียงขาด / packet loss แบบเป็นช่วงยาว
**แปลผล:** ถ้าเกิน ~500ms (จูนได้) ควร flag

---

### `speech_dropout_ratio_proxy`

**คือ:** สัดส่วน “เฟรมที่เป็น speech” แต่พลังงานใกล้ noise floor (proxy)
**ดูทำไม:** บางครั้ง VAD บอกพูด แต่จริง ๆ เสียงพูดขาด/อู้มาก
**แปลผล:** ยิ่งสูงยิ่งน่าสงสัยว่าเสียงพูดไม่ชัดหรือ VAD จับผิด

---

# E) Stereo / Overlap / Crosstalk (ถ้ามี 2 channels)

### `overlap_ratio`

**คือ:** % เวลาในเฟรมที่ทั้ง ch0 และ ch1 เป็น “speech” พร้อมกัน (double-talk proxy)
**ดูทำไม:** overlap สูงทำให้ diarization และการวิเคราะห์บทสนทนายากขึ้น
**แปลผล:** > 0.2 ถือว่าสูง (แล้วแต่ domain)

---

### `channel_corr_01`

**คือ:** Pearson correlation ระหว่าง ch0/ch1 ในโดเมน sample
**ดูทำไม:** ถ้า correlation สูงมาก อาจแปลว่า

* ไม่ได้แยกคนจริง (เป็น stereo duplicate)
* หรือมี leakage/crosstalk สูง
  **แปลผลคร่าว ๆ:**
* ใกล้ 1 = สองช่องเหมือนกันมาก
* ใกล้ 0 = แยกกันดี/ไม่สัมพันธ์
* ใกล้ -1 = สวนเฟส (เจอน้อย)

---

### `crosstalk_db_01`

**คือ:** proxy “เสียงอีกฝั่งรั่ว” โดยดูพลังงาน ch0 ตอน ch1 พูด แต่ ch0 ไม่พูด เทียบกับ silence
**ดูทำไม:** ประเมินว่าแชนแนลแยกคนดีจริงไหม
**แปลผล:** ค่าสูง = leakage สูง → diarization/turn-taking จะสับสน

> ค่าเป็น proxy ใช้เปรียบเทียบใน dataset เดียวกันดีที่สุด

---

# F) Spectral / Hum / Echo

> ฟีเจอร์หมวดนี้ช่วย “บอกชนิด noise” และ “signature” ของสภาพแวดล้อม

### `spectral_centroid_hz_speech` / `_noise`

**คือ:** ค่าเฉลี่ยตำแหน่งความถี่ (center of mass) ของพลังงาน
**ดูทำไม:**

* centroid สูงขึ้นบ่อย ๆ = hiss/เสียงแหลม
* centroid ต่ำมาก = rumble/เสียงอู้/low freq หนัก

### `spectral_flatness_speech` / `_noise`

**คือ:** บอก “ความ noise-like” (flatness สูง = คล้าย white noise)
**ดูทำไม:** แยกเสียงพูดที่ชัด vs เสียงแบบ hiss/noise

### `spectral_rolloff95_hz_speech` / `_noise`

**คือ:** ความถี่ที่สะสมพลังงานถึง 95%
**ดูทำไม:** งานโทรศัพท์ 8kHz จะ rolloff ต่ำกว่า 16kHz ชัดเจน

---

### `hum50_ratio` / `hum60_ratio`

**คือ:** สัดส่วนพลังงานรอบ 50Hz/60Hz และ harmonic เทียบกับพลังงานรวมใต้ 300Hz
**ดูทำไม:** ตรวจไฟบ้านฮัม / ground loop / line noise
**แปลผล:**

* ค่าสูงผิดปกติ = มี hum ชัด (ควรทำ notch หรือ denoise)

---

### `echo_proxy_corr`

**คือ:** proxy ของ echo โดยดู autocorrelation ของ envelope (20–200ms lag) ในช่วงต้นไฟล์
**ดูทำไม:** ถ้า agent เปิด speaker/มี feedback จะมี “pattern” ซ้ำ ๆ
**แปลผล:**

* ค่าสูงผิดปกติ = น่าสงสัย echo/repeat

> เป็น proxy: ไม่ใช่ตัววัด RT60 หรือ ERLE จริง

---

## Flags คืออะไร? (การคัดไฟล์อัตโนมัติ)

คอลัมน์ `flags` เป็นการรวม “ธงเตือน” หลายแบบ เช่น:

* `low_speech_ratio` : speech_ratio_any ต่ำกว่า `flag_min_speech_ratio`
* `low_snr` : est_snr_db_best ต่ำกว่า `flag_min_snr_db`
* `clipping` : clipping_pct_max สูงกว่า `flag_max_clipping_pct`
* `dropouts_or_dead_samples` : longest_zero_run_ms_max สูงเกิน
* `high_overlap_double_talk` : overlap_ratio สูงเกิน
* `too_quiet_lufs` / `too_loud_lufs` : LUFS หลุดช่วง

> แนะนำ: ใช้ flags เป็น “รายการตรวจ” ไม่ใช่ตัดทิ้งแบบทันที แล้วค่อยปรับ threshold ให้เข้ากับ dataset ของคุณ

---

## วิธีจูน threshold ให้เข้ากับ data จริง (แนะนำ workflow)

1. รันกับไฟล์ 100–500 ไฟล์ก่อน
2. ดู distribution ของคอลัมน์หลัก:

   * `lufs_i`
   * `est_snr_db_best`
   * `speech_ratio_any`
   * `clipping_pct_max`
   * `longest_zero_run_ms_max`
3. เลือก threshold แบบ percentile (เช่น ตัด bottom 5% SNR, top 1% clipping)
4. ฟังตัวอย่างที่ถูก flag เพื่อ calibrate ว่า “ควรตัด/ควรแก้/ควรปล่อย” แค่ไหน

---

## ข้อจำกัดที่ควรรู้

* VAD เป็น energy-based → ถ้ามีเพลง/TV/noise ต่อเนื่อง อาจคิดว่าเป็น speech
* LUFS เป็น approximate → ใช้เทียบ/หา outlier ได้ดี แต่ไม่ใช่มาตรฐานเป๊ะ 100%
* Echo proxy เป็น heuristic → ใช้ flag เบื้องต้น ไม่ใช่การวัด echo cancellation metrics

---

## แนะนำต่อยอด (ถ้าจะจริงจังกับ Call Center)

* เปลี่ยน VAD เป็น **Silero VAD / WebRTC VAD / TEN VAD** เพื่อแยก speech แม่นขึ้น
* ถ้า stereo แยกคน: ทำ “channel role detection” (agent/caller) อัตโนมัติ
* ทำ dashboard: histogram + scatter (SNR vs LUFS, speech_ratio vs duration) เพื่อเห็น outliers ชัดมาก
