import json, os, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
job = json.load(open("tts/job.json"))
m = ChatterboxTTS.from_pretrained(device="cpu")
os.makedirs("tts_out", exist_ok=True)
REF = {"S": "tts/ref.wav", "T": "tts/ref_f.wav"}
for j in job["jobs"]:
    gap = torch.zeros(1, int(m.sr * j.get("gap", 0.25)))
    parts, times, t = [], [], 0.0
    for spk, s in j["lines"]:
        w = m.generate(s, audio_prompt_path=REF[spk], exaggeration=j.get("exaggeration", 0.6), cfg_weight=j.get("cfg", 0.45))
        d = w.shape[1] / m.sr
        times.append([round(t, 3), round(t + d, 3), spk, s]); t += d + gap.shape[1] / m.sr
        parts += [w, gap]
    ta.save(f"tts_out/{j['name']}.wav", torch.cat(parts, 1), m.sr)
    json.dump(times, open(f"tts_out/{j['name']}.json", "w"), indent=1)
