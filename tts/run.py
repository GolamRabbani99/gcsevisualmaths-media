import json, os, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
job = json.load(open("tts/job.json"))
m = ChatterboxTTS.from_pretrained(device="cpu")
os.makedirs("tts_out", exist_ok=True)
gap = torch.zeros(1, int(m.sr * job.get("gap", 0.3)))
parts, times, t = [], [], 0.0
for s in job["sentences"]:
    w = m.generate(s, audio_prompt_path="tts/ref.wav",
                   exaggeration=job.get("exaggeration", 0.5), cfg_weight=job.get("cfg", 0.5))
    d = w.shape[1] / m.sr
    times.append([round(t, 3), round(t + d, 3), s]); t += d + gap.shape[1] / m.sr
    parts += [w, gap]
ta.save(f"tts_out/{job['name']}.wav", torch.cat(parts, 1), m.sr)
json.dump(times, open(f"tts_out/{job['name']}.json", "w"), indent=1)
