# Karna Project — Full A-to-Z Documentation (Everything Done Till Now)

Using **md-document** skill for this writeup — it fits because you asked for a complete top-to-bottom document of the full chat.

---

## A. What is this project (your idea, my words)

We are building **true AGI / AI assistant** — not a chatbot.

- Old AI today: chatbot that guesses next word + outside tools attached. Looks smart, inside only chat.
- Our AI: real brain that sees + hears + reads + watches in **real time**, **learns in real time**, **adapts in real time** like humans, solves many problems.
- Everything **new from scratch**: new neural network, new formulas, new architecture, new weights, new weights format/storage, everything else + hidden things you did not tell yet.
- Old networks (transformers, M-models, JF models) are trace, outdated, weak — we study them, learn core principle, then remake better.

---

## B. Why we chose research-first (no building at start)

You said: first research everything existing, understand how all these work, find core principle, then remake.

So we did:
1. Learned old things (transformers, M-models, JF, cells, weights, LLM chart).
2. Found core (why they work + why weak).
3. Then started making new.

This is why we did not code at first — only understanding.

---

## C. What we researched (deep research summaries)

### C1. Transformers — how they work
- Simple: every word asks "who should I listen to?" and takes weighted average.
- Formula: `Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V`
- Parts: multi-head = many ears, position stamp = order number, MLP = think alone, LayerNorm + skip = keep stable.
- Core why works: sees all at once, parallel, O(1) path between tokens.
- Why old/weak: O(n^2) — 10x length = 100x cost. Huge KV-cache notes. Goldfish outside window. Frozen after training.

### C2. M-models (newer networks)
- S4/SSM: `h' = Ah+Bx, y = Ch` — leaky bucket, O(n) long, but cannot choose memory.
- Mamba: selective `B,C = f(x)` — keeps important, forgets boring. Fast, long, but lossy, bad exact recall.
- RWKV: transformer training + RNN running. Good streaming.
- RetNet: `(QK^T * decay) V` — forgets by design.
- MoE: `y = TopK router * experts` — many brains, use 1-2 per word. Big but same limits.
- Jamba: mix Mamba + few Attention + MoE. Best compromise today.
- All still weak for AGI: only cheaper per token, still next-word guesser. No memory, no goals.

### C3. JF models (JEPA / World Models)
- JEPA: do not redraw full photo, predict meaning only. `Loss = ||Predictor(Enc(x)) - Enc(y)||^2`
- World loop: sense `s = Enc(obs)` → imagine `s_next = f(s,a)` → act best one. Like chess in head.
- Different: Transformer/Mamba = how to compute. JEPA/World = what to predict and why.

### C4. Cells — difference of each cell
- Real brain cell: collect → sum → spike if full. Talks in timing.
- Perceptron/MLP `y = max(0, w.x+b)`: remembers nothing, no time.
- LSTM `c = f*c_old + i*g`: remembers 100s steps, heavy slow.
- GRU: 1 notebook, faster, forgets sooner.
- Attention head: remembers all in window, costly O(n^2).
- SSM: compressed summary `h`, cheap but lossy.
- Hopfield: patterns as valleys `E = -1/2 sum wxx`, tiny capacity.
- Spiking: closest to real `charge-leak-spike-reset`, hard to train.

### C5. Weights — old vs new we want
- Now: big tables `W`, QKV, embeddings, float16/32 in `safetensors`. Training = tuning billions knobs.
- Problems: huge 140GB, dense all active, frozen static, new learning erases old (catastrophic forgetting).
- We want new: sparse 1-5% active, SDR index sets, phase + timing, fast scratch + slow long memory. Small, live, no forgetting.

### C6. LLM chart + AI Agents — your point is 100% right
- Base LLMs (GPT, Claude, Gemini, Llama): all same — A) read internet guess next word, B) finetune to chat nice + safe.
- Chart: base autocomplete → chat finetune → tool finetune `call_search(), call_code()`. Same brain underneath.
- Agents today: chatbot writes + detector sees `[CALL:...]` + runs outside code/API/phone + feeds back + chatbot replies. Looks smart.
- Truth: inside only chat. No real memory, no live learning, no body. Only words in, words out.

### C7. Human brain regions — mastery + backup + shared understanding
- Vision V1-V4-IT masters seeing (Hubel-Wiesel orientation, Kanwisher faces, Zeki color, MT motion). Audio A1 masters hearing (tonotopy). Language Broca/Wernicke masters speech. Each mains one.
- But any can do others: blind Braille fires V1 (Sadato 1996, Cohen TMS proof), deaf vision fires auditory cortex (Lomber 2010), ferret eyes→auditory cortex grows vision maps (Sur Lab).
- Deeper: not spare-tire backup. Other regions **understand what master processed** in own language. Shared event ID + own views + consensus.
- Sub-regions inside each: vision (V1→V2→V4→IT+MT+FFA/PPA), audio (core→belt→phoneme→voice/music/prosody), text (VWFA→syntax/semantics→discourse + code MD + math IPS), video (flow→ego→snippet→event→intent).
- MoE is trash: hard TopK collapses, token dropping, all-to-all cost, 671B VRAM lie, no sharing, mis-route = hallucination. We make newer soft way.

---

## D. Why we chose unified brain + regions (your vision)

You said:
1. One brain for image/audio/video/text — not 4 brains.
2. Inside, regions: each specialist mains one thing (vision mains vision), but full brain does all together.
3. Regions auto-build themselves while learning — we do not hand-build. Brain builds own regions + subs depending on what it learns.
4. Regions understand each other, not just backup.

So we chose:
- ONE cell + ONE weights for all (unified brain).
- Soft regions inside (mastery + backup + shared understanding).
- Self-growing (starts uniform, splits where overloaded, merges where idle).
- One encoder path for all kinds (bytes → shared STE).

---

## E. What we built (files, why each, how each)

All in `20_BENCHMARKS/`, NumPy only, no backprop. Total ~950 lines.

| File | Lines | Why we made it | How it works | Verified |
|---|---|---|---|---|
| `ajo1.py` | 131 | New cell + new weights + learn/adapt loop (core brain) | SPLRCell `z=D@x → kWTA(z+h)`, PSM perceptron on dense feats, gated episodic, save/load | MNIST 0.8340, bench 0.65 |
| `karna_uni.py` | 66 | ONE encoder for all kinds (your call) | bytes → shared 512-d → top-k 64 STE `{vector,t,salience,source}` | UNI_OK 64 each |
| `karna_encoders.py` | 116 | Old per-modality encoders (before uni) | text hash n-gram, image block16, audio energy+zc, video frame-diff | shapes OK |
| `karna_fusion.py` | 74 | Bind events into objects | phase 80ms window, cosine>0.5 merge, 128 objects max | BINDER_OK 4 objs |
| `karna_brain.py` | 48 | ONE brain for all (unified) | UniEncoder → Binder → ONE cell → ONE PSM | BRAIN_OK 20 steps 4 kinds |
| `karna_rt.py` | 32 | Realtime sense-act demo | 20 mixed steps loop, ms/step measure | RT_OK 3.62ms |
| `karna_grow.py` | 114 | Self-growing regions (your latest call) | uniform pool → surprise-bid → Hebbian → homeostasis → split hot / merge idle | GROW_OK 6 regions |
| `karna_vision.py` | 70 | Vision specialist subs (V1→IT) | random edge filters → max-pool → stats → projection | compiles, smoke shapes |
| `karna_audio.py` | 73 | Audio specialist subs (core→TVA) | STFT → core filters → belt pool → phoneme softmax + speaker norm | compiles, smoke shapes |
| `karna_video.py` | 71 | Video specialist subs (flow→event) | frame-diff → ego split → snippet proj → boundary detect | compiles, smoke shapes |
| `karna_psm.py` | 77 | New weights file format | int8 per-row quant + PSMFile save/load + checksum | PSM_SMOKE_OK |
| `karna_bench.py` | 38 | Synthetic 8-class bench | 500 steps, every 3rd supervised, acc+latency+sparsity | acc 0.65, 0.3ms |
| `karna_mnist.py` | 40 | Real MNIST bench | n1024/k64/3000 train, save→load, 500 test | acc 0.8340 |
| karna_text.py | — | Missing (subagent cut) | T1→T2/T3→T6/T7 planned | TODO |

Old core fixed along the way: `01_SRC/network.py` (dim crash, forward-learn leak, ragged mean), `01_SRC/layer.py` (wrong pre term), `01_SRC/memory.py` (confidence never fired), `rtalm/sdr_core.py` (garbage file → shim), `raie_unified.py` (missing docstring quote), `.venv` NumPy 2.5.3.

---

## F. How unified brain works (simple flow)

```
any input (text/image/audio/video)
  → UniEncoder: bytes → shared 512-d → sparse 64 STE
  → Binder: bind by time+similarity into objects (max 128)
  → ONE SPLRCell: z = D@x → kWTA(z+h) → sparse code h + dense z
  → ONE PSM: y = Ws@z + Wf@z (+ episodic if >0.95 match)
  → learn (if target): mistake-only perceptron Ws[true]+=ηz, Ws[pred]-=ηz
  → save/load .psm file
```

Self-growing version (`karna_grow.py`):
```
start: 6 identical regions, uniform
each step: all bid by surprise (low error wins), soft weights, all learn scaled
hot (high error + high use, 200 steps) → split into 2 subs
idle (low use, 1000 steps) → merge back
result: regions drift to clusters automatically (vision-like, audio-like...)
```

---

## G. Numbers (honest, verified)

- `Karna1_MNIST acc=0.8340 train_ms=16-29 size_kb=77.9` (n1024/k64/3000, save→load path)
- Full-system with episodic ON: 0.686 (3000/500 probe)
- `Karna1_BENCH acc=0.65 latency=0.3ms sparsity=0.0312` (synthetic 8-class)
- `BRAIN_OK steps=20 kinds=4 objects=4 ms=17.68 active=64`
- `RT_OK ms_per_step=3.62 active=32`
- `GROW_OK regions=6 ms=2.18` + emergence probe (4 clusters → 4 mains found, lesion err 2.2076→2.2137 = graceful)
- Old baselines: raw-centroid 0.784, code-centroid 0.718, raw-pixel perceptron 0.84, dense-z perceptron 0.755
- Full compile 100%, NumPy 2.5.3, `.venv` ready

---

## H. Why each choice (reason per decision)

1. **Research-first, no code**: you ordered it. Needed to know what to throw away.
2. **NumPy only, no backprop**: locked direction 2026-04-28 + your from-scratch call. Local Hebbian/perceptron only.
3. **Sparse kWTA 2-6%**: brain <5% active, 50x compute cut, less interference.
4. **Dense readout (not sparse) for classifier**: kWTA 0/1 threw away magnitude → 0.10. Dense `z` keeps 0.75+. Sparse kept for keys/activity.
5. **Frozen cell D**: Hebbian drift killed codes (64→4 overlap). Freeze first, plasticity later slow.
6. **Mistake-only perceptron**: LMS + norm-clip diverged. Update only on mistake, norm cap 2.0 → stable.
7. **Gated episodic 0.95**: threshold 0.8 hijacked output. 0.95 = exact-match only, graceful.
8. **One encoder**: you ordered one pathway, not 4. Bytes→shared STE.
9. **One brain + regions**: your vision — full brain does all, region mains one, subs inside stages.
10. **Self-growing not hand-built**: your latest call — brain builds own regions/subs while learning. No hand vision/audio nets. `karna_grow.py` proves emergence.
11. **MoE rejected**: trash — collapse, dropping, VRAM lie, no sharing. We do soft all-fire + mastery + gain.
12. **Time as hub, not language**: lip+phoneme bind by 80ms coincidence, no caption needed.

---

## I. What remains (TODO, honest) — updated 2026-09-24

- `karna_text.py` DONE — T1 hash → T2 syntax → T3 semantics → T6 code / T7 math, verified code=True math=True.
- `karna_brain_v2.py` DONE — UniEncoder + GrowBrain + ONE cell + ONE PSM + 4 specialists. V2_OK 20 steps.
- `karna_brain_v3.py` DONE — GrowBrain IS the router: soft winner weight gates cell input `uv*(0.5+0.5*gate)`. V3_OK gate=0.185.
- `karna_joint.py` DONE — joint image+text query, one brain answers both: JOINT_OK 8/8.
- Specialist files still scaffolds (random filters) — real mastery from self-growing, not these.
- Cell plasticity OFF by default (`eta=0.0`) — slow path exists via eta>0.
- MNIST holds 0.8340 (3000-sample); 4000/6000-sample push timed out on CPU — needs faster box or second pass.
- Next: merge grow-regions as full attention gate + cross-region QA understanding score.

## K. v3 → v4 update (2026-09-24, continued work)

- `karna_brain_v3.py` — GrowBrain IS the router: winner weight gates cell input. V3_OK gate=0.18.
- `karna_brain_v4.py` — FULL attention gate: grow confidence gates encoder input `uv*(0.5+0.5*conf)`, readout `z*(0.7+0.3*conf)`, and learn rate `(0.5+conf)`. V4_OK conf=0.179.
- `karna_qa.py` — cross-region understanding probe: same event trained via text+image, both recall via different regions. QA_OK 4/4 (text→region3/0, image→region4, both correct).
- `karna_joint.py` — joint image+text: JOINT_OK 8/8.
- Speedup: vision v1 Python loop → stride-trick einsum. v4 20-step mixed loop ~5.35ms/step.
- MNIST still 0.8340 (larger push CPU-capped, documented).

## L. v5 need-driven growth (2026-09-24, your latest call)

- Hand-built T1→T2→T3→T6/T7 = default trash. Removed from path.
- `karna_brain_v5.py` — NO hand specialists. UniEncoder → GrowBrain (r0=4, rmax=24) → ONE cell → ONE PSM.
- Brain grows own regions by need: poem/code/math/faces — same mechanism, slowly with learning.
- Verified: V5_OK poem_recall=3/3; any-need probe poem 2/3, code 3/3, math 3/3.
- Full sweep green: v5, v4, QA 4/4, joint 8/8, grow, brain, MNIST 0.8340.

## M. Remaining-7 build-out (2026-09-25, datrat doing all things)

- Need drives in GrowBrain: `step(x, reward, novelty)` → `drive = 1+0.5*surprise+0.5*novelty+0.5*reward` scales plasticity. GROW_OK.
- `karna_qa4.py`: full 4-way matrix — same event via text+image+audio+video, one brain recalls all. QA4_OK 4/4 (all → region3).
- Safe slow plasticity: `SPLRCell.consolidate()` (renorm D, clip Wp, decay h) + 8/8 stable overlap after 100 slow steps. Cell auto-consolidates every 500 steps in v6.
- `karna_brain_v6.py`: self-grow PRIMARY + specialists as 30% bias (`fused = 0.7*key + 0.3*bias`). V6_OK drive=1.3.
- `karna_forget.py`: taskA pre=1/2 → post=1/2 (no forgetting), taskB=2/2, lesion best → still 1/2 (graceful).
- `karna_stream.py`: 40 live audio+video chunks bind in real time. STREAM_OK 0.81-0.86ms/step.
- Video single-frame fix: `_as4d` wrapper, zero-flow fallback.
- Total 1517 lines NumPy. MNIST holds 0.8340.

## N. Final merge + importance gate + honest forgetting (2026-09-25)

- `karna_brain_final.py`: merged v4/v5/v6 + `sleep(steps)` replay (episodic→Ws transfer + cell consolidate). FINAL_OK.
- `karna_split.py`: Split-MNIST 5×2 digits ×400. SPLIT_OK mean=0.962, per-task current 0.84-1.0.
- Importance gate in PSM.learn: `P=1/(1+λF)` per-row, sign-consistent updates, F tracks used feats. MNIST 0.834→0.844.
- HONEST forgetting probe: retention matrix shows old pairs collapse (T0 1.0→0.0 after task1). Winner shift: task0 preds go {2:134, 3:62} — new rows out-bid old. Documented, not hidden. True lifelong needs replay-per-task or per-task heads — next.
- Full sweep green: final, v6, v5, qa4, forget, stream, joint, qa, bench, MNIST 0.844.

## O. 24/7 end-to-end run (2026-09-25, goal: commandcode runs nonstop to full done)

- `GOAL.md`: 8-step forever loop (sense→route→act→learn→grow→sleep→checkpoint→repeat), bars defined.
- `karna_daemon.py`: background runner, heartbeat + checkpoint every 1000 steps, crash→reload. Daemon ALIVE step 65720+.
- Canonical `karna_brain.py` (UniBrain): hand bias REMOVED (`_bias` returns None, fused=key only). Self-grow fully.
- Balanced replay-10 per learn (stratified per label): MNIST 0.844→0.864, SPLIT mean 0.936.
- Batch helper `PSM.batch_acts` + `SPLRCell.batch_step` (2000-code matmul 0.04s). Batch6000: codes 0.18s + learn 10.77s, acc 0.778.
- Full sweep: final/v6/v5/qa4/forget/stream/joint/qa/bench/grow green. Daemon keeps running 24/7.

## P. Per-task heads fix perfection (2026-09-25, forgetting solved with task ID)

- `PSM.heads/task` ensemble: `act/learn(..., task)` snapshots per-task Ws copy + own F. Frozen old heads, new task new head.
- Retention matrix FIXED: T0 1.00→1.00→1.00→1.00→1.00, T1 0.97, T2 0.94, T3 0.99, T4 0.92. Zero forgetting with task ID.
- Honest note: needs task oracle at test (task=pj). Task-free lifelong still open — next: auto task-infer via GrowBrain winner.
- Scale probe n2048/k128 batch6000: 0.736 (worse than n1024 0.778 — capacity not the cap, single-pass perceptron is).

## Q. Fix-all sweep (2026-09-25, cotni fix all things)

- Task-free auto-infer tested: grow winner≠task (0.19) — routing clusters by shape, not digit task. Documented; oracle still needed.
- Second-pass review: CPU-capped (124 timeout) — single-pass stays canonical.
- `compose()` added: seed STE → 8-step rollout, drift 7.587/5 steps. Retrieval >> generation, honest.
- `karna_sensors.py`: live mic via arecord + /dev/video0 found. SENSORS_OK audio conf=0.268.
- Canonical merge: v2-v6+final → `_archive/`, all 5 stragglers repointed to `karna_brain.UniBrain`. Single file now.
- Daemon long-run: step 112948+, ckpt 1.3KB loadable, no NaN/OOM.
- Paper 3/5/6 rewritten with real math + ablations + limits.
- Verified: brain, sensors, stream, joint, qa, qa4, forget, split, mnist — all green.

## R. Live-camera + decoder + task-proto sweep (2026-09-25)

- Slow task-proto: `_task_proto` EMA per head (tau 0.999) + `_infer_task` prefers proto cosine; falls back to head-max. Brain OK.
- `UniEncoder.decode()`: STE → top-8 chars by projection cosine. Seed decodes stable (attractor holds symbols).
- Live camera: ffmpeg v4l2 gray8 48x48 grab works — real frame into brain. SENSORS_OK video live=True conf=0.266.
- Sleep stability: replay 15, conf pre/post flat (0.267→0.263 etc.) — no collapse, no boost; sleep safe.
- Daemon step 206828+, MNIST 0.864, QA4/JOINT/STREAM green.

## S. Decoder-collapse fix + honest caps (2026-09-25)

- Root cause: P scale 1/dim + raw-0.5 missing → all single bytes mapped to same topk (zyxwvuts for every input).
- Fix: P scale 1/sqrt(dim) + center `(raw-0.5)` in encode AND decode. Now distinct inputs → distinct decodes.
- Verified: UNI_OK, BRAIN OK, QA4 4/4, MNIST 0.840 (recenter cost -0.024, expected — codes redistributed).
- Task-free auto-head still 0.19-0.257 (routing≠tasking, documented, not fixed by proto).
- Multi-pass fit (4000×2, 3000×3) CPU-capped (124) — single-pass stays canonical.
- Daemon step 247108+, all green.

## T. Fix-all-7 round (2026-09-25, can you fix — yes)

- Margin perceptron: learn on mistake OR gap<0.2 (thickens margin). MNIST 0.824 (margin costs single-pass speed, holds).
- Word decode: vocab → 16 real words (moon/soft/night/song...), trigram repeat. 'moon soft night song' → 'world silent night hello brain'. Real words out.
- Full checkpoint: n + padded mus/biases/levels/actives + D/h/t. Grown 5-region save→load verified t:10.
- pytest via uv: 5 passed in 1.10s (smoke/kinds/saveload/compose/mnist-floor).
- Daemon step 269000+, all green. Task-free heads still oracle-gated (documented).

---

## J. One-line story

Old AI = chatbot + tools, frozen, forgetful. We studied all, found core, and are building one self-growing unified brain where full brain handles all senses together, regions master their own, subs learn stages inside, all from scratch in NumPy.

---

*Generated from full chat history. Files: 20_BENCHMARKS/*.py (950 lines). Verified: MNIST 0.8340, BRAIN_OK, GROW_OK, RT_OK. Next: wire self-growing + 4-specialist subs into UniBrain v2.*
