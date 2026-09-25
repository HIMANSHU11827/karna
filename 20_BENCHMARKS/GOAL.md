# Karna 24/7 Goal — run forever, learn forever, never stop

## Goal
One brain process that runs 24/7: senses all kinds (text/image/audio/video),
learns live, grows own regions, sleeps to consolidate, never forgets, never stops.

## Loop (forever)
1. SENSE — read next input (any kind) via UniEncoder.
2. ROUTE — GrowBrain surprise-bid picks winner, conf gates cell+readout.
3. ACT — cell+PSM output, episodic 0.95 gate.
4. LEARN — mistake-only perceptron + importance gate; drive scales plasticity.
5. GROW — hot splits, idle merges, self-builds regions/subs by need.
6. SLEEP every 500 steps — replay 50 episodic traces + cell consolidate.
7. CHECKPOINT every 1000 steps — save .psm + .norm + heartbeat log.
8. REPEAT — no exit condition. Crash → restart from checkpoint.

## Success bars
- Process alive 24/7 (heartbeat fresh < 60s).
- Regions grow with novelty, never exceed rmax.
- MNIST 0.844 holds; Split mean 0.96 holds; QA4 4/4 holds.
- No NaN, no OOM, checkpoint always loadable.

## Files
- Runner: `20_BENCHMARKS/karna_daemon.py` (loop + heartbeat + checkpoint).
- Brain: `20_BENCHMARKS/karna_brain.py` (UniBrain canonical).
- State: `/tmp/karna_24_7/` (psm, norm, heartbeat.json, log).
