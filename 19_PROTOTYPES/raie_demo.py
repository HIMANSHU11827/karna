#!/usr/bin/env python3
"""
RAIE Demo: Real-Time Adaptive AGI Assistant
=============================================

Shows all features working:
1. Real-time text processing with spelling correction
2. Image + Audio + Text multimodal fusion
3. Continuous learning from feedback
4. Performance monitoring
5. Real-time adaptation
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/himanshu/Desktop/AGI_RESEARCH_LAB/19_PROTOTYPES')
from raie_engine import RAIE, Modality


def main():
    print("=" * 70)
    print("RAIE: Real-Time Adaptive AGI Assistant")
    print("=" * 70)
    
    engine = RAIE(embedding_dim=64)
    
    # Pre-train spelling corrections
    print("\n[Setup] Learning spelling corrections...")
    corrections = {
        "whas": "what", "weater": "weather", "luv": "love",
        "teh": "the", "recieve": "receive", "seperate": "separate",
        "definately": "definitely", "occured": "occurred",
        "wierd": "weird", "thier": "their", "alot": "a lot",
    }
    for wrong, right in corrections.items():
        engine.text_encoder.learn_correction(wrong, right)
        engine.text_encoder.learn_correction(wrong.capitalize(), right.capitalize())
    
    print(f"  Loaded {len(corrections)} spelling corrections")
    
    # === Demo 1: Fuzzy spelling correction ===
    print("\n" + "─" * 70)
    print("Demo 1: Fuzzy Spelling Correction")
    print("─" * 70)
    
    test_inputs = [
        "Whas is the weater like?",
        "I luv machine learning!",
        "Teh recieved package was wierd",
        "Seperate thier alotted portions",
        "This is a normal sentence.",
    ]
    
    for text in test_inputs:
        corrected = engine.text_encoder.correct_spelling(text)
        response = engine.process_text(text)
        print(f"\n  Input:     '{text}'")
        print(f"  Corrected: '{corrected}'")
        print(f"  Response:  '{response}'")
    
    # === Demo 2: Continuous learning ===
    print("\n" + "─" * 70)
    print("Demo 2: Continuous Learning from Feedback")
    print("─" * 70)
    
    # Learn some responses
    print("\n  Teaching responses...")
    engine.text_encoder.vocab["hello"] = 10
    engine.text_encoder.vocab["hi"] = 10
    engine.text_encoder.vocab["hey"] = 10
    engine.text_encoder.vocab["greetings"] = 10
    
    # Positive feedback
    engine.learn_from_interaction("hello", True)
    engine.learn_from_interaction("hi there", True)
    engine.learn_from_interaction("hey!", True)
    
    # Negative feedback (teaching correct response)
    engine.learn_from_interaction("whas up", False, "what's up")
    
    print(f"  Learned from 4 interactions")
    print(f"  Loss trend: {engine.learner.get_loss_trend()}")
    print(f"  Update count: {engine.learner.update_count}")
    
    # === Demo 3: Multimodal processing ===
    print("\n" + "─" * 70)
    print("Demo 3: Multimodal Processing")
    print("─" * 70)
    
    # Text + Image
    print("\n  Text + Image:")
    text_emb = engine.text_encoder.encode("I see a beautiful landscape")
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    image_emb = engine.image_encoder.encode(image)
    fused = engine.fusion.fuse({Modality.TEXT: text_emb, Modality.IMAGE: image_emb})
    response = engine.response_gen.generate(fused)
    print(f"    Text: 'I see a beautiful landscape'")
    print(f"    Image: 32x32 RGB")
    print(f"    Response: '{response}'")
    
    # Audio only
    print("\n  Audio:")
    audio = np.random.randn(1600).astype(np.float32)
    audio_emb = engine.audio_encoder.encode(audio)
    fused = engine.fusion.fuse({Modality.AUDIO: audio_emb})
    response = engine.response_gen.generate(fused)
    print(f"    Audio: 100ms @ 16kHz")
    print(f"    Response: '{response}'")
    
    # === Demo 4: Performance under load ===
    print("\n" + "─" * 70)
    print("Demo 4: Performance Under Load")
    print("─" * 70)
    
    n = 5000
    print(f"\n  Processing {n} text requests...")
    start = time.perf_counter()
    for i in range(n):
        engine.process_text(f"Request {i}: hello world what is AGI?")
    total = time.perf_counter() - start
    
    print(f"  Total time: {total:.2f}s")
    print(f"  Avg latency: {total/n*1000:.3f}ms")
    print(f"  Throughput: {n/total:.0f} req/s")
    
    # === Final report ===
    print("\n" + "─" * 70)
    print("Final Performance Report")
    print("─" * 70)
    
    import json
    report = engine.get_performance_report()
    
    print(f"\n  Total requests: {report['total_requests']}")
    print(f"  Context size: {report['context_size']}")
    print(f"  Fusion weights: {report['fusion_weights']}")
    
    print(f"\n  Latency Stats:")
    for op in ['text', 'image', 'audio', 'learning']:
        stats = report['latency']['stats'].get(op, {})
        if stats:
            print(f"    {op:12s}: p50={stats.get('p50', 0):.3f}ms  p95={stats.get('p95', 0):.3f}ms  p99={stats.get('p99', 0):.3f}ms")
    
    print(f"\n  Learning:")
    print(f"    Update count: {report['learning']['update_count']}")
    print(f"    Loss trend: {report['learning']['loss_trend']}")
    
    print(f"\n  Budget:")
    print(f"    Text total: {report['budget_ms']['text_total']:.0f}ms")
    print(f"    Image total: {report['budget_ms']['image_total']:.0f}ms")
    print(f"    Audio total: {report['budget_ms']['audio_total']:.0f}ms")
    print(f"    Learning step: {report['budget_ms']['learning_step']:.0f}ms")
    
    # All within budget?
    text_p99 = report['latency']['stats'].get('text', {}).get('p99', 0)
    image_p99 = report['latency']['stats'].get('image', {}).get('p99', 0)
    audio_p99 = report['latency']['stats'].get('audio', {}).get('p99', 0)
    
    print(f"\n  Budget Compliance:")
    print(f"    Text  (p99={text_p99:.2f}ms / budget={report['budget_ms']['text_total']:.0f}ms): {'PASS' if text_p99 < report['budget_ms']['text_total'] else 'FAIL'}")
    print(f"    Image (p99={image_p99:.2f}ms / budget={report['budget_ms']['image_total']:.0f}ms): {'PASS' if image_p99 < report['budget_ms']['image_total'] else 'FAIL'}")
    print(f"    Audio (p99={audio_p99:.2f}ms / budget={report['budget_ms']['audio_total']:.0f}ms): {'PASS' if audio_p99 < report['budget_ms']['audio_total'] else 'FAIL'}")
    
    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
