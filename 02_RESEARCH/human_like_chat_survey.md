# Human-Like Chat Systems — Technical Survey

## 1. Intent Recognition (NLU)

Intent recognition maps natural language to a structured intent + slots. This is the core of any conversational system.

### Architecture Evolution

**Pre-neural (2010-2015):**
- SVMs / Random Forests with TF-IDF features
- Regex patterns + keyword matching (Siri, Alexa early versions)
- Handcrafted grammars (VoiceXML, SRGS)

**Neural era (2015-2019):**
- LSTM + CRF for slot filling (BIO tagging)
- Joint models: intent classification + slot filling shared encoder
- Attention-based models (BiLSTM + attention)
- Semantic parsing: Seq2Seq to SQL/logic forms (SYNTAXNET, CoSQL)

**Transformer era (2019-2024):**
- BERT/RoBERTa fine-tuning for intent (classification head)
- Joint BERT + CRF for slot filling
- DistilBERT for latency-constrained applications
- Zero-shot / few-shot: SetFit, pattern-exploiting training

**Current state (2024+):**
- LLMs as zero-shot intent recognizers (GPT-4, Claude, Llama)
- In-context learning: provide examples, LLM classifies
- Hybrid: LLM for intent + lightweight classifier for latency
- Semantic parsing: LLM → structured output (JSON mode, function calling)

### Key Components

**Intent Classification:**
- Multi-class classification over intent taxonomy
- Hierarchical intents (top-level → sub-intent)
- Out-of-domain detection (reject intent)
- Confidence calibration (temperature scaling)

**Slot Filling:**
- Sequence labeling: BIO (Begin, Inside, Outside) tags
- Entity types: PERSON, DATE, LOCATION, etc.
- Multi-turn slot carryover
- Implicit slots (inferred from context)

### Techniques That Work Well

1. **Joint modeling**: Single encoder, two heads (intent + slots). Shared representations reduce error propagation.
2. **Data augmentation**: Synonym replacement, back-translation, paraphrasing for low-resource intents.
3. **Active learning**: System asks for labels on uncertain examples.
4. **Contrastive learning**: Similar intents are closer in embedding space.

### Failure Modes
- Overlapping intents (e.g., "book flight" vs "check flight status")
- Domain shift (training distribution != production distribution)
- Code-switching (mixed languages)
- Ambiguous slots ("Paris" — city or person?)

---

## 2. Spelling Correction

### Edit Distance Approaches

**Levenshtein Distance:**
```
d(i,j) = min(
    d(i-1, j) + 1,      # deletion
    d(i, j-1) + 1,      # insertion
    d(i-1, j-1) + cost  # substitution (0 if same, 1 if different)
)
```
- Cost: O(n*m) where n, m are string lengths
- Weighted variants: different costs for keyboard-proximity errors

**Damerau-Levenshtein:** Adds transposition (swapping adjacent chars) as single edit.

**Optimal String Alignment:** Restricts to single transposition per substring.

### Phonetic Algorithms

**Soundex (1918):** Maps names to 4-char code (letter + 3 digits). Groups similar-sounding consonants.

**Metaphone (1990):** Improved English phonetic encoding. Handles more edge cases than Soundex.

**Double Metaphone:** Returns primary + secondary phonetic codes. Handles names of non-English origin.

**Caverphone:** Designed for New Zealand place names (historical matching).

### Noisy Channel Model (Probabilistic Approach)

P(correct|incorrect) ∝ P(incorrect|correct) * P(correct)

- **Error model P(w|c):** Probability of typing w when you meant c
- **Language model P(c):** Prior probability of word c
- Combine: argmax_c P(w|c) * P(c)

Implementation: Generate candidates within edit distance 2, score by noisy channel.

### SymSpell (2019)

Symmetric Delete algorithm — precomputes deletes of dictionary words.
- Lookup speed: O(1) for exact, O(k) for fuzzy where k = number of deletes
- 1 million x faster than BK-tree approach
- Used in Microsoft Office, Google (reportedly)

### Modern Approaches

**Transformer-based:**
- BERT can correct spelling as masked language modeling
- T5/BART for text-to-text correction
- ByT5: byte-level, handles any language without vocabulary

**Approaches for conversational systems:**
- Context-aware correction (use surrounding words)
- Learned error models from user data
- Keyboard-aware (QWERTY proximity for mobile)
- Acoustic model output correction (ASR post-processing)

### Key Insight for AGI

Current spelling correction operates at character/word level. A human-like system would:
- Correct spelling based on *intent* (not just character distance)
- Know when NOT to correct (proper nouns, technical terms, slang)
- Use semantic context: "I went to the park" vs "I went to the pak"

---

## 3. Dialogue Management

Dialogue management controls conversation flow: what to say next given history and state.

### Finite State Machines (FSM)

States → Transitions → Actions

Example for flight booking:
```
GREET → ASK_DESTINATION → ASK_DATE → CONFIRM → BOOK → GOODBYE
```

- Simple, predictable, debuggable
- Brittle: hard to handle off-script behavior
- Scales poorly: states explode combinatorially

### Frame-Based (Slot-Filling)

Dialogue as filling a frame (template):
```
{intent: "book_flight", destination: null, date: null, class: null}
```

Policy: ask for missing slots one by one. More flexible than FSM.

### Information State Update (ISU)

Maintains rich dialogue state including:
- Shared knowledge (what both participants know)
- Private knowledge (what each participant knows)
- Dialogue purpose (what they're trying to achieve)
- Obligations (what they're expected to do next)

### POMDP-Based (Statistical)

Models dialogue as Partially Observable Markov Decision Process:
- **State**: True user intent (hidden, we infer from observations)
- **Observation**: What user said (possibly ambiguous)
- **Action**: System response
- **Reward**: Task completion (positive), turns (negative)

Solved via reinforcement learning. Computationally expensive but handles uncertainty well.

### Neural Dialogue Managers

- **End-to-end**: Seq2Seq maps dialogue history → response (blended with LLM)
- **Policy networks**: DQN/RNN learns dialogue policy
- **Hybrid**: Neural for NLU + rule-based for safety-critical paths

### LLM-Based (Current State)

Modern systems use LLMs as dialogue managers:
- **Prompt as policy**: System instructions define behavior
- **Memory as state**: Conversation history maintains state
- **Tool use**: Function calling for actions (book flight, check weather)
- **In-context learning**: Few-shot examples teach patterns

### Self-Correction in Dialogue

When the system realizes it made a mistake:
1. **Detect**: Low confidence, user correction ("no, I said X"), contradiction
2. **Recover**: Apologize, restate understanding, ask clarification
3. **Update**: Modify internal state, don't repeat mistake

Techniques:
- **Confidence-based routing**: If NLU confidence < threshold, ask clarification
- **Self-reflection**: "Did my last response address the user's question?"
- **Chain-of-thought**: Explain reasoning before committing to answer

---

## 4. Error Handling & Self-Correction

### Error Types in Chat Systems

| Error Type | Example | Detection |
|------------|---------|-----------|
| NLU error | "book" → wrong intent | Low confidence, user follow-up |
| Context error | Wrong entity resolution | Contradiction with history |
| Knowledge error | Wrong fact | User correction, fact-checking |
| Generation error | Hallucination | Self-review, grounding check |
| Dialogue error | Wrong topic | User says "no, I meant X" |

### Confidence-Based Detection

**Calibration methods:**
- Temperature scaling (post-hoc)
- Platt scaling
- Isotonic regression
- Ensemble disagreement

**When confidence is low:**
1. Ask clarification: "Did you mean X or Y?"
2. Offer multiple options: "Here are some possibilities..."
3. Fallback to human (for critical applications)

### Self-Correction Mechanisms

**Reflexion (Shinn et al., 2023):**
- Generate response
- Self-evaluate against criteria
- Generate reflective feedback
- Revise response

**Chain-of-Thought (Wei et al., 2022):**
- Force model to reason step-by-step
- Each step can be verified
- Errors caught early in reasoning chain

**Self-Refine (Madaan et al., 2023):**
- Generate initial output
- Provide feedback to self
- Refine based on feedback
- Iterate

**Constitutional AI (Anthropic, 2022):**
- Set of principles (constitution)
- Model evaluates own responses against principles
- Revises if violation detected

### Practical Error Recovery

1. **Acknowledge**: "I think I misunderstood..."
2. **Restate**: "You want to book a flight to X on Y, correct?"
3. **Correct**: Update internal state
4. **Proceed**: Continue with corrected understanding

---

## 5. Key Insights for AGI System Design

### What Current Systems Do Well
- Fast intent classification (transformers)
- Probabilistic spelling correction (noisy channel)
- Template-based dialogue (reliable for narrow domains)
- Confidence-based fallback

### What Current Systems Do Poorly
- True understanding (vs pattern matching)
- Long-term memory across sessions
- Genuine error recovery (vs scripted apologies)
- Adaptation to individual users
- Compositional reasoning

### Gap Analysis for AGI

| Capability | Current | AGI Goal |
|------------|---------|----------|
| Intent recognition | Pattern-based | Understanding |
| Spelling correction | Character-level | Intent-aware |
| Dialogue management | Template/LLM | Reasoning-based |
| Error handling | Confidence-based | Self-reflective |
| Memory | Session-only | Lifelong, structured |
| Learning | Pre-trained | Continual |

### Architectural Principles from This Study

1. **Layered understanding**: Characters → Words → Intent → Dialogue → Task
2. **Uncertainty propagation**: Confidence at each layer, aggregation across layers
3. **Feedback loops**: Self-correction at every level
4. **Structured memory**: Not just embeddings, but explicit representations
5. **Compositionality**: Simple units combine to handle complex cases

---

## References

- Louvion et al., "Dialogue Management", 2023
- Chen et al., "BERT for Joint Intent Classification and Slot Filling", 2019
- SymSpell, Wolf Garbe, 2019
- POMDP-based dialogue management, Young et al., 2013
- Reflexion, Shinn et al., 2023
- Self-Refine, Madaan et al., 2023
- Constitutional AI, Anthropic, 2022
