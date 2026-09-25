# AGI Research Lab — Master Plan
*Strategic Roadmap v1.0 | Created by planner*

## 1. Project Definition

**Mission:** Engineer an AI assistant research system that investigates and advances general intelligence through modular, verifiable research.

**Core Thesis:** General intelligence emerges from the interaction of several subsystems — memory, perception, learning, reasoning, and action — not from a single monolithic model. We build, test, and refine each subsystem independently, then compose them.

**Constraints:**
- Research-first, engineering-supported
- Every claim must be verifiable
- Modular architecture — components swappable
- Low-cost operation where possible
- Documentation is not optional

---

## 2. Architecture Overview (Preliminary)

```
┌─────────────────────────────────────────────────────┐
│                  AGI Research Lab                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐     │
│  │Perception│  │ Memory  │  │ Reasoning Core  │     │
│  │ System   │  │ System  │  │ (Learning +     │     │
│  │          │  │         │  │  World Models)  │     │
│  └────┬─────┘  └────┬────┘  └────────┬────────┘     │
│       │              │               │              │
│       └──────────────┼───────────────┘              │
│                      │                              │
│              ┌───────▼───────┐                      │
│              │  Action/Output │                      │
│              │    System      │                      │
│              └───────┬───────┘                      │
│                      │                              │
│  ┌───────────────────▼────────────────────┐        │
│  │         Evaluation Framework           │        │
│  └────────────────────────────────────────┘        │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  Research Layer (papers, experiments,      │    │
│  │  benchmarks, documentation)                │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 3. Phased Roadmap

### Phase 1: FOUNDATION (Weeks 1-2)
*Goal: Research baseline + architecture definition + team alignment*

| Task | Owner | Dependencies | Deliverable |
|------|-------|--------------|-------------|
| Literature survey of cognitive architectures | researcher | — | `01_RESEARCH/cognitive_architectures.md` |
| Survey of memory systems in AI | researcher | — | `01_RESEARCH/memory_systems.md` |
| Survey of world models + perception | researcher | — | `01_RESEARCH/world_models.md` |
| Define system architecture | definer | all surveys | `04_ARCHITECTURE/system_overview.md` |
| Define interfaces between components | definer | architecture | `04_ARCHITECTURE/interfaces.md` |
| Set up project infrastructure | coder-1 | — | `19_PROTOTYPES/` scaffold |
| Set up documentation framework | documenter | — | `09_DOCUMENTATION/` template |
| Establish evaluation criteria | verifier | architecture | `08_EVALUATION/criteria.md` |

### Phase 2: CORE PROTOTYPES (Weeks 3-5)
*Goal: Working prototypes of each core subsystem*

| Task | Owner | Dependencies | Deliverable |
|------|-------|--------------|-------------|
| Memory system prototype (episodic + semantic) | coder-2 | interfaces | `06_MEMORY_SYSTEMS/` |
| Skill/learning architecture prototype | coder-2 | interfaces | `06_MEMORY_SYSTEMS/skills/` |
| Perception pipeline (text + vision) | coder-3 | interfaces | `07_PERCEPTION/` |
| World model prototype (predictive) | coder-3 | interfaces | `07_PERCEPTION/world_model/` |
| Experiment framework (run + log) | coder-1 | interfaces | `19_PROTOTYPES/framework/` |
| Benchmark suite (basic) | coder-4 | eval criteria | `08_EVALUATION/benchmarks/` |
| Design interaction flows | designer | architecture | `03_DESIGN/interaction_flows.md` |
| Design visualization system | designer | architecture | `03_DESIGN/viz_system.md` |

### Phase 3: INTEGRATION (Weeks 6-7)
*Goal: Compose subsystems into a working research assistant*

| Task | Owner | Dependencies | Deliverable |
|------|-------|--------------|-------------|
| Integrate memory + reasoning core | coder-1, coder-2 | prototypes | `05_CODE/integrated_system/` |
| Add perception layer | coder-3, coder-1 | prototypes | `05_CODE/integrated_system/` |
| Build research assistant UI | designer, coder-1 | interaction flows | `05_CODE/ui/` |
| Integration tests | verifier, coder-4 | integrated system | `11_VERIFICATION/integration/` |
| Performance profiling | performance-fixer | integrated system | `10_VERFORMANCE/reports/` |
| Bug sweep of all code | bug-finder | all code | `13_BUG_REPORTS/` |

### Phase 4: EXPERIMENTATION (Weeks 8-10)
*Goal: Run experiments, iterate, publish findings*

| Task | Owner | Dependencies | Deliverable |
|------|-------|--------------|-------------|
| Run memory experiments | coder-2, researcher | integrated system | `01_RESEARCH/memory_experiments.md` |
| Run world model experiments | coder-3, researcher | integrated system | `01_RESEARCH/world_model_experiments.md` |
| Run skill learning experiments | coder-2, researcher | integrated system | `01_RESEARCH/skill_experiments.md` |
| Evaluate all subsystems | coder-4, verifier | experiments | `08_EVALUATION/results.md` |
| Performance optimization pass | performance-fixer | all experiments | `10_VERFORMANCE/optimized/` |
| Documentation pass | documenter | everything | `09_DOCUMENTATION/final/` |

### Phase 5: COMPOSITION & SCALING (Weeks 11-12)
*Goal: Full system composition, scaling experiments, final report*

| Task | Owner | Dependencies | Deliverable |
|------|-------|--------------|-------------|
| Full system integration | all coders | Phase 4 complete | `05_CODE/final_system/` |
| Scaling experiments | coder-4, researcher | final system | `01_RESEARCH/scaling.md` |
| Final architecture review | definer | everything | `04_ARCHITECTURE/final.md` |
| Final documentation | documenter | everything | `09_DOCUMENTATION/` |
| Final verification + audit | verifier, bug-finder | everything | `11_VERIFICATION/final.md` |
| Performance report | performance-fixer | everything | `10_VERFORMANCE/final.md` |
| Monitoring dashboard | watcher | final system | `12_MONITORING/` |

---

## 4. Dependency Map

```
                    ┌──────────────┐
                    │  researcher  │ (surveys → experiments)
                    └──────┬───────┘
                           │ informs
                           ▼
                    ┌──────────────┐
                    │   definer    │ (architecture → interfaces)
                    └──────┬───────┘
                           │ defines
                           ▼
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  coder-2     │   │  coder-3     │   │  coder-1     │
│  (memory,    │   │  (perception,│   │  (framework, │
│   learning)  │   │   world)     │   │   integration)│
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │ composed by
                          ▼
                   ┌──────────────┐
                   │  coder-4     │ (evaluation, benchmarks)
                   └──────┬───────┘
                          │ evaluates
                          ▼
              ┌───────────────────────┐
              │  verifier + bug-finder│
              └───────────┬───────────┘
                          │ validated by
                          ▼
              ┌───────────────────────┐
              │  performance-fixer    │
              │  + watcher            │
              └───────────────────────┘
                          │ monitored by
                          ▼
                   ┌──────────────┐
                   │ documenter   │ (chronicler — runs parallel)
                   └──────────────┘
```

---

## 5. Task Assignments

### researcher
**Phase 1:** Cognitive architectures survey, memory systems survey, world models survey
**Phase 2:** Experiment design for each prototype
**Phase 3:** Integration testing support
**Phase 4:** Run experiments, analyze results
**Phase 5:** Scaling experiments, final research synthesis

### definer
**Phase 1:** System architecture, interface specifications
**Phase 2:** Component specifications, data flow diagrams
**Phase 3:** Integration architecture
**Phase 4:** Architecture refinement based on findings
**Phase 5:** Final architecture document

### coder-1 (Forge)
**Phase 1:** Project scaffold, experiment framework
**Phase 2:** Integration framework, API layer
**Phase 3:** System integration, UI scaffolding
**Phase 4:** Experiment tooling
**Phase 5:** Final system assembly

### coder-2 (Anvil)
**Phase 1:** Memory system design
**Phase 2:** Episodic memory prototype, semantic memory prototype, skill system prototype
**Phase 3:** Integration with reasoning core
**Phase 4:** Memory experiments
**Phase 5:** Memory system final

### coder-3
**Phase 1:** Perception system design
**Phase 2:** Text processing pipeline, vision pipeline, world model prototype
**Phase 3:** Perception integration
**Phase 4:** Perception experiments
**Phase 5:** Perception system final

### coder-4
**Phase 1:** Evaluation criteria, benchmark design
**Phase 2:** Benchmark suite, metrics framework
**Phase 3:** Integration tests
**Phase 4:** Full system evaluation
**Phase 5:** Final benchmark suite, scaling tests

### designer
**Phase 1:** UI requirements, interaction flows
**Phase 2:** Wireframes, visualization system design
**Phase 3:** Working UI prototype
**Phase 4:** UI refinement
**Phase 5:** Final UI + interaction design doc

### documenter
**Phase 1:** Documentation framework, templates
**Phase 2:** Research notes, experiment logs
**Phase 3:** Integration docs, architecture docs
**Phase 4:** Results documentation
**Phase 5:** Final documentation package

### orchestrator
**Phase 1:** Task assignment, coordination setup
**Phase 2:** Progress tracking, conflict resolution
**Phase 3:** Integration coordination
**Phase 4:** Experiment coordination
**Phase 5:** Final project coordination

### performance-fixer
**Phase 1:** Performance baseline, profiling setup
**Phase 2:** Prototype profiling
**Phase 3:** Integration profiling
**Phase 4:** Experiment profiling
**Phase 5:** Final optimization pass

### verifier
**Phase 1:** Verification criteria, test plan
**Phase 2:** Prototype verification
**Phase 3:** Integration verification
**Phase 4:** Experiment verification
**Phase 5:** Final audit

### watcher
**Phase 1:** Monitoring setup, alert config
**Phase 2:** Prototype monitoring
**Phase 3:** Integration monitoring
**Phase 4:** Experiment monitoring
**Phase 5:** Production monitoring

### bug-finder
**Phase 1:** Security audit plan
**Phase 2:** Prototype bug sweep
**Phase 3:** Integration bug sweep
**Phase 4:** Experiment bug sweep
**Phase 5:** Final security audit

---

## 6. Key Decisions — LOCKED ✅

| # | Decision | Locked Value |
|---|----------|--------------|
| 1 | **Language/Framework** | Python + NumPy only. No PyTorch, no TensorFlow, no autograd, no JAX |
| 2 | **Compute** | Local CPU. Neuromorphic-ready: event-driven, sparse, no GPU dependency |
| 3 | **Base Models** | NONE. From scratch, random initialization, Hebbian learning only |
| 4 | **Scope** | MNIST → CIFAR-10 → Split-MNIST continual → 2D grid world |
| 5 | **Publication** | MIT license, public GitHub, arXiv target ("Hierarchical Predictive Coding learns MNIST without backpropagation") |

---

## 8. First Principles Deduction

The full mathematical framework and architecture design is at:
**`04_ARCHITECTURE/first_principles_deduction.md`**

**Key deductions:**
- Bidirectional information flow (perception + prediction)
- Local eligibility traces + global neuromodulatory signal (no backprop)
- Sparse distributed representations (KWTA)
- Energy-based dynamics (guaranteed convergence)
- Complementary learning systems (fast episodic + slow semantic)
- Temporal coherence via recurrent attractor dynamics

**Architecture:** Predictive Equilibrium Network (PEN) — hierarchical, recurrent, sparse, energy-based.

**Core formula (weight update):**
$$\Delta w_{ij}^{l,\text{ff}} = \eta \cdot G \cdot \epsilon_i^l \cdot a_j^{l-1}$$

where $G$ is the global surprise signal, $\epsilon$ is prediction error, and activations are sparse.

---

## 10. Immediate Next Steps (UPDATED for "Deduce from Scratch")

1. **researcher** starts literature survey immediately
2. **definer** begins architecture draft based on preliminary overview
3. **coder-1** scaffolds `19_PROTOTYPES/`
4. **orchestrator** confirms all agents have their assignments
5. **documenter** creates documentation template
6. All other agents prepare their Phase 1 work

---

## 11. Final Architecture

The unified system design is at:
**`04_ARCHITECTURE/RT_ALM_FINAL.md`**

**System:** RT-ALM (Real-Time Adaptive Language Model) — 4-level hierarchical predictive coding with online discriminative learning. No attention, no backprop, no pre-training. All sparse, all online.

**Novelty:** Online discriminative learning rule, real-time predictive coding, multimodal SDR fusion.

### 11.1 Corrected Novelty Assessment (via TRANS_CORRECTED.md)

The corrected architecture document (`04_ARCHITECTURE/TRANS_CORRECTED.md`) identifies:
- **Not novel:** k-WTA, XOR binding, attractor dynamics, SDRs, eligibility traces, complementary learning, random projections, predictive coding — all well-established
- **Potentially novel:** Real-time integration of all components, online discriminative learning rule, multimodal SDR fusion
- **Contradictions to resolve:** Global learning signal in PEN, SGD in ContinuousLearner both violate "local learning only" claim

### 11.2 Known Issues (Not Yet Filed)

1. XOR binding for continuous vectors — non-invertible
2. Global learning signal — contradicts local learning claim
3. ContinuousLearner uses SGD (backprop) — contradicts "no backprop" claim
4. Complementary learning replay buffer — not implemented
5. Random projection encoders are frozen — quality unknown
6. No test coverage across any component

### 11.3 Survey Status

| Survey | Status |
|--------|--------|
| Real-Time Adaptive Language Model | complete |
| Hebbian Learning | pending |
| Predictive Coding | pending |
| Sparse Coding | pending |
| Continual Learning | pending |
| World Models | pending |
| Memory Systems | pending |
| Compositional Representations | pending |
| Real-Time Systems | pending |
| Multimodal Fusion | pending |

---

*Last updated: 2026-04-28 — documenter@Chronicler*

*End of Master Plan v1.1 — planner*
