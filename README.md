# Qwen3.8-27B and ThinkingCap on RTX 3090s: moved to Merlin

**This repository is retired. Qwen3.8-27B and ThinkingCap are now served by Merlin:
[github.com/halt95/merlin-engine](https://github.com/halt95/merlin-engine).**

Merlin 3.0.0 runs both checkpoints on one, two or four RTX 3090s (and other Ampere and Ada cards) with one install and
one command, together with Flash-Next, Swift1.5, Nemotron 3.5 Super VL and Ornith 1.5:

```bash
git clone https://github.com/halt95/merlin-engine && cd merlin-engine
./setup.sh --model 27b                # or --model 27b-thinkingcap
```

- Release notes: [Merlin 3.0.0](https://github.com/halt95/merlin-engine/releases/tag/merlin-v3.0.0)
- The 27B recipe, bench cards and history: [models/qwen38-27b](https://github.com/halt95/merlin-engine/tree/main/models/qwen38-27b)
  and [models/thinkingcap](https://github.com/halt95/merlin-engine/tree/main/models/thinkingcap)
- Checkpoints (unchanged): [halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) and
  [halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin)

The earlier releases stay here as tags: [v1.3.1](https://github.com/halt95/qwen38-27b-3090s/tree/v1.3.1),
[v1.3.0](https://github.com/halt95/qwen38-27b-3090s/tree/v1.3.0) and
[v1.0.0](https://github.com/halt95/qwen38-27b-3090s/tree/v1.0.0). Their README, patches and bench cards are unchanged at
those tags. Issues and new work go to [halt95/merlin-engine](https://github.com/halt95/merlin-engine/issues).
