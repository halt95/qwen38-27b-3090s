# Package manifest

Every file in this repository except this manifest and `release/SHA256SUMS`, with its Git
mode, size in bytes and SHA-256 of the committed bytes. `release/SHA256SUMS` covers this
manifest too. Checkpoint files are listed in `release/checkpoint.sha256`.

| file | mode | bytes | sha256 |
|---|---|---:|---|
| `.dockerignore` | 100644 | 515 | `5e7f25ebd5848472e2b9ebae1008b5578d40595c83288a11d0dab275d3aa9dc7` |
| `.gitattributes` | 100644 | 210 | `28f6a02ec2d74085a3ccceb3ece67d3f4051a823db9ea474f039f4b30b7c1493` |
| `.gitignore` | 100644 | 32 | `81b54bd26992a95306b15f8edd81a854ea95b20a2a80c2fa7ebcff5b04e06b48` |
| `CHANGELOG.md` | 100644 | 1824 | `55b15830bc5303b97c8b3a52ec29fdd7245c09b2cef712b18b4eba0a37e6b875` |
| `Dockerfile` | 100644 | 6521 | `9d475ff8ad905cda8ebd2131d384b8e873e0257130a4da00602f9b87fac62105` |
| `LICENSE` | 100644 | 11358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | 100644 | 1944 | `ab2e0daa60fa83c4586af7b1453e96dfa1a98985670c149eceb5d6ffe59db57a` |
| `README.md` | 100644 | 45244 | `52437ec4132bbd5f20d017fc0c69a4aa11ecd9fdc8098bcce751e4a91371a139` |
| `benchmarks/2026-09-25/BENCH-CARD.md` | 100644 | 7750 | `d6d71056479509eab4147af96b9a603ee241691033fb87ddb3ffb1564187fef6` |
| `benchmarks/2026-09-26/BENCH-CARD.md` | 100644 | 13155 | `88e2a5c64b7d5941eb4c989893f9f741407c55e1f3821c7ffab54c7dcfb509bf` |
| `benchmarks/2026-09-26/qwen38-27b-v1.0.0-ctx-pp-tg-itl.png` | 100644 | 226596 | `46486ae333ab96c6747c3c5af6f4aac54b4003d069361aa307799d5f439c8664` |
| `docker-compose.yml` | 100644 | 1821 | `698658a4519dd1de75161de1da6e9296cb2fbbb122fddbef3287825e91698806` |
| `docs/images/qwen38-27b-v1.0-layout.png` | 100644 | 210362 | `40cc79bd8072eccb470fc9e2ebfb65c840bef6ef82a63e941e7952e794013981` |
| `docs/images/qwen38-27b-v1.0-long-prefill.png` | 100644 | 146769 | `64ebf2beb08ebd3af4f035c49263d0c7f03ca636b06f231e752638468bf535d0` |
| `docs/images/qwen38-27b-v1.0-summary.png` | 100644 | 469639 | `90f19cd037e0f4aaa53f621308767410f40496da842524f91f0365ea1a9c8166` |
| `docs/loop-check.md` | 100644 | 4313 | `f87ff1082e7f9a86e374ebcb3dc138d5c18749de439df255fc708376f8f60153` |
| `patches/vllm-0.30.0/0001-flashinfer-port-vllm-47979-uniform-multi-token-decode.patch` | 100644 | 9727 | `814748c6674d22217bed4fd5f341e4d3284aaea05a39e818ecc8ec5fed306539` |
| `patches/vllm-0.30.0/0002-flashinfer-cache-the-decode-plan-when-the-schedule-c.patch` | 100644 | 5160 | `023f3acdd557c3f3ea477c3d6f07921a77c5718735adcf4d8104cec88d7b530f` |
| `patches/vllm-0.30.0/0003-host-resident-embedding-table-on-by-default.patch` | 100644 | 5927 | `0dd3da2638f524950d066f08ff1d70057caf5c1a39bdf18382b10b9df31e8ec4` |
| `patches/vllm-0.30.0/0004-gdn-split-K-GEMM-for-in_proj_ba-at-small-token-count.patch` | 100644 | 7951 | `238ba0d7842bc75ab538db51ef8bbc64d4c8f0b288389ff1543731c6dfa3c61d` |
| `patches/vllm-0.30.0/0005-spec-decode-skip-the-seq_lens-device-sync-on-MTP-dra.patch` | 100644 | 9815 | `92683edc303fabfdc4c673bed3cc9779fda3a2949e863c5407e5a19e40c245d3` |
| `patches/vllm-0.30.0/README.md` | 100644 | 7781 | `1a4dcf065250d9ed1cb3b564adce255c9cf74bbb81f0b8554dceb4de60b323f4` |
| `quant/artifact_identity.py` | 100755 | 6000 | `4eb132971f8474349e2f3d016b8b84b252e71ccf725e35366c0c33abf5aae0d0` |
| `quant/kv_scales-philbert-e4m3.json` | 100644 | 2996 | `4f75f69644b212c6536822202bf17f1f471a68d0805de51193e6db18efd95d61` |
| `quant/kvcal.py` | 100644 | 2161 | `b8f45f0067e451dd1351f54ec2d168ddd6e42de30c79ea82b5d9eec4a7c1c2b0` |
| `quant/kvcal_write.py` | 100755 | 3954 | `1a24f6f286e07fdc32f95dc68373536b95f027778cac5f516392806b553157ef` |
| `quant/mtpcal.py` | 100755 | 5554 | `53a05da857d5b804839589c58f5239869fb851d4d88d37fabbbda294abb7199e` |
| `quant/quantize_head_mtp.py` | 100755 | 13791 | `dfb1b5cbc5a768042a305cb23adff7d413ad17145ca45b42de700445d0cea78e` |
| `release/checkpoint.sha256` | 100644 | 965 | `0d01f839e46ca586b1c6884eb56c9cd549946c51a8ce5cb71d28ac29418a34b4` |
| `release/hash_lock.py` | 100755 | 5975 | `3f02902241e3a50fe03443e1bed47c1d047fc3114599f3662a4e9d1034d890fa` |
| `release/install-env.sh` | 100755 | 3176 | `521da3c0f92f2ae6ca053cf47fe8222aab4a511b401891e8bed05c4aa4b3686a` |
| `release/make_manifest.py` | 100755 | 3103 | `1e4c44e8ba35fb8084127266cd47071373536f661cb10ce404167cf229b53f90` |
| `release/requirements-container-toolkit.txt` | 100644 | 855 | `8353f513392de21fe30e34affa96f883533aff0cc1bb9c5adb227b4f67bb7834` |
| `release/requirements-pinned.txt` | 100644 | 22321 | `996f0dd41bdcdb1082b9e0f416e565224840b4621e553b4ba2f5b1f29b85e57a` |
| `release/requirements-pip.txt` | 100644 | 176 | `04064a5ecb22b4af65078f624c06e62d65b1b74f09f7bb4cd40d06f484466bd5` |
| `serve/docker-entrypoint.sh` | 100755 | 4427 | `a1ddc34e4a2b180041ad755642af9ee7a47ac691f47f0cedf05c4951b5aecb8e` |
| `serve/serve-tp4.sh` | 100755 | 4148 | `ce7cc8361c4b42ad246cd8063de5476a734c68dc4d3c362696aba064ac396133` |
| `serve/serve.sh` | 100755 | 4007 | `ef31458a86f47632661506108d52584dbfe61fe8270a451507494b4b3552ebcd` |
| `serve/templates/NOTICE` | 100644 | 521 | `92f64034f03d9ad3a49e27feb10cf58a823f2e910fd509e548c7070f2ca20d2d` |
| `serve/templates/froggeric-qwen-v22.5.jinja` | 100644 | 28234 | `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2` |
