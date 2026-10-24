# Local throughput tuning

Apple M3 Pro, 36 GB unified memory, PyTorch 2.14.1, float32. The original configuration ran MASt3R inference on MPS and sparse alignment on CPU, with eight PyTorch CPU threads.

## Controlled measurements

The following trials use the same frozen 20-view inputs and the same completed neural pair cache. Each trial recomputes canonical maps and alignment from scratch. The short throughput trials use 60 coarse + 60 fine iterations and are kept separate from experimental results.

| Configuration | Wall time | Effective seconds per condition |
|---|---:|---:|
| CPU, 8 threads, one condition | 57.82 s | 57.82 s |
| CPU, 8 threads each, two concurrent conditions | 70.43 s | 35.22 s |
| CPU, 8 threads each, three concurrent conditions | 81.10 s | 27.03 s |
| MPS, one condition, CPU host 4 threads | 19.87 s | 19.87 s |
| MPS, CPU host 8 threads each, two concurrent conditions | 29.07 s | 14.54 s |
| MPS, CPU host 8 threads each, three concurrent conditions | 35.28 s | 11.76 s |

Three concurrent MPS conditions give approximately **4.9x the baseline cached-alignment throughput** in this benchmark. This ratio does not include inference for uncached neural pairs. CPU thread tuning alone was less effective: a single 20-view trial took 58.80 s at one thread, 47.91 s at four threads, and 57.82 s at eight threads.

The full **300+300 iteration MPS alignment** was also exercised successfully on the 20-view set: 75.65 s with cached neural pairs, including fresh canonical maps and dense output. No model, image resolution, view set, blur parameter, graph, iteration count, learning rate or precision was reduced in the production sweep.

## Applied configuration

- Three worker processes, eight CPU threads each.
- MASt3R neural inference and sparse alignment both on MPS.
- Shared, verified neural pair cache; independently recomputed canonical maps and alignment for each condition.
- Locked shared baseline predictions and metric records; stable target ownership across context sizes.
- New results in `../office_fast/`; previous CPU-alignment results preserved in `../office_sweep/`.
- Completed VGGT results reused unchanged. MASt3R clean/DROP/ADD alignment is consistently recomputed on MPS to avoid mixing CPU and GPU baselines.

The live three-worker run was inspected through macOS IOAccelerator counters and reported **100% device utilization** in the observed sample. Live 5-view add conditions completed in roughly 7–8 seconds while workers overlapped; missing neural pairs can increase condition times. These live timings are not the controlled benchmark above.

Process-safe result merging and cache compatibility are covered by the regression tests in `experiments/test_parallel.py`. Raw stage timings, profiles and benchmark predictions remain in this directory for inspection.
