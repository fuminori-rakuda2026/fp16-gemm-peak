---
name: GPU result
about: Post a measurement from your own GPU
title: "[result] <GPU name> — <mean TFLOPS at 16384>"
labels: result
---

## What I ran

```
python3 measure_fp16.py --sizes 4096,8192,16384
```

## Output

Paste the whole block, including the `#` line (it carries torch version and clock/thermal state).

```
(paste here)
```

## Anything unusual

Co-tenancy, power limit, laptop on battery, thermal throttling, driver version — anything that
would change the number.
