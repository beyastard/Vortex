import numpy as np

rng = np.random.default_rng(42)
tokens = rng.integers(0, 32000, size=5_000_000, dtype=np.uint32)
tokens[:4_750_000].tofile("./data/train.bin")
tokens[4_750_000:].tofile("./data/val.bin")
print("Done — train.bin:", tokens[:4_750_000].nbytes // 1024 // 1024, "MB")
