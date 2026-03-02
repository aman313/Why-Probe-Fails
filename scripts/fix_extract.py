from pathlib import Path
path = Path("/home/ubuntu/Why-Probe-Fails/src/extract_activations.py")
lines = path.read_text().splitlines()
out = lines[:391] + lines[517:]
path.write_text("\n".join(out) + "\n")
