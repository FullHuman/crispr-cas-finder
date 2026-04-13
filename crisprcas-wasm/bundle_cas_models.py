#!/usr/bin/env python3
"""Bundle CAS HMM profiles and model XMLs into a single JSON for the web app."""
import json
import os
import glob

cas_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CRISPRCasFinder", "CasFinder-2.0.3")
profiles_dir = os.path.join(cas_dir, "CASprofiles-2.0.3")
models_dir = os.path.join(cas_dir, "DEF-SubTyping-2.0.3")

models = []
for xml_file in sorted(glob.glob(os.path.join(models_dir, "*.xml"))):
    name = os.path.splitext(os.path.basename(xml_file))[0]
    with open(xml_file) as f:
        content = f.read()
    models.append({"name": name, "family": "CasFinder", "content": content})

profiles = []
for hmm_file in sorted(glob.glob(os.path.join(profiles_dir, "*.hmm"))):
    name = os.path.splitext(os.path.basename(hmm_file))[0]
    with open(hmm_file) as f:
        data = f.read()
    profiles.append({"name": name, "data": data})

bundle = {"models": models, "profiles": profiles}
out_path = os.path.join(os.path.dirname(__file__), "www", "cas-models.json")
with open(out_path, "w") as f:
    json.dump(bundle, f)

size = os.path.getsize(out_path)
print(f"Created {out_path}: {len(models)} models, {len(profiles)} profiles, {size / 1024 / 1024:.1f} MB")
