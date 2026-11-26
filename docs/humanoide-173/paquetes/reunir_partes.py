from pathlib import Path
import hashlib,json
base=Path(__file__).resolve().parent
for item in json.loads((base/"partes.json").read_text(encoding="utf8")):
    out=base/item["name"]; h=hashlib.sha256()
    with out.open("wb") as dst:
        for name in item["parts"]:
            with (base/name).open("rb") as src:
                while chunk:=src.read(1048576):
                    dst.write(chunk); h.update(chunk)
    if h.hexdigest()!=item["sha256"]: raise RuntimeError("SHA-256 incorrecto: "+str(out))
    print("Verificado:",out)
