"""Audit demonstrations, freeze splits, or add training-only DAgger rounds."""
import argparse
from pathlib import Path

from dexman.data.manifest import add_corrections, create_manifest, read_manifest
from dexman.data.validation import audit_dataset
from dexman.experiments import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "split", "field-release", "field-snapshot", "add-corrections"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--evidence",type=Path,help="Directorio de ensayos de cosecha en Isaac Sim")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; preserve the previous report/partitions")
    if args.action == "audit":
        result = audit_dataset(args.dataset)
    elif args.action == "split":
        result = create_manifest(args.dataset, args.seed)
    elif args.action in ("field-release", "field-snapshot"):
        from dexman.data.field_release import field_manifest
        if not args.evidence:
            parser.error("field-release requiere --evidence")
        result = field_manifest(args.dataset,args.evidence,snapshot=args.action=="field-snapshot")
    else:
        if not args.manifest:
            parser.error("add-corrections requires --manifest")
        manifest, _ = read_manifest(args.manifest)
        result = add_corrections(manifest, args.dataset)
    write_json(args.output, result)
    if args.action == "audit":
        print(f"Complete={len(result['episodes'])}, successes={result['successes']}, "
              f"corrections={result['corrections']}, partial={result['partial_files']}, errors={len(result['errors'])}")
        if not result["valid"]:
            raise SystemExit(2)
