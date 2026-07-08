#!/usr/bin/env python3
"""Create a Zenodo draft deposition and upload Final_datasets.

By default this script creates a draft and uploads files, but does not publish.
Set ZENODO_TOKEN in the environment before running.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import requests


PRODUCTION_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"


def _headers(token: str, *, json_content: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def _raise(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    raise RuntimeError(f"Zenodo API error during {action}: {response.status_code} {payload}")


def _iter_files(data_dir: Path, readme_path: Path) -> Iterable[tuple[Path, str]]:
    yield readme_path, "README.md"
    for path in sorted(data_dir.glob("*.csv")):
        yield path, path.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Final_datasets to a Zenodo draft deposition.")
    parser.add_argument("--data-dir", default="Final_datasets", type=Path)
    parser.add_argument("--metadata", default="zenodo_metadata.json", type=Path)
    parser.add_argument("--readme", default="docs/final_datasets_README.md", type=Path)
    parser.add_argument("--token-env", default="ZENODO_TOKEN")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org instead of production Zenodo.")
    parser.add_argument("--publish", action="store_true", help="Publish after upload. Use only after manually reviewing metadata and licenses.")
    parser.add_argument("--response-file", default="zenodo_deposition_response.json", type=Path)
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing Zenodo access token. Set {args.token_env}=<token> first.")

    if not args.data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_dir}")
    if not args.metadata.is_file():
        raise SystemExit(f"Metadata file not found: {args.metadata}")
    if not args.readme.is_file():
        raise SystemExit(f"Dataset README not found: {args.readme}")

    api = SANDBOX_API if args.sandbox else PRODUCTION_API
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    print(f"Creating Zenodo draft at {api} ...")
    response = requests.post(f"{api}/deposit/depositions", json={}, headers=_headers(token, json_content=True), timeout=60)
    _raise(response, "create deposition")
    deposition = response.json()
    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]

    print(f"Updating metadata for deposition {deposition_id} ...")
    response = requests.put(
        f"{api}/deposit/depositions/{deposition_id}",
        data=json.dumps(metadata),
        headers=_headers(token, json_content=True),
        timeout=60,
    )
    _raise(response, "update metadata")
    deposition = response.json()

    files = list(_iter_files(args.data_dir, args.readme))
    for path, remote_name in files:
        size_gb = path.stat().st_size / (1024 ** 3)
        print(f"Uploading {remote_name} ({size_gb:.3f} GiB) ...", flush=True)
        with path.open("rb") as handle:
            response = requests.put(
                f"{bucket_url}/{remote_name}",
                data=handle,
                headers=_headers(token),
                timeout=None,
            )
        _raise(response, f"upload {remote_name}")

    response = requests.get(f"{api}/deposit/depositions/{deposition_id}", headers=_headers(token), timeout=60)
    _raise(response, "retrieve final draft")
    deposition = response.json()
    args.response_file.write_text(json.dumps(deposition, indent=2), encoding="utf-8")

    doi = deposition.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    html = deposition.get("links", {}).get("html") or deposition.get("links", {}).get("latest_draft_html")
    print("Draft deposition created and files uploaded.")
    print(f"Deposition ID: {deposition_id}")
    if doi:
        print(f"Reserved DOI: {doi}")
    if html:
        print(f"Draft URL: {html}")
    print(f"Response saved to: {args.response_file}")

    if args.publish:
        print("Publishing deposition ...")
        response = requests.post(
            f"{api}/deposit/depositions/{deposition_id}/actions/publish",
            headers=_headers(token),
            timeout=120,
        )
        _raise(response, "publish deposition")
        published = response.json()
        args.response_file.write_text(json.dumps(published, indent=2), encoding="utf-8")
        print(f"Published record: {published.get('doi_url') or published.get('record_url')}")
    else:
        print("Not published. Review the draft on Zenodo, then rerun with --publish or publish from the web UI.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
