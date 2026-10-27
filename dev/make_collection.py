"""Create the S2V collection in the kandinskylab organisation.

    .venv/bin/python dev/make_collection.py            # prompts for the token
    HF_TOKEN=... .venv/bin/python dev/make_collection.py --dry-run

One collection holding both formats, the native repository first and its
Diffusers bundle right after, distilled after base. The eight repositories are private, so the collection is
too — a public one would list items most visitors cannot open.

The token needs write access to the organisation itself; one scoped to a user
account sees the org as if it did not exist. It is read from `$HF_TOKEN` or
prompted for, never stored.
"""

from __future__ import annotations

import argparse
import getpass
import os

from huggingface_hub import add_collection_item, create_collection

NAMESPACE = "kandinskylab"
TITLE = "Kandinsky S2V 1.0"
# The Hub rejects anything over 150 characters here.
DESCRIPTION = (
    "An avatar animated from a reference frame and a speech track — the whole figure, "
    "not only the head. Lite and Pro, base and distilled, two formats."
)
ITEMS = (
    "Kandinsky-S2V-1.0-Lite",
    "Kandinsky-S2V-1.0-Lite-Diffusers",
    "Kandinsky-S2V-1.0-Lite-distill",
    "Kandinsky-S2V-1.0-Lite-distill-Diffusers",
    "Kandinsky-S2V-1.0-Pro",
    "Kandinsky-S2V-1.0-Pro-Diffusers",
    "Kandinsky-S2V-1.0-Pro-distill",
    "Kandinsky-S2V-1.0-Pro-distill-Diffusers",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print what would be created")
    parser.add_argument("--public", action="store_true", help="the sibling collections are public")
    args = parser.parse_args()

    visibility = "public" if args.public else "private"
    print(f"[collection] {NAMESPACE}/{TITLE} ({visibility})")
    for item in ITEMS:
        print(f"[collection]   {NAMESPACE}/{item}")
    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN") or getpass.getpass("HF token: ")
    collection = create_collection(
        title=TITLE,
        namespace=NAMESPACE,
        description=DESCRIPTION,
        private=not args.public,
        exists_ok=True,
        token=token,
    )
    for item in ITEMS:
        add_collection_item(
            collection.slug,
            item_id=f"{NAMESPACE}/{item}",
            item_type="model",
            exists_ok=True,
            token=token,
        )
        print(f"[collection]   + {item}")
    print(f"[collection] https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()
