#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys

def apply_patch_main(patch_file_path, target_file_path, dry_run=False, verbose=False):
    """
    Applies a patch to a target file using the system's patch utility.

    Args:
        patch_file_path (str): Path to the patch file.
        target_file_path (str): Path to the target file.
        dry_run (bool): If True, simulates application without modifying files.
        verbose (bool): If True, increases output verbosity.

    Returns:
        bool: True if patch application (or simulation) was successful, False otherwise.
    """

    if not os.path.exists(patch_file_path):
        print(f"Error: Patch file not found: {patch_file_path}", file=sys.stderr)
        return False

    # For dry-run, target_file might not exist if the patch creates it.
    # The patch utility handles "can't find file to patch" errors appropriately.
    # For actual application, if the patch is not for file creation, the target should ideally exist.
    # However, relying on `patch` to report this is more robust.

    command = ["patch"]

    if dry_run:
        command.append("--dry-run")

    # Common options:
    # -p<n> Strip the smallest prefix containing <n> leading slashes from file names.
    # We'll use -p1 as a common default, assuming patches are generated relative to a project root.
    # Users might need to adjust this or the patch paths themselves for other scenarios.
    # For simplicity in this initial tool, we won't make -p configurable yet.
    command.extend(["-p1", "--input", patch_file_path, target_file_path])

    if verbose:
        print(f"Executing command: {' '.join(command)}", file=sys.stderr)

    try:
        # Note: `patch` typically reads the target file from the current directory
        # if it's a relative path in the patch header, or uses the provided target_file_path
        # as the file to patch if the patch doesn't specify a path or uses relative paths
        # that align. For simplicity, we pass target_file_path directly.
        # If patch file has `--- a/file.txt` and `+++ b/file.txt`, and target_file_path is `original_file.txt`,
        # `patch original_file.txt <patch_file_path>` is a common invocation.
        # The behavior of `patch` with absolute/relative paths in diffs can be complex.
        # We are aiming for a common use case. The `-p1` helps with typical project-based diffs.
        # If target_file_path is '-', patch reads from stdin, which is not intended here.

        # To ensure `patch` writes to the correct target_file_path when the patch itself
        # might contain different paths (e.g. `--- a/src/file.txt`), we can use `-o <outfile>`
        # if we want to force output to a specific file different from what `patch` would choose.
        # However, for applying a patch *to* target_file_path, the typical usage is `patch <target_file_path> -i <patch_file_path>`.
        # Let's adjust the command order for clarity and common usage:
        command = ["patch"]
        if dry_run:
            command.append("--dry-run")
        if verbose: # some patch versions use --verbose, others just output more with non-error
            pass # command.append("--verbose") # GNU patch doesn't have --verbose, it's default on failure

        command.extend(["-p1", target_file_path, "--input", patch_file_path])


        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if verbose or process.returncode != 0:
            if process.stdout:
                print("--- STDOUT ---", file=sys.stderr)
                print(process.stdout, file=sys.stderr)
            if process.stderr:
                print("--- STDERR ---", file=sys.stderr)
                print(process.stderr, file=sys.stderr)

        if process.returncode == 0:
            if dry_run:
                print(f"Dry run: Patch can be applied successfully to {os.path.basename(target_file_path)}.", file=sys.stdout)
            else:
                print(f"Patch applied successfully to {os.path.basename(target_file_path)}.", file=sys.stdout)
            return True
        else:
            print(f"Error: Patch application failed for {os.path.basename(target_file_path)}. Exit code: {process.returncode}", file=sys.stderr)
            # GNU patch exit codes: 0 for success, 1 for some hunks failed, 2 for serious trouble.
            if process.returncode == 1:
                print("Note: One or more hunks may have failed to apply or were already applied.", file=sys.stderr)
            return False

    except FileNotFoundError:
        print("Error: The 'patch' command was not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Apply a standard unified diff patch to a file using the system 'patch' utility.",
        epilog="Example: standard_patch_applier.py my_feature.patch src/main.py"
    )
    parser.add_argument("patch_file", help="Path to the patch file (e.g., output of diff -u)")
    parser.add_argument("target_file", help="Path to the file to be patched")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate patch application, show what would happen, but do not modify files."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase output verbosity, showing stdout/stderr from the patch command."
    )

    args = parser.parse_args()

    if not apply_patch_main(args.patch_file, args.target_file, args.dry_run, args.verbose):
        sys.exit(1)

if __name__ == "__main__":
    main()
