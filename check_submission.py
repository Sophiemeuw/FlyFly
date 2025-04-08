import os
import zipfile


def zip_submission_directory(directory_path, output_path=None):
    if output_path is None:
        output_path = Path(directory_path).as_posix() + ".zip"

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            # Modify dirs in-place to skip __pycache__ folders
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory_path)
                zipf.write(full_path, arcname=relative_path)
    return output_path


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Check submission directory.")
    parser.add_argument(
        "submission_dir",
        help="Path to the submission directory",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="Path to the output zip file",
        default=None,
    )
    args = parser.parse_args()

    if args.submission_dir is None:
        print(
            "Submission directory is not provided and is assumed to be at the default location."
        )
        submission_dir = Path("submission").absolute()
    else:
        submission_dir = Path(args.submission_dir).absolute()

    output_path = args.output

    # check if the submission directory exists
    assert (
        submission_dir.exists()
    ), f"The submission directory {submission_dir} does not exist."

    # check if the controller can be instantiated
    import sys
    import importlib

    sys.path.append(str(submission_dir.parent))
    package_name = submission_dir.name
    package_module = importlib.import_module(package_name)
    assert (
        package_module.controller.Controller()
    ), "The controller class cannot be instantiated. Please check your implementation."
    controller = package_module.controller.Controller()

    output_path = zip_submission_directory(submission_dir, output_path)
    print("Submission directory is valid.")
    print("Zipped submission directory to:", Path(output_path).absolute())
