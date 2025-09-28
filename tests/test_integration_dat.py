import subprocess
from subprocess import DEVNULL
import numpy as np
from pathlib import Path
import pytest

EXAMPLES_DIR = Path("examples")
REFERENCE_DIR = Path("tests/reference/Fortran")


examples = ["example01", "example02", "example04"]


def collect_dat_files(example_name: str):
    ref_dir = REFERENCE_DIR / example_name
    out_dir = EXAMPLES_DIR / example_name / "output"

    ref_files = sorted(ref_dir.glob("*.dat"))
    output_files = {f.name: f for f in out_dir.glob("*.dat")}

    for ref_file in ref_files:
        fname = ref_file.name
        if fname not in output_files:
            yield f"Missing output file: {fname} in {out_dir}", None, None
        else:
            yield None, ref_file, output_files[fname]


@pytest.mark.parametrize("example", examples)
def test_example_outputs_against_reference(example, tmp_path):
    example_path = EXAMPLES_DIR / example
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    scripts = sorted(example_path.glob("main_*.py"))
    yamls = sorted(example_path.glob("*.yaml"))

    if not scripts:
        pytest.fail(f"No main_*.py scripts found in {example_path}")
    if not yamls:
        pytest.fail(f"No .yaml input files found in {example_path}")

    for script in scripts:
        prefix = script.stem.split("_", 1)[-1]
        matching_inputs = [y for y in yamls if y.name.startswith(prefix)]

        if not matching_inputs:
            pytest.fail(
                f"No matching inputs for script {script.name} "
                f"(looking for {prefix}*.yaml)"
            )

        for yaml_file in matching_inputs:
            try:
                subprocess.run(
                    ["python", script.name, yaml_file.name],
                    cwd=example_path,
                    check=True,
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                pytest.fail(
                    f"Script failed in {example}:\n"
                    f"   Script: {script.name}\n"
                    f"   Input : {yaml_file.name}\n"
                    f"   Error : {e}"
                )

    for dat_file in (example_path / "output").glob("*.dat"):
        target = output_dir / dat_file.name
        target.write_bytes(dat_file.read_bytes())

    TOL_RTOL = 5e-2
    TOL_ATOL = 5e-2
    errors = []

    for msg, ref_file, gen_file in collect_dat_files(example):
        if msg:
            errors.append(msg)
            continue

        try:
            ref_data = np.loadtxt(ref_file)
            gen_data = np.loadtxt(gen_file)
        except Exception as e:
            errors.append(f"Failed to load {ref_file.name} or {gen_file.name}: {e}")
            continue

        if ref_data.shape != gen_data.shape:
            errors.append(
                f"Shape mismatch in {ref_file.name}: "
                f"{ref_data.shape} (ref) vs {gen_data.shape} (gen)"
            )
            continue

        if not np.allclose(ref_data, gen_data, rtol=TOL_RTOL, atol=TOL_ATOL):
            diff = np.abs(ref_data - gen_data)
            max_diff = np.max(diff)
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            ref_val = ref_data[idx]
            gen_val = gen_data[idx]
            rel_diff = diff[idx] / np.abs(ref_val) if ref_val != 0 else float("inf")

            errors.append(
                f"\nMismatch in {ref_file.name}:\n"
                f"   ➤ Max difference : {max_diff:.3e} at index {idx}\n"
                f"   ➤ Fortran value  : {ref_val:.6e}\n"
                f"   ➤ Python value   : {gen_val:.6e}\n"
                f"   ➤ Absolute diff  : {diff[idx]:.6e}\n"
                f"   ➤ Relative diff  : {rel_diff:.6e}\n"
                f"   ➤ Tolerances     : atol={TOL_ATOL:.1e}, rtol={TOL_RTOL:.1e}"
            )

    if errors:
        full_report = "\n\n".join(errors)
        pytest.fail(f"Integration test failed for {example}:\n\n{full_report}")
