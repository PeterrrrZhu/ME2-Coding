from pathlib import Path
import ast


# Target script to run
TARGET_SCRIPT = Path(__file__).with_name("notebook_testing_no_comment.py")

# Fast test parameters (override the larger values in the target script)
PARAM_OVERRIDES = {
    "n_r": 24,
    "n_theta": 16,
    "dt": 2.0e-6,
    "t_end": 0.05,
    "save_every": 20,
}

# Visual quick test: show contour/surface/center plot and animation.
VISUAL_TEST = True

# Skip Fourier execution in the called script.
SKIP_FOURIER = True


def apply_top_level_overrides(source_code, overrides):
    """Replace top-level parameter assignments in the target script."""
    tree = ast.parse(source_code, filename=str(TARGET_SCRIPT))
    replaced = set()

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in overrides:
                new_value = ast.parse(repr(overrides[target.id]), mode="eval").body
                node.value = new_value
                replaced.add(target.id)

    missing = [name for name in overrides if name not in replaced]
    if missing:
        raise ValueError(f"Could not find top-level assignments for: {missing}")

    ast.fix_missing_locations(tree)
    return tree


def remove_fourier_run_call(tree):
    """Remove top-level call Transform(...) so Fourier part is skipped."""
    kept_nodes = []
    for node in tree.body:
        remove_node = False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "Transform":
                remove_node = True
        if not remove_node:
            kept_nodes.append(node)
    tree.body = kept_nodes
    ast.fix_missing_locations(tree)
    return tree


def setup_headless_mode():
    """Disable interactive plotting so test runs complete quickly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None
    plt.pause = lambda *args, **kwargs: None


def run_target_with_overrides():
    if not VISUAL_TEST:
        setup_headless_mode()

    source = TARGET_SCRIPT.read_text(encoding="utf-8")
    patched_tree = apply_top_level_overrides(source, PARAM_OVERRIDES)
    if SKIP_FOURIER:
        patched_tree = remove_fourier_run_call(patched_tree)
    code = compile(patched_tree, str(TARGET_SCRIPT), "exec")

    runtime_globals = {
        "__name__": "__main__",
        "__file__": str(TARGET_SCRIPT),
    }

    print("Running notebook_testing_no_comment.py with overrides:")
    for key, value in PARAM_OVERRIDES.items():
        print(f"  {key} = {value}")
    print(f"Visual test mode: {'ON' if VISUAL_TEST else 'OFF'}")
    if not VISUAL_TEST:
        print("Headless mode: ON (show/pause disabled)")
    if SKIP_FOURIER:
        print("Fourier run: SKIPPED")

    exec(code, runtime_globals, runtime_globals)


if __name__ == "__main__":
    run_target_with_overrides()
