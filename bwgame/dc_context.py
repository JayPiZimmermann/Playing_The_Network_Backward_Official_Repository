"""
Full-lifecycle DC decomposition context manager.

Provides a single entry point that handles the entire DC workflow:
model preparation, patching, forward pass, and cleanup.

Supports both last-layer and intermediate-layer output_layer selection.
When an intermediate layer is chosen, layers after it run in normal
(non-DC) mode, treating the rest of the model + loss as a single
differentiable function on the reconstructed output.

Usage:
    from bwgame import dc_decomposition

    model = MyModel()
    x = torch.randn(1, 3, 32, 32)

    with dc_decomposition(model, x) as dc:
        out = dc.output
        loss = criterion(out, target)
        loss.backward()

    grad = dc.reconstruct_gradient()
    sens = dc.sensitivities
"""

from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .operations.base import (
    InputMode, init_catted, reconstruct_output,
    Sensitivities, extract_sensitivities,
    DC_ENABLED,
    adf_backprop_context,
)
from .patcher import (
    patch_model, unpatch_model, cleanup_dc_attrs,
    find_output_layer, mark_output_layer,
    set_dc_enabled, set_adf_forward, set_sensitivity_alpha,
)
from .functional_replacer import replace_functional_with_modules
from .operations.add import Add


class DCResult:
    """
    Result of a DC decomposition forward pass.

    Provides access to the reconstructed output and, after backward,
    the four DC sensitivities.
    """

    def __init__(self, x_cat: Tensor, out_cat: Optional[Tensor], output: Tensor):
        self._x_cat = x_cat
        self._out_cat = out_cat
        self._output = output

    @property
    def output(self) -> Tensor:
        """The reconstructed output (pos - neg)."""
        return self._output

    @property
    def output_4(self) -> Tensor:
        """The raw [4*batch] output tensor (only available when output_layer is the last layer)."""
        if self._out_cat is None:
            raise RuntimeError(
                "output_4 is not available when output_layer is an intermediate layer. "
                "Use .output for the reconstructed [batch] output."
            )
        return self._out_cat

    @property
    def input_4(self) -> Tensor:
        """The [4*batch] input tensor (for accessing .grad after backward)."""
        return self._x_cat

    @property
    def sensitivities(self) -> Sensitivities:
        """
        The 4 DC sensitivities after backward.

        Returns:
            Sensitivities with delta_pp, delta_np, delta_pn, delta_nn
        """
        if self._x_cat.grad is None:
            raise RuntimeError("Call backward() first")
        return extract_sensitivities(self._x_cat.grad)

    def reconstruct_gradient(self) -> Tensor:
        """Reconstruct standard gradient from sensitivities: pp - np - pn + nn."""
        return self.sensitivities.reconstruct_gradient()


def _is_add_operation(node, traced_model) -> bool:
    """Check if a graph node represents an addition operation."""
    import operator

    if node.op == 'call_module':
        try:
            target_module = traced_model
            for attr in node.target.split('.'):
                target_module = getattr(target_module, attr)
            if isinstance(target_module, Add):
                return True
        except (AttributeError, TypeError):
            pass

    if node.op == 'call_function':
        if node.target is operator.add or node.target is torch.add:
            return True

    return False


def _validate_not_on_residual_branch(
    model: nn.Module,
    output_layer: nn.Module,
) -> None:
    """
    Validate that the output layer is not on a residual branch.

    A layer is "on a residual branch" if its output feeds into an Add
    where the other input bypasses it (a skip connection). This would
    cause a [4*batch] vs [batch] shape mismatch at the Add.

    Residual connections entirely AFTER the output layer (in the normal
    zone) are allowed -- only skip connections that cross the DC/normal
    boundary are problematic.

    Raises:
        ValueError: If the output layer is on a residual branch.
    """
    from torch.fx import symbolic_trace

    try:
        traced = symbolic_trace(model)
    except Exception:
        return

    # Find the output_layer node in the graph
    output_layer_name = None
    for name, module in model.named_modules():
        if module is output_layer:
            output_layer_name = name
            break

    if output_layer_name is None:
        return

    output_layer_node = None
    for node in traced.graph.nodes:
        if node.op == 'call_module' and node.target == output_layer_name:
            output_layer_node = node

    if output_layer_node is None:
        return

    # BFS forward from the output_layer node to find all reachable nodes
    reachable_from_output = set()
    queue = [output_layer_node]
    while queue:
        current = queue.pop(0)
        if current in reachable_from_output:
            continue
        reachable_from_output.add(current)
        for user in current.users.keys():
            queue.append(user)

    # Check each reachable Add: if any input is NOT reachable from the
    # output_layer, then a skip connection crosses the boundary
    for node in reachable_from_output:
        if _is_add_operation(node, traced):
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and arg not in reachable_from_output:
                    node_desc = node.target if isinstance(node.target, str) else node.name
                    raise ValueError(
                        f"DC output layer detection failed: The output layer "
                        f"'{output_layer_name}' "
                        f"({output_layer.__class__.__name__}) is on a residual "
                        f"branch -- a skip connection bypasses it and feeds "
                        f"into Add '{node_desc}', which would cause a "
                        f"[4*batch] vs [batch] shape mismatch.\n"
                        f"\n"
                        f"Choose an output_layer that is not on a residual "
                        f"branch, or one that comes after all residual "
                        f"connections."
                    )


def _get_module_execution_order(
    model: nn.Module,
    x: Tensor,
) -> List[Tuple[str, nn.Module]]:
    """
    Run a dummy forward pass and return modules in execution order.

    Returns:
        List of (name, module) tuples in execution order.
    """
    execution_order = []
    hooks = []

    for name, module in model.named_modules():
        if name:
            def make_hook(n, m):
                def hook(mod, inp, out):
                    execution_order.append((n, m))
                return hook
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()

    return execution_order


def _configure_dc_boundary(
    model: nn.Module,
    output_layer: nn.Module,
    x: Tensor,
) -> Optional[torch.utils.hooks.RemovableHandle]:
    """
    Configure the DC/normal boundary at the output_layer.

    After patch_model has set DC_ENABLED=True on all modules, this function:
    1. Validates the output_layer is not on a residual branch
    2. Determines execution order
    3. Enables DC only for modules at/before the output_layer
    4. Registers a forward hook to convert [4*batch] -> [batch] at the boundary

    Args:
        model: The patched model (all modules have DC_ENABLED=True)
        output_layer: The boundary module
        x: Original input tensor [batch] (NOT x_cat)

    Returns:
        Hook handle if intermediate layer (caller must remove it), or None.
    """
    # Step 1: Validate the output_layer is not on a residual branch
    _validate_not_on_residual_branch(model, output_layer)

    # Step 2: Determine execution order with DC disabled (normal forward)
    set_dc_enabled(model, False)
    execution_order = _get_module_execution_order(model, x)

    # Step 3: Find output_layer position in execution order
    output_layer_idx = None
    for i, (name, module) in enumerate(execution_order):
        if module is output_layer:
            output_layer_idx = i

    if output_layer_idx is None:
        # output_layer not found -- enable all DC and skip boundary setup
        set_dc_enabled(model, True)
        return None

    is_last = (output_layer_idx == len(execution_order) - 1)

    # Step 4: Enable DC only for modules at/before the output_layer
    executed_ids = set()
    for i, (name, module) in enumerate(execution_order):
        if hasattr(module, DC_ENABLED):
            setattr(module, DC_ENABLED, i <= output_layer_idx)
        executed_ids.add(id(module))

    # Modules not reached during dummy forward: leave DC disabled
    for name, module in model.named_modules():
        if hasattr(module, DC_ENABLED) and id(module) not in executed_ids:
            setattr(module, DC_ENABLED, False)

    if is_last:
        return None

    # Step 5: Register boundary hook on output_layer
    def boundary_hook(module, input, output):
        return reconstruct_output(output)

    return output_layer.register_forward_hook(boundary_hook)


def _disable_inplace(model: nn.Module) -> None:
    """Disable inplace operations on all modules.

    Inplace ops (e.g. ``nn.ReLU(inplace=True)``) can corrupt saved tensors
    needed for backward in the [4*batch] DC format.
    """
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False


# Activation types that cannot be DC-decomposed (only ReLU is supported)
_INCOMPATIBLE_ACTIVATIONS = (nn.GELU, nn.SiLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid)

# Activations that are never DC-decomposable (not even with native splitting)
_UNSUPPORTED_ACTIVATIONS = (nn.ReLU6, nn.Hardswish, nn.Hardsigmoid)


def _replace_non_relu_activations(
    model: nn.Module,
    _saved: Optional[List] = None,
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Replace non-ReLU activations with ``nn.ReLU()`` for DC compatibility.

    DC decomposition only supports ReLU (piecewise-linear with clear
    pos/neg decomposition).  Smooth activations like GELU pass through
    undecomposed, producing broken sensitivity maps.  This function
    replaces them with ReLU so the full network is DC-decomposable.

    Returns a list of ``(parent, attr_name, original_module)`` tuples
    that can be passed to :func:`_restore_activations` to undo the
    replacement.
    """
    if _saved is None:
        _saved = []
    for name, module in model.named_children():
        if isinstance(module, _INCOMPATIBLE_ACTIVATIONS):
            _saved.append((model, name, module))
            setattr(model, name, nn.ReLU())
        else:
            _replace_non_relu_activations(module, _saved)
    return _saved


def _replace_unsupported_activations(
    model: nn.Module,
    _saved: Optional[List] = None,
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Replace only truly unsupported activations (not GELU/SiLU) with ReLU.

    GELU and SiLU are kept because they have native DC operations.
    Only ReLU6, Hardswish, Hardsigmoid are replaced.
    """
    if _saved is None:
        _saved = []
    for name, module in model.named_children():
        if isinstance(module, _UNSUPPORTED_ACTIVATIONS):
            _saved.append((model, name, module))
            setattr(model, name, nn.ReLU())
        else:
            _replace_unsupported_activations(module, _saved)
    return _saved


def _restore_activations(saved: List[Tuple[nn.Module, str, nn.Module]]) -> None:
    """Restore original activation modules saved by :func:`_replace_non_relu_activations`."""
    for parent, attr_name, original in saved:
        setattr(parent, attr_name, original)


def run_adf_forward(model, x, input_variance=1.0):
    """Run differentiable ADF forward through patched model.

    Input is formatted as [mean; var] (2*batch along dim 0).
    Returns (input_2, output_mean, output_var) where input_2.requires_grad=True.
    Side effect: populates layer_data.adf_mean/adf_variance on activation modules.

    Args:
        model: Already-patched model.
        x: Input tensor [batch, ...].
        input_variance: Scalar initial variance for all input elements.

    Returns:
        (input_2, output_mean, output_var) tuple. input_2 has requires_grad=True.
    """
    from .game_theory.game_context import DCLayerData, DC_LAYER_DATA, _is_cacheable

    # Initialize layer_data on cacheable modules
    for _, m in model.named_modules():
        if _is_cacheable(m):
            if not hasattr(m, DC_LAYER_DATA):
                setattr(m, DC_LAYER_DATA, DCLayerData())

    set_adf_forward(model, True)
    try:
        B = x.shape[0]
        mean = x.clone()
        var = torch.full_like(x, input_variance)
        input_2 = torch.cat([mean, var], dim=0).requires_grad_(True)

        output_2 = model(input_2)

        output_mean = output_2[:B]
        output_var = output_2[B:]
        return input_2, output_mean, output_var
    finally:
        set_adf_forward(model, False)


@contextmanager
def dc_decomposition(
    model: nn.Module,
    x: Tensor,
    output_layer: Optional[nn.Module] = None,
    mode: InputMode = InputMode.CENTER,
    alpha: float = 0.0,
    alpha_mode: str = 'constant',
    layer_alphas: Optional[Dict[str, float]] = None,
    relu_lambda: float = 1.0,
    backprop_mode: str = 'sum',
    align_forward: bool = False,
    align_backward: bool = False,
    target_grad: Optional[Tensor] = None,
    replace_activations: bool = False,
    store_activations: bool = False,
    adf_mode: str = 'none',
    adf_lambda: float = 0.0,
    adf_input_variance: float = 1.0,
    adf_probabilistic: bool = True,
    adf_propagation: str = 'analytic',
    entropy_bias_adapt: float = 0.0,
):
    """
    Full-lifecycle DC decomposition context manager.

    Handles the entire workflow: replaces functional operations with modules,
    finds and validates the output layer, patches the model for DC,
    runs the forward pass, and unpatches on exit.

    When output_layer is an intermediate layer, only layers up to and
    including it run in DC mode. Layers after it (and the user's loss)
    run in normal mode on the reconstructed output.

    Args:
        model: A standard PyTorch model (will be modified in-place).
        x: Input tensor.
        output_layer: Manually specify the output layer. If None, auto-detected.
            The output layer is the boundary: everything before it is split into
            pos/neg DC streams, everything after (like the loss) receives the
            reconstructed pos-neg output.
        mode: Input splitting mode (default: CENTER).
        alpha: Sensitivity shift for numerical stability (default: 0.0 = disabled).
            Typical value for models with residual connections: 0.5.
        alpha_mode: How to determine alpha per module:
            'constant': Use the same alpha for all modules (default).
            'frobenius': Compute alpha from Frobenius norm of weights.
            'per_layer': Use layer_alphas dict for per-module overrides.
        layer_alphas: Dict mapping module names to alpha values (for 'per_layer' mode).
        relu_lambda: Continuous ReLU split parameter in [0, 1].
            1.0 = max mode, 0.0 = min mode, intermediate = weighted mix.
        backprop_mode: ReLU backprop mode ('dc', 'sum', 'mask_diff', 'stopping').
        align_forward: If True, align DC forward to match original activations.
        align_backward: If True, align DC backward to match original gradients.
            Requires target_grad.
        target_grad: Target gradient for backward alignment capture.
            Required if align_backward=True.
        replace_activations: If True, replace GELU/SiLU by ReLU during the
            decomposition (default: keep them; they have native DC operations).
        store_activations: If True, run a capture pass to store original
            activations on modules (accessible as ``ctx._dc_layer_data.activation``
            in backward patches).
        adf_mode: 'none' (standard DC) or 'adf' (risk-sensitive backward).
        adf_lambda: Risk parameter λ (confidence bound μ + λσ of the gate).
        adf_input_variance: Initial variance for ADF propagation.
        adf_probabilistic: Use smooth Φ masks (True) or hard threshold (False).
        adf_propagation: How variance is propagated through the network.
            'analytic': Jacobian-based [2*batch] moment matching (default).
            'delta_method': Like analytic but preserves variance through
                LayerNorm instead of resetting to γ².
        entropy_bias_adapt: Fan-in entropy bias λ_ent (default: 0.0 = off).
            Subtracts λ_ent·H̃_j from each neuron's bias (game reward modification).

    Yields:
        DCResult with .output, .sensitivities, .reconstruct_gradient()

    Example:
        with dc_decomposition(model, x, alpha=0.5) as dc:
            out = dc.output
            loss = criterion(out, target)
            loss.backward()

        grad = dc.reconstruct_gradient()
        sens = dc.sensitivities

    Example with the ADF (risk-averse) backward:
        with dc_decomposition(model, x, adf_mode='adf', adf_lambda=-1.0) as dc:
            out = dc.output
            loss = criterion(out, target)
            loss.backward()
    """
    # Step 0: Fan-in entropy bias b_j -= λ_ent·H̃_j (biases restored on exit).
    _game_contexts = []
    if entropy_bias_adapt > 0:
        from .game_theory.entropy import entropy_bias_adapt_context
        ctx = entropy_bias_adapt_context(model, entropy_bias_adapt)
        ctx.__enter__()
        _game_contexts.append(ctx)

    if align_backward and target_grad is None:
        raise ValueError("target_grad is required when align_backward=True")

    # Step 1: Disable inplace ops (corrupt saved tensors in DC [4*batch] format)
    _disable_inplace(model)

    # Step 1b: Handle non-ReLU activations
    if replace_activations:
        # Replace ALL non-ReLU activations (GELU, SiLU, etc.) with ReLU
        saved_activations = _replace_non_relu_activations(model)
    else:
        # Keep GELU/SiLU (they have native DC operations), only replace
        # truly unsupported activations (ReLU6, Hardswish, Hardsigmoid)
        saved_activations = _replace_unsupported_activations(model)

    # Step 2: Replace functional calls (torch.relu, +, etc.) with modules
    replace_functional_with_modules(model, inplace=True)

    # Step 3: Find output layer
    if output_layer is None:
        output_layer = find_output_layer(model)

    # Step 3b: Alignment capture (before DC patching, after functional replacement)
    alignment_cache = None
    if align_forward or align_backward:
        from .alignment_cache import AlignmentCache, AlignmentMode
        if align_forward and align_backward:
            a_mode = AlignmentMode.BOTH
        elif align_forward:
            a_mode = AlignmentMode.FORWARD_ONLY
        else:
            a_mode = AlignmentMode.BACKWARD_ONLY
        alignment_cache = AlignmentCache(mode=a_mode)
        alignment_cache.capture_original(
            model, x, target_grad,
            capture_forward=align_forward,
            capture_backward=align_backward,
        )
        alignment_cache.attach_to_model(model)

    # Variables referenced by the pre-yield cleanup helper below.  These
    # are assigned inside the upcoming ``try`` block but must already exist
    # if an exception fires before the assignment.
    _has_layer_data = False
    boundary_hook = None
    _yield_reached = False

    def _pre_yield_cleanup():
        """Unpatch everything installed so far, on any pre-yield failure.

        An exception raised before the ``yield`` (e.g. a CUDA OOM inside
        ``run_adf_forward`` or ``run_capture_pass``) leaves the ``with``
        statement without its ``__exit__``; this helper restores the
        model in that case.
        """
        if boundary_hook is not None:
            try:
                boundary_hook.remove()
            except Exception:
                pass
        if alignment_cache is not None:
            try:
                alignment_cache.detach_from_model(model)
            except Exception:
                pass
        if _has_layer_data:
            try:
                from .game_theory.game_context import cleanup_layer_data
                cleanup_layer_data(model)
            except Exception:
                pass
        try:
            unpatch_model(model)
        except Exception:
            pass
        try:
            cleanup_dc_attrs(model)
        except Exception:
            pass
        try:
            _restore_activations(saved_activations)
        except Exception:
            pass
        for _gc in reversed(_game_contexts):
            try:
                _gc.__exit__(None, None, None)
            except Exception:
                pass

    # Step 4: Patch model for DC decomposition (all modules get DC_ENABLED=True)
    #
    # Everything from here through the first successful ``model(x_cat)``
    # is wrapped in a single try/except so that an OOM (or any other
    # failure) in ``run_adf_forward`` / ``run_capture_pass`` / the main
    # forward itself unpatches the model before the exception bubbles up.
    # See ``_pre_yield_cleanup``.
    try:
        patch_model(model, relu_lambda=relu_lambda, backprop_mode=backprop_mode)
        if output_layer is not None:
            mark_output_layer(output_layer)

        # Step 4b: ADF forward pass and/or capture pass.
        # With ADF mode active, a variance propagation forward pass
        # ('analytic', or 'delta_method', which preserves the variance
        # through LayerNorm) populates layer_data.adf_mean/adf_variance on
        # the activation modules.
        if adf_mode != 'none':
            from .operations.base import adf_propagation_context
            with adf_propagation_context(adf_propagation):
                run_adf_forward(model, x, input_variance=adf_input_variance)
            _has_layer_data = True
            # The capture pass caches the original pre-activation values: the
            # backward gate uses them (not the ADF moment-matched means) as
            # the mean.  run_capture_pass reinitialises DCLayerData, so the
            # ADF stats just populated are saved and restored.
            from .game_theory.game_context import run_capture_pass, DC_LAYER_DATA
            _saved_adf = {}
            for _n, _m in model.named_modules():
                ld = getattr(_m, DC_LAYER_DATA, None)
                if ld is not None and ld.adf_variance is not None:
                    _saved_adf[id(_m)] = (ld.adf_mean, ld.adf_variance,
                                           ld._adf_stack)
            run_capture_pass(model, x, capture_activations=True)
            # Restore ADF stats onto the new DCLayerData objects
            for _n, _m in model.named_modules():
                ld = getattr(_m, DC_LAYER_DATA, None)
                saved = _saved_adf.get(id(_m))
                if ld is not None and saved is not None:
                    ld.adf_mean, ld.adf_variance, ld._adf_stack = saved
        elif store_activations:
            from .game_theory.game_context import run_capture_pass
            run_capture_pass(model, x, capture_activations=True)
            _has_layer_data = True

        # Step 5: Set sensitivity alpha (always set, even for alpha=0, to clear
        # any stale values from previous runs on the same model instance)
        set_sensitivity_alpha(model, alpha=alpha, mode=alpha_mode,
                              layer_alphas=layer_alphas)

        # Step 6: Configure DC boundary (validates, sets per-module DC, adds hook)
        if output_layer is not None:
            boundary_hook = _configure_dc_boundary(model, output_layer, x)

        # Step 7: DC forward pass
        with adf_backprop_context(adf_mode, adf_lambda, adf_probabilistic):
            x_cat = init_catted(x, mode)
            out_raw = model(x_cat)

            if boundary_hook is not None:
                # Intermediate output_layer: out_raw is already [batch] from hook
                output = out_raw
                out_cat = None
            else:
                # Last layer: out_raw is [4*batch], need explicit reconstruct
                out_cat = out_raw
                output = reconstruct_output(out_cat)

            result = DCResult(x_cat, out_cat, output)

            _yield_reached = True
            try:
                yield result
            finally:
                if boundary_hook is not None:
                    boundary_hook.remove()
                if alignment_cache is not None:
                    alignment_cache.detach_from_model(model)
                if _has_layer_data:
                    from .game_theory.game_context import cleanup_layer_data
                    cleanup_layer_data(model)
                unpatch_model(model)
                cleanup_dc_attrs(model)
                _restore_activations(saved_activations)
                for _gc in reversed(_game_contexts):
                    _gc.__exit__(None, None, None)
    except Exception:
        # Pre-yield failure (patch_model / capture / ADF / forward
        # raised before we handed control to the user).  Run the full
        # cleanup path so the next attribution call sees an unpatched
        # model.  If the exception comes from INSIDE the user's
        # ``with dc_decomposition(...)`` body, the post-yield ``finally``
        # above has already cleaned up — ``_yield_reached`` tells us to
        # skip cleanup here and just re-raise.
        if not _yield_reached:
            _pre_yield_cleanup()
        raise
