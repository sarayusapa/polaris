# Shim: neutralize PyTorch's bundled Triton, whose LLVM static initializers collide with
# Isaac Sim's LLVM and segfault Kit at startup. PolaRiS eval never uses triton/torch.compile;
# torch treats this ImportError as "triton unavailable" and disables inductor.
raise ImportError("triton disabled for Isaac Sim compatibility (PolaRiS)")
