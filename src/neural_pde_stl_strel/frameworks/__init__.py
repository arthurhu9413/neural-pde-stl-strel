"""Framework integration demos.

This subpackage contains thin wrappers and hello-world scripts for external
physics-informed ML frameworks evaluated during this project:

- **Neuromancer** (PNNL): differentiable constrained optimization.
- **PhysicsNeMo** (NVIDIA): GPU-accelerated PINN training.
- **TorchPhysics** (Bosch): PyTorch-native PINN library.
- **SpaTiaL** (KTH): spatial specification language.

These modules are *demos*, not production adapters. They are useful for
verifying that a framework is installed correctly and for exploring how
STL penalties might be injected into each framework's training loop.

All external dependencies are optional; the modules guard their imports
with try/except blocks.
"""
