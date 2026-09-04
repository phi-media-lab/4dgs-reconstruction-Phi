# Third-party build patches

These patches alter packaging or build identity only. They do not change the
mathematical implementation of a dependency.

`amd-gsplat-b01acd43-build-identity.patch` fixes an upstream predicate that
tests the `is_git_repo` function object instead of calling it. It also accepts
the full, externally verified source revision through
`AMD_GSPLAT_BUILD_REVISION`, so a wheel made from a hash-verified `git archive`
has an unambiguous PEP 440 local version. The build must set the variable to
`b01acd43e3c7fa942f95fda0974e9125e4de7395`.

`amd-gsplat-b01acd43-glm-include.patch` requires a pinned external GLM tree via
`AMD_GSPLAT_GLM_INCLUDE` and adds it to the ROCm extension's include search
path. The external location is intentional: placing GLM below the gsplat CUDA
source tree makes PyTorch HIPify rewrite GLM's internal include paths and the
build fails. The ROCm branch at the pinned revision otherwise omits GLM and
fails on `glm/gtc/type_ptr.hpp`. The variable must name GLM gitlink revision
`33b4a621a697a305bc3a7610d290677b96beb181`.
