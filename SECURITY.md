# Security policy

Pixel4DGS is an open-source alpha and has no supported stable version yet. It
must not be treated as a hardened service or used with untrusted files.

Once the repository is hosted on GitHub, report a suspected vulnerability
through its private vulnerability-reporting or Security Advisory interface.
Do not include credentials, private datasets, unpublished weights, or exploit
details in a public issue. No direct security address or response-time
commitment is asserted until responsible maintainers publish one.

Security-relevant areas include:

- path containment and symbolic-link handling for scene and asset inputs;
- hash verification and append-only artifact publication;
- checkpoint deserialization and the separation from pickle-free assets;
- external model-weight and native-extension provenance;
- command execution in build/runtime setup;
- denial-of-service risks from adversarial shapes, counts, or file sizes.

The current code rejects untrusted pickle checkpoints at the distributable
asset boundary, but local training checkpoints remain a trusted-workspace
mechanism. Model quality failures, unsupported hardware, and ordinary numerical
instability are engineering bugs rather than security vulnerabilities unless
they cross a trust boundary or enable unintended access or code execution.
