<!-- GSD:project-start source:PROJECT.md -->
## Project

**Jagruk CV**

Jagruk CV is a production-grade Computer Vision (CV) microservice designed to analyze disaster-related images and estimate structural damage severity. It employs a hybrid architecture combining deep learning for perception with classical computer vision for explainable feature extraction, ensuring that every assessment is reliable, auditable, and trust-aware.

**Core Value:** Provide reliable, explainable, and trust-aware damage assessments that integrate seamlessly into larger decision-making pipelines.

### Constraints

- **Input Mode**: **Single image only** (post-disaster) to support ad-hoc adquisition.
- **Framework**: **PyTorch** for deep learning; **OpenCV** for classical computer vision.
- **Architecture**: **Hybrid** (DL Perception + Classical Features + Policy Scoring).
- **Explainability**: Features must be derived from **explicit algorithms**, not internal neural network states.
- **Deployment**: Initial target is local Python environment, followed by Docker containment.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
| Component | Choice | Rationale | Confidence |
|-----------|---------|-----------|------------|
| **Deep Learning Framework** | PyTorch 2.5+ | Industry standard for vision; excellent support for transfer learning and model transparency. | High |
| **CV Library** | OpenCV 4.10+ | Essential for deterministic feature extraction (Canny, Hough, Laplacian). | High |
| **API Layer** | FastAPI | High-performance, asynchronous handling of image uploads and metadata validation. | High |
| **Model Architecture** | EfficientNet-B0 | Optimal balance between parameter efficiency and accuracy for structural features. | Medium |
| **Preprocessing** | Albumentations | Standard for robust vision pipelines; ensures consistent image loading and normalization. | High |
| **Inference Format** | TorchScript / ONNX | Ensures the service is portable and can be optimized for CPU/GPU deployment later. | Medium |
## Standards & Protocols
- **Data Loading**: PIL for safe opening; conversion to NumPy for OpenCV; Tensor for PyTorch.
- **Serialization**: Pydantic for JSON schema enforcement.
## What NOT to use
- **TensorFlow/Keras**: While stable, PyTorch offers better flexibility for the "Hybrid" DL+CV approach requested.
- **Custom-built HTTP servers**: Use FastAPI to avoid re-implementing security and validation logic.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
