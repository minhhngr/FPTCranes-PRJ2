# Technical Alignment

The implementation follows the supplied overall, segmentation, and prediction diagrams. The detailed technical design is used to resolve implementation details such as: 198-dimension family-balanced segmentation, March-2026 locked test when it is the latest available period, five expanding temporal validation folds, bounded Random Forest tuning, complete-pipeline serialization, metadata-driven Streamlit controls, validation queue, and empirical error-band communication.

Where a diagram is intentionally high-level (for example its compact "12 inputs" label), the code keeps the richer feature-family context from the detailed specification while preserving the same branch order and excluding salary from clustering.
