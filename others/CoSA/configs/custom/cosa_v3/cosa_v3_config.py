"""
CoSA v3 (Contextual Spatial Attention v3) Configuration
Our enhanced CoSA model with multi-scale and learnable residual gate.

Model: SiameseUNetCoSA
- Architecture: Siamese U-Net with CoSA blocks
- Multi-scale: Enabled (use_multiscale=True)
- Learnable Gate: Enabled (use_learnable_gate=True)
- TopK: 32
"""

# Model configuration
model = dict(
    type='SiameseUNetCoSA',
    in_channels=3,
    n_classes=1,
    base_channels=64,
    fusion='diff',  # Absolute difference fusion
    topk=32,
    use_multiscale=True,  # Multi-scale feature extraction
    use_learnable_gate=True  # Learnable residual gate
)

# Training configuration
train_cfg = dict(
    batch_size=8,
    epochs=100,
    learning_rate=0.0001,
    optimizer='AdamW',
    weight_decay=1e-5,
    loss='BCEWithLogitsLoss',
    pos_weight=1.0,
    seed=42
)

# Dataset configuration
dataset = dict(
    name='LEVIR-CD',
    root_dir='datasets/to_check_dataset/LEVIR-CD_combined',
    train_size=445,
    val_size=64,
    test_size=128,
    base_size=256,
    augment=True
)

# CoSA v3 specific features
cosa_v3_features = dict(
    multi_scale=True,  # Multi-scale contextual attention
    learnable_gate=True,  # Learnable residual gating mechanism
    topk=32,  # Top-K spatial attention
    residual_connection=True  # Residual connections in CoSA blocks
)

