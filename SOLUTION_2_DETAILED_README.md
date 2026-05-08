# SpaceNet 8 - Solution 2 (number13) - Comprehensive Technical Documentation

## Table of Contents
1. [Solution Overview](#solution-overview)
2. [Architecture Design](#architecture-design)
3. [Data Flow Pipeline](#data-flow-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Strategy](#training-strategy)
6. [Loss Functions](#loss-functions)
7. [Inference Pipeline](#inference-pipeline)
8. [Code Structure](#code-structure)
9. [Key Features & Innovations](#key-features--innovations)

---

## Solution Overview

**Rank:** 2nd Place Winner  
**Approach:** Two-stage transfer learning with pre-training on SpaceNet 2, 3, 5 datasets, followed by finetuning on SpaceNet 8

### High-Level Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOLUTION 2 PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: FOUNDATION FEATURES (Buildings & Roads)              │
│  ├─ Pretrain on SpaceNet-2, 3, 5 data                          │
│  ├─ Model: UNet with ResNet-50 encoder                         │
│  ├─ Outputs: Building mask & Road mask predictions             │
│  └─ Loss: MixedLoss (Focal + Dice)                             │
│                                                                  │
│  Stage 2: FLOOD FEATURES (Change Detection)                    │
│  ├─ Input: Pre-event + Post-event image pairs                  │
│  ├─ Model: Siamese UNet with multi-head decoder                │
│  ├─ Outputs: Flooded buildings & Flooded roads                 │
│  └─ Loss: MixedLoss (Seg) + BCE (Classification)               │
│                                                                  │
│  Stage 3: POST-PROCESSING & SUBMISSION                         │
│  ├─ Flood attribution heuristic                                │
│  ├─ Flood label merging                                        │
│  └─ CSV generation for submission                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Design

### 1. **Foundation Features Model (UnetMhead)**

**Purpose:** Detect buildings and roads from pre-event images

**Architecture:**
```
Input Image (3-channel RGB, 512x512)
        ↓
   [Encoder: ResNet-50]
   └─ Extracts multi-scale features at 5 levels
        ↓
   [UNet Decoder]
   └─ Progressively upsamples features
        ↓
   ┌─────────────────┐
   │  Two Heads      │
   ├─────────────────┤
   │ Building Head   │ → Building predictions (1 channel)
   │ Road Head       │ → Road predictions (1 channel)
   └─────────────────┘
```

**Key Components:**
- **Encoder:** ResNet-50 with ImageNet weights (optional)
- **Decoder Channels:** [256, 128, 64, 32, 16] (progressively decreasing)
- **Two Segmentation Heads:** Independent prediction heads for buildings and roads
- **Input:** Single pre-event image
- **Output:** Two probability maps (buildings, roads)

**Code Location:** `models.py` - Class `UnetMhead`

### 2. **Siamese Multi-Head Model (UnetSiamese_Mhead)**

**Purpose:** Detect flooded buildings and roads using change detection

**Architecture:**
```
Pre-event Image                    Post-event Image
        ↓                                 ↓
   [Shared Encoder: ResNet-50]    [Shared Encoder: ResNet-50]
        ↓                                 ↓
   Enc_1 features                   Enc_2 features
        ├─────────────────────────────────┤
        │     Feature Fusion (6 levels)   │
        │  Concatenation + GroupConv1x1  │
        ↓
   Fused Features → UNet Decoder
        ↓
   ┌──────────────────────────┐
   │    Three Heads           │
   ├──────────────────────────┤
   │ Building Segmentation    │ → Flooded buildings
   │ Road Segmentation        │ → Flooded roads
   │ Classification Head      │ → Flooded/Not-Flooded label
   └──────────────────────────┘
```

**Fusion Strategies:**
- **'cat'**: Concatenate features from both timesteps + 1x1 grouped convolution
- **'add'**: Element-wise addition
- **'cat_add'**: Hybrid of concatenation and addition

**Code Location:** `models.py` - Class `UnetSiamese_Mhead`

---

## Data Flow Pipeline

### Stage 1: Foundation Features Training

```
INPUT
  └─ CSV file with paths to: preimg, building_mask, road_mask
     
DATALOADERS (dataloaders.py - SN8Dataset)
  ├─ Load pre-event images (RGB .tif)
  ├─ Load building masks (binary PNG)
  ├─ Load road masks with speed classes (8 classes)
  └─ Apply augmentations:
     ├─ ShiftScaleRotate
     ├─ RandomSizedCrop
     ├─ Rotation (±10°)
     ├─ Flips (Horizontal, Vertical, Transpose)
     └─ Color augmentations (RGB Shift, Brightness, Gamma, ColorJitter)

TRAINING LOOP (train.py / trainer_ms.py - TrainEpoch)
  ├─ Forward: model(preimg) → buildings_pred, roads_pred
  ├─ Losses:
  │   ├─ loss_buildings = MixedLoss(buildings_pred, building_mask)
  │   └─ loss_roads = MixedLoss(roads_pred, road_mask)
  ├─ Total Loss = loss_buildings + loss_roads
  ├─ Backward & Optimize
  └─ Metrics: IoU at threshold 0.5

VALIDATION
  └─ Same forward pass, accumulate metrics

OUTPUT
  └─ Best model checkpoint saved based on IoU
```

### Stage 2: Flood Features Training

```
INPUT
  └─ CSV file with: preimg, postimg, flood_mask

DATALOADERS (SN8Dataset with flood data)
  ├─ Load pre-event images
  ├─ Load post-event images
  ├─ Load flood masks (binary - flooded/non-flooded)
  ├─ Generate class labels:
  │   ├─ If sum(flood_mask) > 0 → label = 1 (flooded)
  │   └─ Else → label = 0 (non-flooded)
  └─ Apply same augmentations to image pairs + masks

TRAINING LOOP (trainer_ms.py - TrainEpochSiamese)
  ├─ Load foundation model weights (pre-trained)
  ├─ Forward: model(preimg, postimg) 
  │           → seg_buildings, seg_roads, classification
  ├─ Losses:
  │   ├─ seg_loss = MixedLoss(seg_b, flood_mask_b) 
  │   │           + MixedLoss(seg_r, flood_mask_r)
  │   └─ classification_loss = BCE(classification, label)
  ├─ Total Loss = 0.75 * seg_loss + 0.25 * classification_loss
  ├─ Backward & Optimize (with AMP - Automatic Mixed Precision)
  └─ Metrics: IoU on segmentation outputs

VALIDATION
  └─ Same forward pass, accumulate metrics

OUTPUT
  └─ Best model + periodic checkpoints (every 30 epochs)
```

---

## Model Architecture

### Key Model Variants

#### **1. UnetMhead (Two-head U-Net)**
- **Purpose:** Foundation features detection
- **Encoder:** Configurable (ResNet-34/50/101, SeResNet, Inception, EfficientNet)
- **Decoder:** Standard UNet decoder with skip connections
- **Output Heads:** 2 independent heads for buildings and roads
- **Use Case:** Stage 1 training on SpaceNet 2, 3, 5 data

**Configuration:**
```python
encoder_name: str = "resnet50"
encoder_depth: int = 5
decoder_channels: List[int] = (256, 128, 64, 32, 16)
in_channels: int = 3
classes: int = 1  # binary segmentation per head
```

#### **2. UnetSiamese_Mhead (Siamese Multi-Head U-Net)**
- **Purpose:** Flood change detection with multi-output
- **Encoder:** Shared across both timesteps
- **Fusion:** Feature fusion at each encoder level
- **Output Heads:** 3 heads (buildings, roads, classification)
- **Use Case:** Stage 2 training for flood detection

**Key Innovation - Feature Fusion:**
```python
for each encoder level i:
    enc_1[i] → features from pre-event
    enc_2[i] → features from post-event
    
    if fusion_mode == 'cat':
        fused = Concatenate([enc_1[i], enc_2[i]])  # double channels
        fused = GroupConv1x1(fused)  # reduce to original channels
    elif fusion_mode == 'add':
        fused = enc_1[i] + enc_2[i]
    elif fusion_mode == 'cat_add':
        cat_fused = GroupConv1x1(Concatenate([enc_1[i], enc_2[i]]))
        add_fused = enc_1[i] + enc_2[i]
        fused = cat_fused + add_fused
```

#### **3. MAnetMhead & MAnetSiamese_Mhead**
- **Alternative with MANet decoder** (Multi-scale attention)
- **Same concept but with attention mechanisms**
- **Use Case:** Experimental variants (not primary solution)

---

## Training Strategy

### Phase 1: Pre-training (Foundation Features)

**Duration:** 45 epochs  
**Batch Size:** 8  
**Learning Rate:** 0.0001 (decreased at epochs 25, 35)  
**Data Sources:**
- SpaceNet 2 (Buildings dataset)
- SpaceNet 3 (Roads dataset)
- SpaceNet 5 (Roads dataset)
- Off-nadir imagery (additional buildings data)

**Loss Function:** `MixedLoss`
- 75% Focal Loss (focuses on hard negatives)
- 25% Soft Dice Loss

**Optimizer:** Adam with weight decay (0.0001)  
**Scheduler:** MultiStepLR at epochs [25, 35]

### Phase 2: Fine-tuning (SpaceNet 8 Foundation)

**Duration:** 45 epochs  
**Batch Size:** 8  
**Learning Rate:** 0.0001  
**Data:** SpaceNet 8 training data

**Process:**
1. Load weights from Phase 1
2. Train on SpaceNet 8 pre-event images
3. Weights initialize both foundation detection tasks

### Phase 3: Flood Detection Training

**Duration:** 100 epochs (longer for better convergence)  
**Batch Size:** 8  
**Learning Rate:** 0.0001 (decreased every 30 epochs by 0.5×)  
**Data:** SpaceNet 8 flood data

**Loss Weighting:**
- Segmentation Loss (building & road flood): 75%
- Classification Loss (image-level flood label): 25%

**Special Handling:**
- Imbalanced Sampler: To handle class imbalance (fewer flooded samples)
- AMP (Automatic Mixed Precision): For efficiency
- 100 epochs instead of 45: More training for better stability

---

## Loss Functions

### 1. **MixedLoss** (Segmentation)

```python
class MixedLoss(nn.Module):
    def forward(self, input, target):
        # input: raw logits
        # target: binary mask (0 or 1)
        
        loss = α × FocalLoss(input, target) 
             + (1-α) × DiceLoss(input, target)
        
        # Default: α = 0.75
        return loss.mean()
```

**FocalLoss:**
```
FL(pt) = -α(1-pt)^γ log(pt)

pt = sigmoid(input)
γ = 2 (focuses on hard examples)
α = class weight
```

**Soft Dice Loss:**
```
DiceLoss = 1 - (2*TP + smooth) / (2*TP + FP + FN + smooth)

TP, FP, FN computed from sigmoid predictions vs binary target
smooth = 1.0 (prevents division by zero)
```

**Why MixedLoss?**
- Focal Loss handles class imbalance and hard negatives
- Dice Loss directly optimizes IoU metric
- Combined → Better segmentation quality

### 2. **BCE Loss** (Classification)

```python
class BCE(nn.Module):
    def forward(self, input, target):
        # input: raw logits from classification head
        # target: 0 or 1 (flooded/non-flooded image)
        
        loss = BCEWithLogitsLoss(input, target)
        return loss.mean()
```

**Used for:** Image-level flood classification  
**Weight in Total Loss:** 25%

### 3. **Combined Loss** (Flood Detection Training)

```python
Total Loss = 0.75 × SegmentationLoss + 0.25 × ClassificationLoss

SegmentationLoss = MixedLoss(buildings_pred, buildings_mask) 
                 + MixedLoss(roads_pred, roads_mask)
```

---

## Inference Pipeline

### Inference Process

```
Test Image (1300×1300, large)
    ↓
[Patch-based Inference]
├─ Tile image into 512×512 patches with 384 stride
├─ Run model on each patch independently
├─ Collect predictions from all patches
└─ Merge overlapping predictions (averaging)
    ↓
[Post-processing]
├─ Apply sigmoid to convert logits to probabilities
├─ Optionally apply TTA (Test-Time Augmentation)
├─ Threshold predictions (if specified)
└─ Remove padding if added during preprocessing
    ↓
[Output Generation]
├─ Save predicted masks as PNG files
└─ Format for submission CSV
```

### Patch-based Inference Strategy

**Why?** Test images are 1300×1300, model input is 512×512

**Algorithm:**
```python
def predict_patches_foundation(image_array, model, 
                               model_input_shape=(512,512,3), 
                               step_size=384):
    # Calculate number of patches needed
    x_steps = ceil((width - 512) / 384) + 1
    y_steps = ceil((height - 512) / 384) + 1
    
    # Create output arrays
    raw_inference = zeros((x_steps*y_steps, height, width, channels))
    
    # Slide window over image
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # Extract patch, enforce boundaries
            y_end = min(y + 512, height)
            x_end = min(x + 512, width)
            
            patch = image[y:y_end, x:x_end, :]
            
            # Model inference
            pred = model(patch)
            
            # Store in output (handles overlaps via averaging)
            raw_inference[idx, y:y_end, x:x_end] = pred
    
    # Average overlapping regions
    final_pred = nanmean(raw_inference, axis=0)
    return final_pred
```

### Test-Time Augmentation (TTA)

```python
# Original prediction
preds_1 = predict_patches(image)

# Horizontal flip
preds_2 = predict_patches(fliplr(image))
preds_2 = fliplr(preds_2)

# Vertical flip
preds_3 = predict_patches(flipud(image))
preds_3 = flipud(preds_3)

# Average predictions
final_pred = (preds_1 + preds_2 + preds_3) / 3
```

---

## Code Structure

### **1. models.py**
Models for both foundation and flood detection

**Classes:**
- `UnetMhead` - Two-head U-Net for foundation features
- `UnetSiamese` - Basic siamese architecture
- `UnetSiamese_Mhead` - **Main model for flood detection**
- `MAnetMhead` - MANet variant
- `MAnetSiamese_Mhead` - MANet siamese variant

**Key Methods:**
```python
# Foundation model
model = UnetMhead(encoder_name='resnet50')
buildings, roads = model(x_build, x_road=None)

# Flood model
model = UnetSiamese_Mhead(encoder_name='resnet50')
seg_b, seg_r, cls = model(x_pre, x_post)
```

### **2. dataloaders.py**
Data loading and augmentation

**Classes:**
- `SN8Dataset` - Main dataset loader
  - Handles pre/post images
  - Loads multiple masks (buildings, roads, flood)
  - Handles data augmentation
  - Generates flood labels from masks

**Key Functions:**
```python
# Get balanced data generators
gens = get_generators_sn8(config, 
                          data_to_load=['preimg', 'postimg', 'flood'],
                          do_imbalanced=True)

train_loader = gens['training_generator']
val_loader = gens['val_generator']
```

### **3. train.py**
Main training orchestration

**Supports 3 training modes:**
1. `--tr foundation` - Train foundation features
2. `--tr flood` - Train flood detection
3. `--tr previous` - Train on SpaceNet 2/3/5 data

**Key Arguments:**
```bash
python train.py \
    --tr foundation \
    --model_name resnet50 \
    --batch_size 8 \
    --net unet \
    --train_csvs path/to/train.csv \
    --val_csvs path/to/val.csv \
    --weights path/to/pretrained.pth \
    --out_dir ./outputs/ \
    --gpu 0
```

### **4. trainer_ms.py**
Training loop and metrics

**Classes:**
- `TrainEpoch` - Foundation features training
- `ValidEpoch` - Foundation features validation
- `TrainEpochSiamese` - Flood detection training (with AMP)
- `ValidEpochSiamese` - Flood detection validation

**Key Features:**
- Automatic Mixed Precision (AMP) for efficiency
- Per-batch gradient scaling
- Running metrics computation (IoU)

### **5. loss.py**
Loss function implementations

**Main Classes:**
- `BinaryFocalLoss` - Focal loss for binary classification
- `SoftDiceLoss` - Dice coefficient loss
- `FocalAndDice` - Combined focal + dice
- `MixedLoss` - Custom mixed loss for segmentation
- `BCE` - Binary cross-entropy

### **6. inference.py**
Inference and prediction

**Key Functions:**
- `predict_patches_foundation()` - Patch-based inference for foundation
- `predict_patches_flood()` - Patch-based inference for flood
- `test_foundation()` - Full test pipeline for foundation
- `test_flood()` - Full test pipeline for flood
- `FoundationTestDataset` - Test dataset loader
- `FloodTestDataset` - Flood test dataset with pre/post pairing

**Example Usage:**
```bash
# Foundation inference
python inference.py \
    --test_dir_pre /path/to/pre_images \
    --model_path /path/to/foundation_model.pth \
    --pred_dir /output/predictions \
    --model_name resnet50 \
    --tta

# Flood inference
python inference.py \
    --flood \
    --test_dir_pre /path/to/pre_images \
    --test_dir_post /path/to/post_images \
    --mapping_csv /path/to/mapping.csv \
    --model_path /path/to/flood_model.pth \
    --pred_dir /output/predictions \
    --model_name resnet50 \
    --tta
```

---

## Key Features & Innovations

### 1. **Transfer Learning Strategy**
- Pre-train on SpaceNet 2, 3, 5 datasets (3 different tasks)
- Fine-tune on SpaceNet 8 foundation features
- Further fine-tune for flood detection
- **Benefit:** Leverages more labeled data, better initialization

### 2. **Siamese Architecture for Change Detection**
- Shared encoder for pre & post images
- Feature fusion at multiple levels
- Handles temporal information effectively
- **Benefit:** Better captures change between timesteps

### 3. **Multi-head Architecture**
- Separate heads for buildings and roads
- Joint training but independent predictions
- Classification head for image-level labels
- **Benefit:** Flexibility, allows task-specific optimization

### 4. **Patch-based Inference with Overlapping**
- Handles large test images (1300×1300)
- Overlapping patches with averaging
- Smooth boundaries without artifacts
- **Benefit:** Scalable, no information loss

### 5. **Dual Loss Function**
- Segmentation loss (MixedLoss)
- Classification loss (BCE) for image-level labels
- Weighted combination (0.75 / 0.25)
- **Benefit:** Better flood/non-flood discrimination

### 6. **Automatic Mixed Precision (AMP)**
- Reduces memory usage
- Speeds up training
- Maintains numerical stability with GradScaler
- **Benefit:** Larger batches, faster training

### 7. **Balanced Sampling**
- ImbalancedDatasetSampler for flood data
- Handles class imbalance (fewer flooded samples)
- **Benefit:** Better convergence on minority class

### 8. **Test-Time Augmentation (TTA)**
- Multiple augmented passes (horizontal flip, vertical flip)
- Average predictions
- **Benefit:** Robustness, improved inference accuracy

---

## Training & Inference Flow

### Complete Workflow

```
┌─── TRAINING ───┐
│                │
Step 1: Preprocess Data
├─ Prepare CSV files with image/mask paths
├─ Verify file structure
└─ Split train/validation

Step 2: Foundation Features Training
├─ Load pretrained foundation weights (if available)
├─ Train on SpaceNet 2/3/5 (45 epochs)
├─ Finetune on SpaceNet 8 (45 epochs)
└─ Save best model

Step 3: Flood Detection Training  
├─ Load foundation model weights
├─ Train flood siamese model (100 epochs)
├─ Use imbalanced sampler
├─ Save best + periodic checkpoints
└─ Done!

│
├─── INFERENCE ───┐
│                 │
Step 1: Foundation Inference
├─ Load foundation model
├─ Process test images (patch-based)
├─ Apply TTA
└─ Save building & road masks

Step 2: Flood Inference
├─ Load flood model
├─ Match pre/post image pairs
├─ Process with patch-based inference
├─ Apply TTA
└─ Save flood masks

Step 3: Post-processing
├─ Apply flood attribution heuristics
├─ Merge predictions
└─ Generate submission CSV

DONE!
```

---

## Performance Characteristics

### Computational Requirements
- **GPU Memory:** 8-16 GB (NVIDIA RTX 2080+)
- **Training Time:** 
  - Foundation: ~2 hours (45 epochs)
  - Flood: ~3-4 hours (100 epochs)
- **Inference Time:** ~5-10 minutes per image (with TTA)

### Model Sizes
- Foundation model: ~100 MB
- Flood model: ~110 MB

### Expected Metrics
- **Foundation IoU:** 0.7-0.8 (buildings), 0.65-0.75 (roads)
- **Flood IoU:** 0.6-0.7 (buildings), 0.55-0.65 (roads)

---

## Summary

Solution 2 achieves high accuracy through:
1. **Transfer learning** from multiple source datasets
2. **Siamese architecture** for effective change detection
3. **Multi-head design** for flexible multi-task learning
4. **Careful loss engineering** combining segmentation + classification
5. **Robust inference** with overlapping patches and TTA
6. **Smart sampling** to handle class imbalance

This modular approach allows practitioners to understand and modify each component independently, making it excellent for learning and experimentation!
