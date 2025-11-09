from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# ===================== MODEL CONFIGURATION =====================
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"  # Using cuda or cpu for training
_C.MODEL.DEVICE_ID = '0'    # ID number of GPU
_C.MODEL.NAME = 'resnet50'  # Name of backbone
_C.MODEL.ARCH_NAME = 'build_transformer'  # Name of meta architecture
_C.MODEL.LAST_STRIDE = 1    # Last stride of backbone, for resnet
_C.MODEL.PRETRAIN_PATH = ''  # Path to pretrained model of backbone
_C.MODEL.NECK = 'bnneck'

_C.MODEL.DIST_TRAIN = False  # If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False  # If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.IF_LABELSMOOTH = 'on'  # If train with label smooth, options: 'on', 'off'
_C.MODEL.COS_LAYER = False  # If train with arcface loss, options: 'True', 'False'

# loss function settings
_C.MODEL.IF_WITH_CENTER = 'no'  # Whether to include center loss in training (options: 'yes' or 'no')
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'   # Type of metric loss (options: 'triplet', 'center', 'triplet_center')
_C.MODEL.I2T_LOSS_WEIGHT = 1.0  # first stage image to text loss weight for CLIP-ReID
# Transformer setting
_C.MODEL.DROP_PATH = 0.1  # DropPath rate
_C.MODEL.DROP_OUT = 0.0  # Dropout rate
_C.MODEL.ATT_DROP_RATE = 0.0  # Attention dropout rate
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'   # Type of transformer for vit,transred and etc.
_C.MODEL.STRIDE_SIZE = [16, 16]
# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
# Text settings
_C.MODEL.FREEZE_TEXT = False
_C.MODEL.USE_TEXT = False
_C.MODEL.TEXT_PROMPT = 0
_C.MODEL.TEXT_LEN = 77
_C.MODEL.TEXT_TYPE = 'attribute'  # Type of text input (options: 'captions', 'attribute')
_C.MODEL.TEXT_FORMAT = 'hybird'
_C.MODEL.USE_ATTR = False

# Visual Prompt settings
_C.MODEL.VISUAL_PROMPT = 0  # Whether to use visual prompt or length of visual prompt
# Head settings
_C.MODEL.HEAD_UNIFORM = False   # Whether to use contact features

# ===================== INPUT CONFIGURATION =====================
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]  # Image size during training
_C.INPUT.SIZE_TEST = [256, 128]  # Image size during testing
_C.INPUT.PROB = 0.5  # Probability for random horizontal flip
_C.INPUT.RE_PROB = 0.5  # Probability for random erasing
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]  # Mean values for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]  # Standard deviation values for image normalization
_C.INPUT.PADDING = 10  # Padding size for images


# ===================== DATASET CONFIGURATION =====================
_C.DATASETS = CN()
_C.DATASETS.NAMES = ('market1501')  # Names of datasets for training
_C.DATASETS.ROOT_DIR = './data'  # Root directory for datasets
_C.DATASETS.TESTS = ('market1501',)  # Names of datasets for evaluation
_C.DATASETS.VERSION = 1  # Version of datasets splition of train and test

# ===================== DATALOADER CONFIGURATION =====================
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8  # Number of data loading threads
_C.DATALOADER.SAMPLER = 'softmax_triplet'  # Sampler for data loading
_C.DATALOADER.SAMPLER_NAME = 'RandomIdentitySampler'    # Sampler ways for data loading
_C.DATALOADER.AUG = False   # Whether to use data diy augmentation ,only for training
_C.DATALOADER.NUM_INSTANCE = 16  # Number of instance for one batch

# ===================== SOLVER CONFIGURATION =====================
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.OPTIMIZER_NAME = "Adam"  # Name of optimizer
_C.SOLVER.MAX_EPOCHS = 50  # Maximum number of training epochs
_C.SOLVER.BASE_LR = 0.00035  # Base learning rate
_C.SOLVER.LARGE_FC_LR = False  # Whether to use a larger learning rate for fully connected layers
_C.SOLVER.BIAS_LR_FACTOR = 2  # Learning rate factor for bias terms
_C.SOLVER.MOMENTUM = 0.9  # Momentum for optimizer
_C.SOLVER.MARGIN = 0.3  # Margin for triplet loss
_C.SOLVER.CLUSTER_MARGIN = 0.3  # Margin for cluster loss
_C.SOLVER.CENTER_LR = 0.5  # Learning rate for center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005  # Weight for center loss
_C.SOLVER.RANGE_K = 2  # K value for range loss
_C.SOLVER.RANGE_MARGIN = 0.3  # Margin for range loss
_C.SOLVER.RANGE_ALPHA = 0  # Alpha value for range loss
_C.SOLVER.RANGE_BETA = 1  # Beta value for range loss
_C.SOLVER.RANGE_LOSS_WEIGHT = 1  # Weight for range loss
_C.SOLVER.WEIGHT_DECAY = 0.0001  # Weight decay for optimizer
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001  # Weight decay for bias terms
_C.SOLVER.GAMMA = 0.1  # Decay rate for learning rate when using MultiStepLR
_C.SOLVER.STEPS = (40, 70)  # Steps for learning rate decay when using MultiStepLR CosineAnnealingLR
_C.SOLVER.LR_MIN = 0.000016  # for CosineAnnealingLR
_C.SOLVER.WARMUP_FACTOR = 0.01  # Warmup factor for learning rate 0.1 for clip 0.01 for vit
_C.SOLVER.WARMUP_ITERS = 10  # Number of warmup iterations
_C.SOLVER.WARMUP_METHOD = "linear"  # Warmup method (options: 'constant', 'linear')
_C.SOLVER.COSINE_MARGIN = 0.5  # Margin for cosine loss
_C.SOLVER.COSINE_SCALE = 30  # Scale for cosine loss
_C.SOLVER.SEED = 1111  # Random seed for reproducibility
_C.MODEL.NO_MARGIN = True  # Whether to disable margin
_C.SOLVER.CHECKPOINT_PERIOD = 50  # Period for saving checkpoints
_C.SOLVER.LOG_PERIOD = 10  # Period for logging training progress
_C.SOLVER.EVAL_PERIOD = 1  # Period for evaluation
_C.SOLVER.IMS_PER_BATCH = 64  # Number of images per batch
# stage1
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE1 = CN()
_C.SOLVER.STAGE1.TRAIN = False
_C.SOLVER.STAGE1.PRETRAIN_PATH = ''
_C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE1.MAX_EPOCHS = 100    # Number of max epoches
_C.SOLVER.STAGE1.BASE_LR = 3e-4  # Base learning rate
_C.SOLVER.STAGE1.MOMENTUM = 0.9  # Momentum
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005  # Weight decay for optimizer
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005   # Weight decay for bias terms
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01  # warm up factor
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5  # warm up epochs
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016  # for CosineAnnealingLR
_C.SOLVER.STAGE1.WARMUP_ITERS = 500
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"  # method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10  # epoch number of saving checkpoints
_C.SOLVER.STAGE1.LOG_PERIOD = 100   # iteration of display training log
_C.SOLVER.STAGE1.EVAL_PERIOD = 10

# ===================== TEST CONFIGURATION =====================
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128  # Number of images per batch during test
_C.TEST.RE_RANKING = False  # If test with re-ranking, options: 'True','False'
_C.TEST.WEIGHT = ""  # Path to trained model
_C.TEST.NECK_FEAT = 'after'  # Which BNNeck feature to use for testing (options: 'before' or 'after')
_C.TEST.FEAT_NORM = 'yes'    # Whether to normalize features before testing
_C.TEST.DIST_MAT = "dist_mat.npy"   # Name for saving the distmat after testing.
_C.TEST.TIME_MODE = 'mix'       # time mode for reid, 'mix': ,'same': day to day or night to night,'cross': day to night or night to day
# ===================== MISC OPTIONS =====================
_C.OUTPUT_DIR = "./logs"   # Output directory for checkpoints and logs
