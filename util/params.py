import torch

###############################################

EXP_NAME = "IAM-1000"; RESUME = True

###############################################

IMG_HEIGHT = 32
resolution = 16
batch_size = 8
NUM_EXAMPLES = 50
TN_HIDDEN_DIM = 512
TN_DROPOUT = 0.1
TN_NHEADS = 8
TN_DIM_FEEDFORWARD = 512
TN_ENC_LAYERS = 1
TN_DEC_LAYERS = 1
ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
VOCAB_SIZE = len(ALPHABET)
G_LR = 0.0002
D_LR = 0.0002
W_LR = 0.0002
OCR_LR = 0.0002
EPOCHS = 100000
NUM_CRITIC_GOCR_TRAIN = 2
NUM_CRITIC_DOCR_TRAIN = 1
NUM_CRITIC_GWL_TRAIN = 2
NUM_CRITIC_DWL_TRAIN = 1
NUM_FID_FREQ = 100
DATASET = ['IAM']
DATASET_PATHS = {'IAM':'../IAM_32.pickle'}
NUM_WRITERS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_SEQ = True
NUM_WORDS = 3
if not IS_SEQ: NUM_WORDS = NUM_EXAMPLES
IS_CC = False
IS_KLD = False
ADD_NOISE = False
ALL_CHARS = False
SAVE_MODEL = 5
SAVE_MODEL_HISTORY = 100

