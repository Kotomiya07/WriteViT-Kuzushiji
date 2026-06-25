import os
import string

import torch

###############################################

EXP_NAME = os.environ.get("WRITEVIT_EXP_NAME", "Kuzushiji-COCO-column-phase5-recon-ddp2-wandb-20260621")
DATASET = os.environ.get("WRITEVIT_DATASET", "KUZUSHIJI_COCO_COLUMN")
RESUME = os.environ.get("WRITEVIT_RESUME", "0") == "1"


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _kuzushiji_default_alphabet():
    """Broad BMP Japanese alphabet used before dataset metadata is cached."""
    punctuation = "".join(chr(i) for i in range(0x3000, 0x303F + 1))
    symbols = "".join(chr(i) for i in range(0x25A0, 0x25FF + 1))
    kana = "".join(chr(i) for i in range(0x3041, 0x30FF + 1))
    kana = kana.replace("゗", "").replace("゘", "")
    kanji = "".join(chr(i) for i in range(0x4E00, 0x9FFF + 1))
    ascii_chars = string.ascii_letters + string.digits
    return "".join(dict.fromkeys(punctuation + symbols + kana + kanji + ascii_chars))

if DATASET == 'IAM':
    DATASET_PATHS = './File/IAM.pickle'
    NUM_WRITERS = 339
    WORDS_PATH = './File/english_words.txt'
    ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    MY_STRING = "The Statue of Liberty, arguably one of New York City's most iconic symbols, is a popular tourist attraction for first-time visitors to the city. This 150-foot monument was gifted to the United States from France in order to celebrate 100 years of America's independence. When Claire visited the Statue of Liberty for the first time, SHE instantly admired it as a symbol of freedom."

if DATASET == 'VNDB':
    DATASET_PATHS = './File/VN.pickle'
    NUM_WRITERS = 106
    WORDS_PATH = "./File/vn_words.txt"
    ALPHABET = 'aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵzAÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÉÈẺẼẸÊẾỀỂỄỆFGHIÍÌỈĨỊJKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢPQRSTUÚÙỦŨỤƯỨỪỬỮỰVWXYÝỲỶỸỴZ0123456789!'
    MY_STRING = "Trong cuộc sống này dù có gặp phải bao nhiêu khó khăn thử thách hãy luôn giữ vững niềm tin chăm chỉ học hỏi từng ngày sống chân thành yêu thương những người xung quanh và không ngừng ước mơ bởi chính sự kiên nhẫn và nỗ lực sẽ giúp ta vượt qua mọi giới hạn chạm tới thành công và hạnh phúc trọn vẹn"

if DATASET == 'CVL':
    DATASET_PATHS = './File/IAM.pickle'
    NUM_WRITERS = 283
    WORDS_PATH = './File/english_words.txt'
    ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    MY_STRING = "The Statue of Liberty, arguably one of New York City's most iconic symbols, is a popular tourist attraction for first-time visitors to the city. This 150-foot monument was gifted to the United States from France in order to celebrate 100 years of America's independence. When Claire visited the Statue of Liberty for the first time, SHE instantly admired it as a symbol of freedom."

if DATASET == 'KUZUSHIJI_COCO_COLUMN':
    HF_DATASET_ID = os.environ.get(
        "WRITEVIT_HF_DATASET_ID", "Kotomiya07/kuzushiji-dataset-coco"
    )
    HF_DATASET_SPLIT = os.environ.get("WRITEVIT_HF_DATASET_SPLIT", "train")
    HF_CACHE_DIR = os.environ.get("WRITEVIT_HF_CACHE_DIR", "./.cache/huggingface")
    KUZUSHIJI_SPLIT_SEED = _env_int("WRITEVIT_SPLIT_SEED", 42)
    KUZUSHIJI_VAL_RATIO = _env_float("WRITEVIT_VAL_RATIO", 0.1)
    KUZUSHIJI_TEST_RATIO = _env_float("WRITEVIT_TEST_RATIO", 0.1)
    KUZUSHIJI_LEVEL = "column"
    KUZUSHIJI_MAX_COLUMN_WIDTH = _env_int("WRITEVIT_MAX_COLUMN_WIDTH", 512)
    KUZUSHIJI_MAX_LABEL_LENGTH = _env_int("WRITEVIT_MAX_LABEL_LENGTH", 96)
    KUZUSHIJI_MIN_LABEL_LENGTH = _env_int("WRITEVIT_MIN_LABEL_LENGTH", 1)
    KUZUSHIJI_MAX_PAGES = _env_int("WRITEVIT_MAX_PAGES", 0)
    DATASET_PATHS = HF_DATASET_ID
    NUM_WRITERS = 44
    WORDS_PATH = './File/kuzushiji_column_texts.txt'
    ALPHABET = _kuzushiji_default_alphabet()
    MY_STRING = "いろはにほへと ちりぬるを わかよたれそ つねならむ"



###############################################
BACKBONE = "resnet18" # resnet18, vgg11, vgg19
IMG_HEIGHT = 32
resolution = 16
batch_size = _env_int("WRITEVIT_BATCH_SIZE", 4)
NUM_EXAMPLES = _env_int("WRITEVIT_NUM_EXAMPLES", 15)
VOCAB_SIZE = len(ALPHABET)
G_LR = _env_float("WRITEVIT_G_LR", 5e-5)
D_LR = _env_float("WRITEVIT_D_LR", 5e-5)
W_LR = _env_float("WRITEVIT_W_LR", 5e-5)
OCR_LR = _env_float("WRITEVIT_OCR_LR", 5e-5)

EPOCHS = _env_int("WRITEVIT_EPOCHS", 50)
NUM_CRITIC_GOCR_TRAIN = _env_int("WRITEVIT_NUM_CRITIC_GOCR_TRAIN", 2)
NUM_CRITIC_DOCR_TRAIN = _env_int("WRITEVIT_NUM_CRITIC_DOCR_TRAIN", 1)
ADV_LOSS_WEIGHT = _env_float("WRITEVIT_ADV_LOSS_WEIGHT", 2.0)
HINGE_MARGIN = _env_float("WRITEVIT_HINGE_MARGIN", 1.0)
D_LOGIT_REG_WEIGHT = _env_float("WRITEVIT_D_LOGIT_REG_WEIGHT", 0.0)
INK_LOSS_WEIGHT = _env_float("WRITEVIT_INK_LOSS_WEIGHT", 0.5)
GRAD_BALANCE_ALPHA = _env_float("WRITEVIT_GRAD_BALANCE_ALPHA", 0.7)
GRAD_BALANCE_BETA = _env_float("WRITEVIT_GRAD_BALANCE_BETA", 0.7)
GRAD_BALANCE_MAX = _env_float("WRITEVIT_GRAD_BALANCE_MAX", 5.0)
USE_GRAD_BALANCE = os.environ.get("WRITEVIT_USE_GRAD_BALANCE", "0") == "1"
G_GRAD_CLIP = _env_float("WRITEVIT_G_GRAD_CLIP", 5.0)
D_GRAD_CLIP = _env_float("WRITEVIT_D_GRAD_CLIP", 5.0)
OCR_GRAD_CLIP = _env_float("WRITEVIT_OCR_GRAD_CLIP", 5.0)
W_GRAD_CLIP = _env_float("WRITEVIT_W_GRAD_CLIP", 5.0)
G_EMA_DECAY = _env_float("WRITEVIT_G_EMA_DECAY", 0.999)
GENERATOR_NOISE_INIT = _env_float("WRITEVIT_GENERATOR_NOISE_INIT", 0.05)
USE_AMP = os.environ.get("WRITEVIT_AMP", "0") == "1"
AMP_DTYPE = os.environ.get("WRITEVIT_AMP_DTYPE", "bfloat16")
GRAD_ACCUM_STEPS = _env_int("WRITEVIT_GRAD_ACCUM_STEPS", 1)
USE_FUSED_OPTIMIZER = os.environ.get("WRITEVIT_FUSED_OPTIMIZER", "1") == "1"
USE_TORCH_COMPILE = os.environ.get("WRITEVIT_COMPILE", "0") == "1"
COMPILE_MODE = os.environ.get("WRITEVIT_COMPILE_MODE", "reduce-overhead")
USE_CHANNELS_LAST = os.environ.get("WRITEVIT_CHANNELS_LAST", "0") == "1"
DATALOADER_NUM_WORKERS = _env_int("WRITEVIT_NUM_WORKERS", 2)
DATALOADER_PREFETCH_FACTOR = _env_int("WRITEVIT_PREFETCH_FACTOR", 2)
DATALOADER_PERSISTENT_WORKERS = os.environ.get("WRITEVIT_PERSISTENT_WORKERS", "1") == "1"
D_AUG_PROB = _env_float("WRITEVIT_D_AUG_PROB", 0.0)
D_AUG_TRANSLATE_PIXELS = _env_int("WRITEVIT_D_AUG_TRANSLATE_PIXELS", 2)
D_AUG_CUTOUT_RATIO = _env_float("WRITEVIT_D_AUG_CUTOUT_RATIO", 0.12)
D_AUG_BRIGHTNESS = _env_float("WRITEVIT_D_AUG_BRIGHTNESS", 0.08)
D_AUG_CONTRAST = _env_float("WRITEVIT_D_AUG_CONTRAST", 0.15)
D_CONSISTENCY_WEIGHT = _env_float("WRITEVIT_D_CONSISTENCY_WEIGHT", 0.0)
RESET_D_ON_RESUME = os.environ.get("WRITEVIT_RESET_D_ON_RESUME", "0") == "1"
USE_FUSED_INK_DENSITY = os.environ.get("WRITEVIT_FUSED_INK_DENSITY", "0") == "1"
DDP_FIND_UNUSED_PARAMETERS = os.environ.get("WRITEVIT_DDP_FIND_UNUSED_PARAMETERS", "1") == "1"
DDP_TIMEOUT_MINUTES = _env_int("WRITEVIT_DDP_TIMEOUT_MINUTES", 30)
RECON_LOSS_WEIGHT = _env_float("WRITEVIT_RECON_LOSS_WEIGHT", 0.25)
RECON_FOREGROUND_WEIGHT = _env_float("WRITEVIT_RECON_FOREGROUND_WEIGHT", 2.0)
WRITER_FAKE_WEIGHT = _env_float("WRITEVIT_WRITER_FAKE_WEIGHT", 1.5)
WRITER_EMBED_MATCH_WEIGHT = _env_float("WRITEVIT_WRITER_EMBED_MATCH_WEIGHT", 0.0)
PREVIEW_RECONSTRUCTION = os.environ.get("WRITEVIT_PREVIEW_RECONSTRUCTION", "1") == "1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL = _env_int("WRITEVIT_SAVE_MODEL", 10)
SAVE_MODEL_HISTORY = _env_int("WRITEVIT_SAVE_MODEL_HISTORY", 50)

def init_project():
    import os, shutil
    if not os.path.isdir('saved_images'): os.mkdir('saved_images')
    if os.path.isdir(os.path.join('saved_images', EXP_NAME)): shutil.rmtree(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Real'))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Fake'))
