# 🎓 Bài Tập Lớn Môn NLP - Học Kì 2025

**Đề tài**: Xử Lý Ngôn Ngữ Tự Nhiên - Text Classification & Machine Translation

**Sinh viên thực hiện**:
- Ngô Viết Thuyết
- Nguyễn Đức Hưng

**Trường**: Đại học Công nghệ - ĐHQGHN

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Bài 1: Text Classification](#-bài-1-text-classification)
- [Bài 2: Machine Translation](#-bài-2-machine-translation)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
- [Kết Quả](#-kết-quả)
- [Tài Liệu Tham Khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng Quan

Repository này chứa giải pháp cho 2 bài tập lớn môn NLP:

1. **Bài 1**: Phân loại văn bản (Text Classification) sử dụng Transformer
2. **Bài 2**: Dịch máy song ngữ Anh-Việt (Machine Translation) sử dụng mBART-50

---

## 📝 Bài 1: Text Classification

### 🎯 Mục tiêu

Xây dựng mô hình phân loại văn bản sử dụng kiến trúc **Transformer từ scratch** (không dùng pre-trained).

### 📊 Dataset

- **Source**: Custom dataset
- **Task**: Multi-class text classification
- **Data location**: `Data/Data1/`

### 🏗️ Kiến trúc

**Transformer Architecture**:
- **Encoder**: Multi-head self-attention + Feed-forward network
- **Positional Encoding**: Sinusoidal encoding
- **Layers**: 6 encoder layers
- **Attention Heads**: 8 heads
- **Hidden Size**: 512
- **Feed-forward Dim**: 2048

### 📁 Files

```
Code/Bai1/
└── transformerFinalHung.ipynb    # Notebook chính với Transformer implementation
```

### 🚀 Cách chạy

```bash
# 1. Mở Jupyter Notebook
jupyter notebook Code/Bai1/transformerFinalHung.ipynb

# 2. Chạy lần lượt các cell từ đầu đến cuối
```

---

## 🌐 Bài 2: Machine Translation

### 🎯 Mục tiêu

Fine-tune mô hình **mBART-50** cho 2 hướng dịch:
1. **Tiếng Việt → Tiếng Anh** (Medical domain)
2. **Tiếng Anh → Tiếng Việt** (Medical domain)

### 📊 Dataset

- **Source**: Medical research abstracts
- **Size**: 500,000 parallel sentences (EN-VI)
- **Domain**: Medical/Healthcare
- **Data location**: `Data/Data2/`

**Files**:
```
Data/Data2/
├── train.en.txt              (500,000 sentences)
├── train.vi.txt              (500,000 sentences)
├── public_test.en.txt        (3,000 sentences)
└── public_test.vi.txt        (3,000 sentences)
```

### 🏗️ Model Architecture

**mBART-50** (facebook/mbart-large-50-many-to-many-mmt):
- **Type**: Multilingual Seq2Seq Transformer
- **Parameters**: ~611M
- **Languages**: 50 languages
- **Pre-training**: Denoising autoencoding
- **Fine-tuning**: Medical domain EN-VI translation

### 📁 Files

```
Code/Bai2/
├── mBART50_VI_EN.ipynb          # Notebook VI→EN (33 code cells + 33 markdown)
├── mBART50_EN_VI.ipynb          # Notebook EN→VI (26 code cells + 26 markdown)
├── NOTEBOOK_GUIDE.md            # Hướng dẫn chi tiết cấu trúc notebook
├── MARKDOWN_VI_EN.txt           # Markdown cho VI→EN
├── MARKDOWN_EN_VI.txt           # Markdown cho EN→VI
├── insert_markdown.py           # Script tự động thêm markdown
└── add_markdown.py              # Script helper
```

### 🔧 Training Configuration

#### **Common Config**:
```python
# Data preprocessing
N_TOTAL = 30000              # Số pairs dùng để train
TRAIN_RATIO = 0.975          # 97.5% train, 2.5% val
MIN_CHARS = 2
MAX_CHARS = 400

# Training
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4    # Effective batch = 8
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
FP16 = True                  # Mixed precision training
```

#### **VI→EN Specific**:
```python
SRC_LANG = "vi_VN"
TGT_LANG = "en_XX"
LEARNING_RATE = 1e-5
```

#### **EN→VI Specific**:
```python
SRC_LANG = "en_XX"
TGT_LANG = "vi_VN"
LEARNING_RATE = 3e-5         # Higher LR for EN→VI
```

### 📈 Kết quả

#### **VI→EN (Vietnamese → English)**

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| **BLEU** | 17.77 | **31.75** | +13.98 |
| **Gemini Judge** | - | 78.5/100 | - |
| **Medical Score** | 54.8/100 | 70.1/100 | +15.3 |

**Top Errors**:
- NE_mismatch_omit: 61.5% (Thiếu tên riêng)
- Terminology_miss: 33.5% (Thiếu thuật ngữ y khoa)
- Number_mismatch: 11.5% (Số không khớp)

#### **EN→VI (English → Vietnamese)**

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| **BLEU** | 26.09 | **43.42** | +17.33 |
| **Gemini Judge** | - | 75.2/100 | - |
| **Medical Score** | 67.1/100 | 72.9/100 | +5.8 |

**Top Errors**:
- Repetition: 19.5% (Lặp từ)
- Number_mismatch: 16.0% (Số không khớp)
- Vietnamese_function_word_missing: 4.5% (Thiếu từ chức năng)

### 💡 Features

#### **Data Pipeline**:
- ✅ Auto dataset discovery trên Kaggle
- ✅ Data cleaning (remove whitespace, normalize)
- ✅ Filtering (length constraints)
- ✅ Deduplication (MD5 hash)
- ✅ Train/Val/Test splitting

#### **Training**:
- ✅ Mixed precision training (FP16)
- ✅ Gradient accumulation
- ✅ Early stopping (patience=2)
- ✅ Best model selection (by val_loss)
- ✅ Auto push to HuggingFace Hub

#### **Evaluation**:
- ✅ sacreBLEU scoring
- ✅ Error analysis (automatic error tagging)
- ✅ Gemini Judge scoring (LLM-based evaluation)
- ✅ Medical domain scoring (rule-based)
- ✅ Visualization (loss curves, error distribution)

### 🚀 Cách chạy

#### **Option 1: Kaggle Notebook (Recommended)**

```bash
# 1. Upload dataset lên Kaggle
#    - Tạo dataset với 4 files: train.en.txt, train.vi.txt, public_test.en.txt, public_test.vi.txt

# 2. Tạo notebook mới và enable GPU
#    Settings > Accelerator > GPU T4 x2

# 3. Add Kaggle Secrets
#    - HF_TOKEN: Hugging Face write token
#    - GeminiAPI: Google Gemini API key

# 4. Upload notebook
#    - mBART50_VI_EN.ipynb (cho VI→EN)
#    - hoặc mBART50_EN_VI.ipynb (cho EN→VI)

# 5. Chạy lần lượt các cells
```

#### **Option 2: Local (với GPU)**

```bash
# 1. Clone repo
git clone https://github.com/NgoVietThuyet/NLP.git
cd NLP

# 2. Cài đặt dependencies
pip install transformers datasets sacrebleu accelerate torch

# 3. Setup environment variables
export HF_TOKEN="your_huggingface_token"
export GEMINI_API_KEY="your_gemini_key"

# 4. Chạy notebook
jupyter notebook Code/Bai2/mBART50_VI_EN.ipynb
```

### 🌟 Model Checkpoints

**Hugging Face Hub**:
- **VI→EN**: [ngothuyet/mbart50-vien](https://huggingface.co/ngothuyet/mbart50-vien)
- **EN→VI**: [ngothuyet/mbart50-envi](https://huggingface.co/ngothuyet/mbart50-envi)

**Usage**:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# VI→EN
model = AutoModelForSeq2SeqLM.from_pretrained("ngothuyet/mbart50-vien")
tokenizer = AutoTokenizer.from_pretrained("ngothuyet/mbart50-vien")

# EN→VI
model = AutoModelForSeq2SeqLM.from_pretrained("ngothuyet/mbart50-envi")
tokenizer = AutoTokenizer.from_pretrained("ngothuyet/mbart50-envi")
```

---

## 📂 Cấu Trúc Thư Mục

```
NLP/
├── README.md                               # File này
├── BAO_CAO_BAI1_FINAL.tex                 # Báo cáo Bài 1
├── BAO_CAO_BAI2_FINAL_V3.tex              # Báo cáo Bài 2
├── Bao_cao_VLSP2025_MT.tex                # Báo cáo VLSP2025
│
├── Code/
│   ├── Bai1/
│   │   └── transformerFinalHung.ipynb     # Transformer từ scratch
│   │
│   └── Bai2/
│       ├── mBART50_VI_EN.ipynb            # Fine-tune VI→EN
│       ├── mBART50_EN_VI.ipynb            # Fine-tune EN→VI
│       ├── NOTEBOOK_GUIDE.md              # Hướng dẫn chi tiết
│       ├── MARKDOWN_VI_EN.txt             # Markdown VI→EN
│       ├── MARKDOWN_EN_VI.txt             # Markdown EN→VI
│       └── *.py                           # Helper scripts
│
└── Data/
    ├── Data1/                             # Dataset Bài 1
    └── Data2/                             # Dataset Bài 2
        ├── train.en.txt                   # 500k English sentences
        ├── train.vi.txt                   # 500k Vietnamese sentences
        ├── public_test.en.txt             # 3k test EN
        └── public_test.vi.txt             # 3k test VI
```

---

## 💻 Yêu Cầu Hệ Thống

### **Minimum Requirements**:

```
OS: Windows 10/11, Linux, macOS
Python: 3.8+
RAM: 16GB (32GB recommended)
GPU: NVIDIA GPU with 8GB+ VRAM (T4, P100, V100, A100)
CUDA: 11.0+
Storage: 20GB free space
```

### **Dependencies**:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
accelerate>=0.20.0
sacrebleu>=2.3.0
jupyter
pandas
matplotlib
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### **1. Clone Repository**

```bash
git clone https://github.com/NgoVietThuyet/NLP.git
cd NLP
```

### **2. Tạo Virtual Environment**

```bash
# Dùng venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Hoặc dùng conda
conda create -n nlp python=3.10
conda activate nlp
```

### **3. Cài Đặt Dependencies**

```bash
# Install PyTorch (with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers & Co.
pip install transformers datasets accelerate

# Install evaluation tools
pip install sacrebleu google-generativeai

# Install Jupyter
pip install jupyter ipykernel
```

### **4. Setup API Keys**

#### **Hugging Face Token**:
```bash
# Get key from: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxxxxxxxxx"

# Hoặc login qua CLI
huggingface-cli login
```

#### **Gemini API Key**:
```bash
# Get key from: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="AIzaxxxxxxxxxxxxx"
```

### **5. Run Notebooks**

```bash
# Start Jupyter
jupyter notebook

# Mở notebook:
# - Code/Bai1/transformerFinalHung.ipynb
# - Code/Bai2/mBART50_VI_EN.ipynb
# - Code/Bai2/mBART50_EN_VI.ipynb
```

---

## 📊 Kết Quả

### **Bài 2: Machine Translation Summary**

| Direction | BLEU Baseline | BLEU Fine-tuned | Improvement |
|-----------|---------------|-----------------|-------------|
| **VI→EN** | 17.77 | **31.75** | +13.98 |
| **EN→VI** | 26.09 | **43.42** | +17.33 |

**Observation**: EN→VI đạt BLEU cao hơn vì model dễ học pattern EN→VI và medical terminology tiếng Việt ít ambiguous hơn.

---

## 🐛 Troubleshooting

### **Issue 1: CUDA Out of Memory**

```python
# Giảm batch size
TRAIN_BS = 1
GRAD_ACC = 8

# Clear cache
torch.cuda.empty_cache()
```

### **Issue 2: Tokenizer KeyError**

```python
# Phải set src_lang và tgt_lang
tokenizer.src_lang = "vi_VN"
tokenizer.tgt_lang = "en_XX"
```

### **Issue 3: Gemini API Rate Limit**

```python
# Tăng sleep time
time.sleep(2)  # giữa mỗi request
```

---

## 📚 Tài Liệu Tham Khảo

### **Papers**:
1. **Attention Is All You Need** - https://arxiv.org/abs/1706.03762
2. **mBART** - https://arxiv.org/abs/2001.08210

### **Libraries**:
- **Transformers**: https://huggingface.co/docs/transformers
- **sacreBLEU**: https://github.com/mjpost/sacrebleu

### **Models**:
- **mBART-50**: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
- **Our models**:
  - https://huggingface.co/ngothuyet/mbart50-vien
  - https://huggingface.co/ngothuyet/mbart50-envi

---

## 👥 Contributors

- **Ngô Viết Thuyết** - [GitHub](https://github.com/NgoVietThuyet)
- **Nguyễn Đức Hưng**

---

## 📄 License

MIT License - Free to use for research and education.

---

## 📧 Contact

- **Repository**: https://github.com/NgoVietThuyet/NLP
- **Issues**: https://github.com/NgoVietThuyet/NLP/issues

---

**Last Updated**: December 2024
