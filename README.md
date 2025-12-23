# Bài Tập Lớn NLP - 2025

**Sinh viên**: 

Ngô Viết Thuyết: 23021730 

Đào Đức Mạnh: 23021618 

Lưu Văn Hùng: 23021566

**Trường**: Đại học Công nghệ - ĐHQGHN

---

## Tổng Quan

Repository chứa 2 bài tập lớn môn NLP:
- **Bài 1**: Xây dựng mô hình dịch máy Seq2Seq với Transformer (từ scratch)
- **Bài 2**: Áp dụng cho bài toán VLSP 2025 Shared Task - Machine transaltion

---

## Bài 1: 

### Mô tả
Xây dựng mô hình phân loại văn bản bằng **Transformer từ đầu** (không dùng pre-trained).

### Kiến trúc
- 6 encoder layers, 8 attention heads
- Hidden size: 512, FFN: 2048
- Positional encoding: Sinusoidal

### File
```
Code/Bai1/transformerFinalHung.ipynb
```

---

## Bài 2: 

### Mô tả
Fine-tune **mBART-50** cho dịch máy Y khoa Anh-Việt (2 chiều).

### Dataset
- **Train**: 500,000 câu song song EN-VI (Medical domain)
- **Test**: 3,000 câu
- **Source**: `Data/Data2/` (train.en.txt, train.vi.txt, public_test.*.txt)

### Kết quả

| Hướng | Baseline BLEU | Fine-tuned BLEU | Cải thiện |
|-------|---------------|-----------------|-----------|
| **VI→EN** | 17.77 | **31.75** | +13.98 |
| **EN→VI** | 26.09 | **43.42** | +17.33 |

### Files
```
Code/Bai2/
├── mBART50_VI_EN.ipynb      # Notebook VI→EN (66 cells)
├── mBART50_EN_VI.ipynb      # Notebook EN→VI (52 cells)
```

### Models
- **VI→EN**: [ngothuyet/mbart50-vien](https://huggingface.co/ngothuyet/mbart50-vien)
- **EN→VI**: [ngothuyet/mbart50-envi](https://huggingface.co/ngothuyet/mbart50-envi)

### Training Config
```python
# Common
EPOCHS = 3
BATCH_SIZE = 2 (effective = 8 với grad_acc=4)
N_TOTAL = 30,000 pairs

# VI→EN
LR = 1e-5

# EN→VI
LR = 3e-5
```

### Features
- Data cleaning + deduplication (500k → 340k)
- Mixed precision training (FP16)
- Early stopping (patience=2)
- Error analysis (tự động tag lỗi)
- Gemini Judge scoring
- Medical domain scoring
- Auto push to HuggingFace

---

## Cấu Trúc Thư Mục

```
NLP/
├── README.md
├── Code/
│   ├── Bai1/transformerFinalHung.ipynb
│   └── Bai2/
│       ├── mBART50_VI_EN.ipynb
│       ├── mBART50_EN_VI.ipynb
└── Data/
    ├── Data1/          # Dataset Bài 1
    └── Data2/          # Dataset Bài 2 (500k EN-VI pairs)
```

---

## Cài Đặt

### Requirements
```
Python 3.8+
GPU: 8GB+ VRAM (T4/P100/V100)
```

### Installation
```bash
# Clone
git clone https://github.com/NgoVietThuyet/NLP.git
cd NLP

# Install
pip install torch transformers datasets sacrebleu accelerate

# Setup API keys
export HF_TOKEN="your_token"
export GEMINI_API_KEY="your_key"

# Run
jupyter notebook Code/Bai2/mBART50_VI_EN.ipynb
```

---

## Chạy Trên Kaggle (Recommended)

1. Upload dataset lên Kaggle
2. Enable GPU T4 x2
3. Add Secrets: `HF_TOKEN`, `GeminiAPI`
4. Upload notebook và chạy

---

## Kết Quả Đánh Giá (Bài 2)

### VI→EN
- **BLEU**: 31.75
- **Gemini Judge**: 78.5/100
- **Medical Score**: 70.1/100
- **Top errors**: NE mismatch (61%), Terminology miss (33%)

### EN→VI
- **BLEU**: 43.42
- **Gemini Judge**: 75.2/100
- **Medical Score**: 72.9/100
- **Top errors**: Repetition (19%), Number mismatch (16%)

---

## Troubleshooting

**CUDA OOM**: Giảm `BATCH_SIZE = 1`, tăng `GRAD_ACC = 8`
**KeyError tokenizer**: Set `tokenizer.src_lang` và `tokenizer.tgt_lang`
**Gemini 429**: Tăng `time.sleep(2)` giữa requests

---

## Tài Liệu

- **Papers**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [mBART](https://arxiv.org/abs/2001.08210)
- **Models**: [mBART-50](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- **Docs**: [Transformers](https://huggingface.co/docs/transformers), [sacreBLEU](https://github.com/mjpost/sacrebleu)

