# GitHub Setup Guide

## 🚀 Hướng dẫn đưa project lên GitHub

### Bước 1: Khởi tạo Git Repository

```bash
# Khởi tạo git repository trong thư mục project
git init

# Thêm tất cả files vào staging
git add .

# Commit lần đầu
git commit -m "Initial commit: Self-Prompting QA System with Web Interface

✨ Features:
- Advanced web interface with clean UI/UX
- Multiple clustering algorithms (K-Means, Hierarchical, Spectral, Gaussian Mixture)
- Enhanced question diversity with 10+ categories
- Real-time metrics and model comparison
- Topic coherence analysis with 60-80% accuracy
- Professional minimalist design

🔧 Technical:
- Flask web server with REST API
- Flan-T5 and DistilBERT model integration
- Advanced evaluation metrics
- Comprehensive documentation"
```

### Bước 2: Tạo Repository trên GitHub

1. **Truy cập GitHub**: Đăng nhập vào https://github.com
2. **Tạo Repository mới**:
   - Click **"New repository"** 
   - Repository name: `self-prompting-qa-system`
   - Description: `🚀 Enhanced Web Interface for Self-Prompting Large Language Models - Advanced QA System with Clustering & Model Comparison`
   - Chọn **Public** (để mọi người có thể xem)
   - **KHÔNG** check "Add a README file" (vì đã có sẵn)
   - **KHÔNG** check "Add .gitignore" (vì đã có sẵn)
   - **KHÔNG** check "Choose a license" (vì đã có LICENSE file)
3. **Click "Create repository"**

### Bước 3: Kết nối Local Repository với GitHub

```bash
# Thêm remote origin (thay your-username bằng username GitHub của bạn)
git remote add origin https://github.com/your-username/self-prompting-qa-system.git

# Đổi nhánh chính thành main (nếu cần)
git branch -M main

# Push code lên GitHub
git push -u origin main
```

### Bước 4: Thiết lập Repository trên GitHub

#### 4.1 **Repository Settings**
1. Vào **Settings** tab
2. **General**:
   - Features: Enable Issues, Wiki (nếu muốn)
   - Pull Requests: Enable merge commits
3. **Security**: 
   - Enable vulnerability alerts

#### 4.2 **Repository Topics & Tags**
Thêm topics để dễ tìm kiếm:
```
self-prompting, question-answering, nlp, machine-learning, 
clustering, web-interface, flask, flan-t5, bert, python
```

#### 4.3 **GitHub Pages (Optional)**
Nếu muốn host demo online:
1. **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** / **root**

### Bước 5: Tạo Release đầu tiên

1. **Vào tab "Releases"**
2. **Click "Create a new release"**
3. **Tag version**: `v1.0.0`
4. **Release title**: `🚀 Self-Prompting QA System v1.0.0 - Initial Release`
5. **Description**:

```markdown
## 🎉 Initial Release - Self-Prompting QA System v1.0.0

### ✨ Major Features
- **🌐 Advanced Web Interface**: Professional UI with real-time metrics
- **🔍 Multiple Clustering Algorithms**: K-Means, Hierarchical, Spectral, Gaussian Mixture
- **🎨 Enhanced Question Diversity**: 10+ categories (Definition, Benefits, Challenges, etc.)
- **📊 Comprehensive Metrics**: Topic Coherence (60-80%), Silhouette Score (0.3-0.6)
- **🤖 Model Comparison**: Flan-T5 vs DistilBERT side-by-side analysis

### 🔧 Technical Highlights
- Flask REST API with responsive design
- Advanced evaluation metrics and performance insights
- Smart category selection with AI-powered relevance matching
- Clean minimalist UI (white/black theme)
- 100% unique question generation

### 🚀 Quick Start
```bash
pip install -r requirements.txt
python web_interface.py
```
Navigate to http://localhost:5001

### 📊 Performance Benchmarks
- Generation: 3-5 QA pairs/second
- Clustering: 2-10 seconds for 15-50 samples  
- Topic Coherence: 60-85% accuracy
- Model Inference: 1-3 seconds per question

### 📚 Based on Research
Self-Prompting Large Language Models for Zero-Shot Open-Domain QA (NAACL 2024)
```

6. **Click "Publish release"**

### Bước 6: Repository Enhancements

#### 6.1 **Issues Templates**
Tạo `.github/ISSUE_TEMPLATE/` với:
- `bug_report.md`
- `feature_request.md`  
- `question.md`

#### 6.2 **GitHub Actions (Optional)**
Tạo `.github/workflows/ci.yml` cho automated testing:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/ # nếu có tests
```

### Bước 7: Documentation Enhancements

#### 7.1 **Wiki Setup**
1. Enable Wiki trong Settings
2. Tạo pages:
   - **Home**: Overview và quick start
   - **API Documentation**: Chi tiết API endpoints
   - **Installation Guide**: Hướng dẫn cài đặt chi tiết
   - **Troubleshooting**: Các vấn đề thường gặp

#### 7.2 **README Badges**
Thêm badges vào README.md:

```markdown
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![GitHub stars](https://img.shields.io/github/stars/your-username/self-prompting-qa-system.svg)
![GitHub forks](https://img.shields.io/github/forks/your-username/self-prompting-qa-system.svg)
```

### Bước 8: Post-Upload Checklist

- [ ] Repository URL hoạt động: `https://github.com/your-username/self-prompting-qa-system`
- [ ] README.md hiển thị đúng với images và formatting
- [ ] .gitignore hoạt động (không upload các files không cần thiết)
- [ ] LICENSE file có mặt
- [ ] Requirements.txt chính xác
- [ ] Issues và Discussions enabled (nếu muốn)
- [ ] Repository topics đã thêm
- [ ] Release v1.0.0 đã publish

### Bước 9: Chia sẻ và Promotion

1. **Social Media**: Chia sẻ trên LinkedIn, Twitter với hashtags #NLP #MachineLearning #Python
2. **Communities**: Post trên Reddit r/MachineLearning, r/Python
3. **Academic**: Submit to awesome lists, ML conferences
4. **Demo**: Host live demo trên Heroku/Railway (optional)

---

## 🎉 Chúc mừng! 

Project của bạn giờ đã live trên GitHub với:
- ⭐ Professional documentation  
- 🔧 Clean codebase
- 📊 Comprehensive features
- 🚀 Ready for collaboration

**Repository URL**: `https://github.com/your-username/self-prompting-qa-system` 