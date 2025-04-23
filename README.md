# CSE544 - Computer Vision Assignments

**Course:** CSE544 (Computer Vision)  
**Instructor:** Dr. Saket Anand  
**Institution:** IIIT Delhi

---

## 📂 Repository Overview

This repository hosts all programming assignments for the CSE544 Computer Vision course. Each assignment is organized into its own top-level folder (`HW1`, `HW2`, etc.), and follows a consistent structure:

```
├── HW1/
│   ├── problem_statement/       # PDF or markdown describing the assignment
│   └── submission/              # Your code, notebooks, models, and results
├── HW2/
│   ├── problem_statement/
│   └── submission/
├── HW3/
│   ├── problem_statement/
│   └── submission/
└── README.md                    # This file
```

- **problem_statement/**  
  Contains the assignment description, dataset links, and detailed instructions.

- **submission/**  
  Contains your implementation: source code, Jupyter notebooks, trained model weights (where applicable), and any output/results.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/samridhgirdhar/CSE544-Computer-Vision.git
cd CSE544-Computer-Vision
```

### 2. Install Dependencies

Each assignment may have its own dependencies, but you can start with the shared requirements:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install common libraries
pip install -r requirements.txt
```

> If a specific assignment has extra dependencies, check its `submission/requirements.txt` or the header of its notebook.

### 3. Git Large File Storage (LFS)

Some assignments (e.g., segmentation) include large model checkpoints (`.pth` files) which exceed GitHub's 100 MB limit. We use [Git LFS](https://git-lfs.github.com/) to handle these:

```bash
# Install Git LFS
git lfs install

# Pull LFS-tracked files
git lfs pull
```

If you encounter missing files or large file errors, ensure Git LFS is installed and initialized.

---

## 🤝 Acknowledgements

- Instructor: Dr. Saket Anand, IIIT Delhi
- Course materials provided by IIIT Delhi’s Computer Vision lab

---

*Repository maintained by Samridh Girdhar*

