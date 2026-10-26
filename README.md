<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=180&section=header&text=Trustworthy%20ML&fontSize=40&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=Adversarial%20attacks%2C%20certified%20robustness%20and%20backdoors&descSize=15&descAlignY=58" width="100%" />

**Trustworthy Machine Learning · Tel Aviv University**

<img src="https://img.shields.io/badge/%F0%9F%8E%93%20Tel%20Aviv%20University-30363d?style=for-the-badge" />
<img src="https://img.shields.io/badge/Course-0368--4075-30363d?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-30363d?style=for-the-badge&logo=python&logoColor=FFD43B" />
<img src="https://img.shields.io/badge/PyTorch-30363d?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" />

</div>

<br/>

> [!NOTE]
> My solutions to the homework of the Trustworthy ML course: attacking image classifiers, defending them, and detecting backdoors.

## 📚 Assignments

| | Assignment | What I implemented |
|:-:|---|---|
| ⚔️ | **[HW1: Adversarial attacks](hw1-release/)** | White-box **PGD** (targeted and untargeted) · black-box **NES** gradient estimation with PGD and momentum · **ensemble** transfer attacks |
| 🛡️ | **[HW2: Defenses and backdoors](hw2-release/)** | **Free adversarial training** · **randomized smoothing** with certification · **Neural Cleanse** backdoor detection · Carlini-Wagner loss |

Each assignment folder has the task (`hw*.pdf`), my code and my write-up (`writeup.pdf`).

## 🗂️ Structure

```
TrustWorthy_ML_0368_4075/
├── hw1-release/
│   ├── attacks.py        # PGD, NES black-box PGD, ensemble PGD
│   ├── main_a.py         # part A runner
│   ├── main_b.py         # part B runner
│   ├── main_c.py         # part C runner
│   ├── models.py, utils.py, consts.py
│   ├── hw1.pdf           # assignment
│   └── writeup.pdf       # my report
└── hw2-release/
    ├── defenses.py       # free adversarial training, randomized smoothing, Neural Cleanse
    ├── attacks.py        # PGD + Carlini-Wagner loss
    ├── main_a.py / main_b.py / main_c.py
    ├── hw2.pdf           # assignment
    └── writeup.pdf       # my report
```

## 🚀 Run

```bash
cd hw1-release            # or hw2-release
pip install -r requirements.txt
python main_a.py          # main_b.py, main_c.py for the other parts
```

> [!IMPORTANT]
> These are my personal solutions, shared for learning purposes. If you are currently taking this course, please follow its academic integrity guidelines.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%" />
