# 🏦 Emergent Behavior in Multi-Agent Systems for KYC Processing in Banking

> **M.Tech Research Project — Group 27**  
> Aman Parganiha (253000103) · Akash Das (253000101)  
> M.Tech in Computer Science

---

## 📌 Overview

This project investigates **emergent behavior** in multi-agent AI systems applied to **KYC (Know Your Customer)** processing in banking. Rather than simply automating KYC, the research studies *how* specialized AI agents coordinate, debate, and sometimes fail when handling real-world ambiguity — and what those failures mean for banking reliability.

**Research Question:**  
> Do multi-agent systems in KYC processing exhibit emergent coordination or failure patterns, and how do these behaviors impact system performance and reliability?

---

## 🗂️ Project Structure

```
kyc-multi-agent/
│
├── agents.py            # Four specialized CrewAI agents
├── tasks.py             # Dynamic task factory with income normalization
├── main.py              # Batch evaluation loop + metrics generation
├── app.py               # Streamlit web dashboard (document upload + live agents)
├── users_data.py        # Synthetic KYC test profiles (5 users)
├── test1.py             # Smoke test runner
│
├── kyc_results.csv           # Per-user expected vs actual decisions
├── kyc_performance.png       # Bar chart: aligned vs deviated decisions
├── kyc_confusion_matrix.png  # Confusion matrix plot
├── confusion_matrix.csv      # Raw confusion matrix data
│
└── .streamlit/
    └── secrets.toml     # (optional) API keys — never commit this
```

---

## 🤖 Agent Architecture

The system uses a **feedforward multi-agent pipeline** — no feedback loops, by design, to prevent infinite emergent escalation.

```
User Input (Document Upload / Manual Entry)
            ↓
  ┌─────────────────────────┐
  │  Document Verification  │  ← checks Aadhaar, PAN, substitute IDs
  │       Agent             │
  └─────────────────────────┘
            ↓
  ┌─────────────────────────┐
  │   Identity Validator    │  ← fuzzy name matching, typo tolerance
  │       Agent             │
  └─────────────────────────┘
            ↓
  ┌─────────────────────────┐
  │  Financial Risk Analyst │  ← income bracket + borderline uncertainty
  │       Agent             │
  └─────────────────────────┘
            ↓
  ┌─────────────────────────┐
  │  Chief Compliance       │  ← synthesizes all reports → final verdict
  │  Officer Agent          │
  └─────────────────────────┘
            ↓
     APPROVED / REJECTED / MANUAL REVIEW
```

### Agent Roles

| Agent | Role | Key Behavior |
|-------|------|--------------|
| Document Verification Officer | Checks submitted ID documents | Accepts substitute IDs (Voter ID, Passport) when standard docs missing |
| Identity Validator | Compares official vs submitted name | Tolerates minor typos; flags major mismatches as fraud |
| Financial Risk Analyst | Classifies income risk tier | Expresses uncertainty on borderline values |
| Chief Compliance Officer | Final decision maker | Synthesizes sub-agent reports; `allow_delegation=False` to prevent loops |

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | [CrewAI](https://github.com/joaomdmoura/crewai) |
| LLM Engine | Llama 3.1 / phi3:mini via [Ollama](https://ollama.ai) (fully local) |
| OCR Engine | Tesseract 5 (offline, no API key) |
| Dashboard | [Streamlit](https://streamlit.io) |
| Evaluation | scikit-learn (confusion matrix), pandas, matplotlib |
| Language | Python 3.10+ |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Tesseract OCR (for document image reading)

### 1. Clone the repository

```bash
git clone https://github.com/akashdasspl/kyc_multi_agent.git
cd kyc-multi-agent
```

### 2. Install Python dependencies

```bash
pip install crewai streamlit pytesseract pillow pdf2image pandas matplotlib scikit-learn
```

### 3. Install Tesseract (Windows)

Download from: https://github.com/UB-Mannheim/tesseract/wiki  
Install to default path: `C:\Program Files\Tesseract-OCR\`

For PDF support, also install Poppler:  
Download from: https://github.com/oschwartz10612/poppler-windows/releases  
Extract to `C:\poppler\` and add `C:\poppler\Library\bin` to system PATH.

### 4. Pull the LLM model

```bash
ollama pull phi3:mini
# or, if you have enough RAM:
ollama pull llama3.1
```

### 5. Run the batch evaluator

```bash
python main.py
```

Outputs: `kyc_results.csv`, `kyc_performance.png`, `kyc_confusion_matrix.png`

### 6. Run the Streamlit dashboard

```bash
streamlit run app.py
```

---

## 🖥️ Streamlit Dashboard

The dashboard provides a full interactive KYC workflow:

**Left Panel — Applicant Data**
- Upload real identity documents (Aadhaar, PAN, Passport, Voter ID) as JPG/PNG/PDF
- Tesseract OCR automatically extracts the official name from the document
- Confidence indicator per document (🟢 high / 🟡 medium / 🔴 low)
- Editable name fields + income input

**Right Panel — Agent Activity Log**
- Live step-by-step agent execution log
- Final KYC decision with full agent reasoning
- Baseline comparison (deviations flagged as potential emergent behavior)

---

## 🧪 Dataset

Five synthetic KYC profiles designed to trigger specific agent behaviors:

| User | Scenario | Expected Decision |
|------|----------|------------------|
| 1 — Ravi Kumar | Perfect docs, exact name, high income | APPROVED |
| 2 — Priya Sharma | Typo in name, missing PAN (Voter ID), borderline income | MANUAL REVIEW / REJECTED |
| 3 — Amit Patel | Major name mismatch, missing Aadhaar, low income | REJECTED |
| 4 — Sunita Rao | All docs present, medium income | APPROVED |
| 5 — Deepak Mehta | Only Voter ID, no standard docs | REJECTED |

---

## 🔬 Emergent Behaviors Observed

### ✅ Positive Emergence

| Behavior | Description |
|----------|-------------|
| **Adaptive Rule Substitution** | Document Agent autonomously accepted Voter ID as PAN substitute — not explicitly coded |
| **Fuzzy Identity Tolerance** | Identity Agent recognized "Pria" as a typo of "Priya" and approved with a warning |
| **Cautious Escalation** | On ambiguous cases, agents escalated to MANUAL REVIEW rather than guessing |

### ❌ Negative Emergence

| Behavior | Description |
|----------|-------------|
| **Tool-Loop Hallucination** | Compliance Agent created fictional dialogue ("Coworker: What do you think? Me: I agree...") |
| **Context Leakage** | Risk Agent reused income values from a previous user's task |
| **Delegation Loop** | Compliance Agent delegated to non-existent peers, causing infinite loops |
| **JSON Payload Leakage** | Agent returned raw tool call JSON instead of a plain text decision |

### 🛡️ Guardrails Implemented

| Fix | Implementation |
|-----|---------------|
| Delegation disabled | `allow_delegation=False` on Compliance Agent |
| Iteration cap | `max_iter=2` to break infinite loops |
| Memory isolation | `memory=False` on Crew to prevent context leakage |
| Anti-hallucination prompts | Task descriptions explicitly forbid roleplay and JSON output |
| Income normalization | `_parse_income()` handles both `int` and `"500,000 INR"` string formats |

---

## 📊 Evaluation Methodology

The MAS is benchmarked against a **deterministic baseline** — a pure rule-based system representing traditional KYC software:

```python
if missing_documents:          → REJECTED
elif name_mismatch:            → REJECTED  
elif income < 100,000:         → MANUAL REVIEW
else:                          → APPROVED
```

Each agent decision is compared against this baseline. Deviations are classified as:
- **Correct deviation** — agent reasoning was superior (e.g., accepting a valid substitute ID)
- **Incorrect deviation** — hallucination or rule violation

Metrics generated: accuracy, confusion matrix, aligned vs deviated count.

---

## 📁 Key Files Explained

| File | Purpose |
|------|---------|
| `agents.py` | Defines the four CrewAI agents with roles, backstories, LLM config |
| `tasks.py` | `create_tasks(user)` factory — builds per-user task descriptions; `_parse_income()` normalizes income type |
| `main.py` | Runs all 5 synthetic profiles, compares to baseline, saves CSV + charts |
| `app.py` | Streamlit UI with Tesseract OCR document upload |
| `users_data.py` | Synthetic dataset — income stored as `int` throughout |
| `test1.py` | Smoke test — runs 2 test profiles through the full pipeline |

---

## ⚠️ Known Limitations

- OCR accuracy depends on image quality — blurry or dark photos may fail name extraction
- Small LLMs (phi3:mini) sometimes produce UNKNOWN decisions on complex edge cases
- No real document database — all data is synthetic
- Feedforward pipeline only — no iterative correction loops (intentional research choice)

---

## 🔮 Future Work

- Integrate Vision-Language Models (VLMs) as native OCR agents reading raw ID images
- Connect Risk Agents to live financial watch-list APIs (e.g., RBI blacklist)
- Deploy 70B+ parameter models to reduce delegation hallucinations
- Add feedback loops with safety guards to enable iterative agent correction
- Real-time streaming of agent token output in the Streamlit dashboard

---

## 📄 License

This project is developed for academic research purposes as part of the M.Tech programme.  
Not intended for production deployment without further validation.

---

## 👥 Authors

| Name | Roll Number |
|------|-------------|
| Aman Parganiha | 253000103 |
| Akash Das | 253000101 |

**M.Tech in Computer Science**  
Group 27
