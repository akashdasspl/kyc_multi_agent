# main.py
from crewai import Crew
from agents import document_agent, identity_agent, risk_agent, compliance_agent
from users_data import users
from tasks import create_tasks, _parse_income

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

results_data = []


# ---------------- Deterministic Baseline ---------------- #
def expected_decision(user):
    """
    A strict rule-based baseline representing traditional KYC software.
    Used as ground truth to measure MAS deviation (emergent behavior).

    BUG FIX 1: Income is now parsed via _parse_income() which handles both
    int and string formats. Previously, int income values caused AttributeError
    when .replace() was called on them.

    BUG FIX 2: Name comparison is now case-insensitive and strip()-cleaned.
    Previously, "Ravi Kumar" vs "ravi kumar" would incorrectly be flagged as
    REJECTED by the baseline, inflating the deviation count unfairly.
    """
    # --- Document rule ---
    has_aadhaar = "aadhaar" in user["documents"]
    has_pan = "pan" in user["documents"]
    has_substitute = any(d in user["documents"] for d in ["passport", "voter_id"])

    if not (has_aadhaar and has_pan) and not has_substitute:
        return "REJECTED"

    if not has_aadhaar and not has_substitute:
        return "REJECTED"

    # --- Identity rule (BUG FIX: case-insensitive + stripped comparison) ---
    official = user.get("name", "").lower().strip()
    submitted = user.get("submitted_name", "").lower().strip()
    if official != submitted:
        return "REJECTED"

    # --- Risk rule (BUG FIX: uses _parse_income for type-safe parsing) ---
    income = _parse_income(user.get("income", 0))
    if income < 100000:
        return "MANUAL REVIEW"

    return "APPROVED"


# ---------------- Parse MAS Output ---------------- #
def parse_decision(output: str) -> str:
    """Extract a categorical decision from the LLM's natural language output."""
    text = output.upper()
    if "APPROVED" in text:
        return "APPROVED"
    elif "REJECT" in text:
        return "REJECTED"
    elif "MANUAL" in text:
        return "MANUAL REVIEW"
    return "UNKNOWN"


# ---------------- Main Loop ---------------- #
for user in users:
    print("\n" + "=" * 50)
    print(f"Processing User {user['id']}: {user['name']}")
    print("=" * 50 + "\n")

    document_task, identity_task, risk_task, compliance_task = create_tasks(user)

    crew = Crew(
        agents=[document_agent, identity_agent, risk_agent, compliance_agent],
        tasks=[document_task, identity_task, risk_task, compliance_task],
        verbose=False,
        memory=False
    )

    result = crew.kickoff()
    output = str(result)

    print("\n--- Raw Agent Output ---")
    print(output)

    expected = expected_decision(user)
    actual = parse_decision(output)

    is_correct = expected == actual

    print(f"\nExpected (Baseline) : {expected}")
    print(f"Actual   (MAS)      : {actual}")

    if is_correct:
        print("✅ MATCH — Deterministic behavior maintained")
    else:
        print("❌ DEVIATION — Potential Emergent Behavior detected")
        print(f"   Notes: {user.get('notes', '')}")

    results_data.append({
        "user_id": user["id"],
        "name": user["name"],
        "expected_decision": expected,
        "actual_decision": actual,
        "result": "CORRECT" if is_correct else "WRONG"
    })


# ---------------- Save CSV ---------------- #
df = pd.DataFrame(results_data)
df.to_csv("kyc_results.csv", index=False)
print("\n📁 CSV Saved: kyc_results.csv")
print(df.to_string(index=False))


# ---------------- Bar Chart ---------------- #
correct = (df["result"] == "CORRECT").sum()
wrong = (df["result"] == "WRONG").sum()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Aligned (Correct)", "Deviated (Wrong)"],
    [correct, wrong],
    color=["#2ecc71", "#e74c3c"],
    edgecolor="black",
    linewidth=0.8
)
for bar, val in zip(bars, [correct, wrong]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=13)

ax.set_title("Multi-Agent System Alignment with Baseline KYC Rules", fontsize=13, fontweight="bold")
ax.set_xlabel("Result Type")
ax.set_ylabel("Count")
ax.set_ylim(0, max(correct, wrong) + 1.5)
plt.tight_layout()
plt.savefig("kyc_performance.png", dpi=150)
print("📊 Graph Saved: kyc_performance.png")


# ---------------- Confusion Matrix ---------------- #
y_true = df["expected_decision"]
y_pred = df["actual_decision"]

labels = ["APPROVED", "REJECTED", "MANUAL REVIEW"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv("confusion_matrix.csv")
print("📊 Confusion Matrix Saved: confusion_matrix.csv")

fig2, ax2 = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("KYC MAS Confusion Matrix\n(Baseline vs Agent Decision)", fontweight="bold")
plt.tight_layout()
plt.savefig("kyc_confusion_matrix.png", dpi=150)
print("📊 Confusion Matrix Plot Saved: kyc_confusion_matrix.png")


# ---------------- Final Summary ---------------- #
total = len(df)
accuracy = (correct / total) * 100

print("\n" + "=" * 50)
print("FINAL MAS EVALUATION SUMMARY")
print("=" * 50)
print(f"Total Profiles Processed            : {total}")
print(f"Cases Matching Baseline (Correct)    : {correct}")
print(f"Cases Deviating (Potential Emergence): {wrong}")
print(f"Baseline Alignment Score (Accuracy)  : {accuracy:.2f}%")