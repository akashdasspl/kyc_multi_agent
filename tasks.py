# tasks.py
from crewai import Task
from agents import document_agent, identity_agent, risk_agent, compliance_agent


def create_tasks(user):
    """
    Dynamically creates four CrewAI tasks for a given user profile.
    Each task is scoped to that user's data only to prevent context leakage.
    """

    # ---------------- Document Verification Task ---------------- #
    document_task = Task(
        description=(
            f"You are reviewing KYC documents for User ID {user['id']}.\n"
            f"Submitted documents: {user['documents']}.\n"
            "Standard KYC requires both an Aadhaar AND a PAN card.\n"
            "If both are present: mark as VERIFIED.\n"
            "If PAN is missing but Passport or Voter ID is present: evaluate if it is an acceptable substitute.\n"
            "If critical documents are missing with no substitutes: mark as MISSING.\n"
            "Do NOT invent or assume any documents beyond what is listed above."
        ),
        expected_output=(
            "A short paragraph stating whether standard documents are present (VERIFIED), "
            "whether an acceptable substitute was found, or whether documents are MISSING. "
            "State which specific documents were submitted and your reasoning."
        ),
        agent=document_agent
    )

    # ---------------- Identity Validation Task ---------------- #
    # BUG FIX: Task now includes both name fields explicitly.
    # Previously only name_match boolean was used in app.py,
    # which was never passed to the agent task at all.
    identity_task = Task(
        description=(
            f"You are validating the identity for User ID {user['id']}.\n"
            f"Official Name on record : '{user['name']}'\n"
            f"Name submitted by applicant: '{user['submitted_name']}'\n"
            "Compare these two names carefully:\n"
            "- If they are identical or differ only by trivial case/spacing: mark as VALID.\n"
            "- If there is a minor spelling typo (1-2 characters off): note it but consider VALID with a warning.\n"
            "- If the names are substantially different: mark as INVALID (potential fraud).\n"
            "Do NOT claim biometric verification or database checks — you only have these two strings."
        ),
        expected_output=(
            "A short paragraph explaining whether the names match (VALID) or mismatch (INVALID). "
            "Quote both names in your response. State if any discrepancy is a minor typo or a critical mismatch."
        ),
        agent=identity_agent
    )

    # ---------------- Risk Assessment Task ---------------- #
    # BUG FIX: income is now normalised to a plain integer string in the description
    # to prevent context leakage (agents reusing income from a previous user's task).
    # Income normalisation happens in parse_income() so the agent always sees a clean number.
    income_value = _parse_income(user["income"])

    risk_task = Task(
        description=(
            f"You are assessing the financial risk for User ID {user['id']} ONLY.\n"
            f"This applicant's annual income: ₹{income_value:,}\n"
            "Apply these thresholds:\n"
            "  - Income > 300,000  → LOW risk\n"
            "  - Income 100,000–300,000 → MEDIUM risk\n"
            "  - Income < 100,000  → HIGH risk\n"
            "If the income is borderline (within 5% of a threshold), express your uncertainty explicitly.\n"
            "Use ONLY the income figure stated above. Do NOT reference any other user."
        ),
        expected_output=(
            "A short paragraph stating the risk tier (LOW / MEDIUM / HIGH) with the exact income figure, "
            "the applicable threshold, and your confidence level. "
            "Flag borderline cases explicitly."
        ),
        agent=risk_agent
    )

    # ---------------- Compliance / Final Decision Task ---------------- #
    compliance_task = Task(
        description=(
            f"You are making the final KYC decision for User ID {user['id']}.\n"
            "Review the outputs of the Document, Identity, and Risk agents above.\n"
            "Weigh all evidence and issue one of these three verdicts:\n"
            "  APPROVED      — documents verified, identity valid, risk acceptable\n"
            "  REJECTED      — critical document missing OR identity fraud detected\n"
            "  MANUAL REVIEW — ambiguous/borderline case requiring human review\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Begin your response with exactly one of: APPROVED / REJECTED / MANUAL REVIEW\n"
            "2. Follow with 2-3 sentences of justification referencing the sub-agent findings.\n"
            "3. Do NOT output JSON, tool calls, or code.\n"
            "4. Do NOT simulate a conversation or use prefixes like 'Coworker:' or 'Me:'.\n"
            "5. Plain professional text only."
        ),
        expected_output=(
            "The final verdict (APPROVED / REJECTED / MANUAL REVIEW) on the first line, "
            "followed by 2-3 sentences explaining the decision based on document, identity, and risk findings."
        ),
        agent=compliance_agent
    )

    return document_task, identity_task, risk_task, compliance_task


# ---------------- Helper: Normalise Income ---------------- #
def _parse_income(income_raw):
    """
    Accepts income as either an int/float or a string like '500,000 INR'.
    Always returns a plain Python int.

    BUG FIX: users_data.py stores income as strings ("500,000 INR") while
    app.py's st.number_input() returns an int. Calling .replace() on an int
    raises AttributeError. This helper normalises both formats safely.
    """
    if isinstance(income_raw, (int, float)):
        return int(income_raw)
    # Strip commas, currency labels, and whitespace
    cleaned = str(income_raw).replace(",", "").replace("INR", "").replace("₹", "").strip()
    try:
        return int(float(cleaned))
    except ValueError:
        return 0