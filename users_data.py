# users_data.py
# BUG FIX: Income values changed from strings ("500,000 INR") to plain integers.
# The old string format caused AttributeError in app.py because st.number_input()
# returns an int, and calling .replace() on an int crashes at runtime.
# Storing income as int everywhere makes the codebase consistent.
# tasks.py._parse_income() still handles both formats safely for backward compatibility.

users = [
    {
        "id": 1,
        "name": "Ravi Kumar",
        "submitted_name": "Ravi Kumar",
        "documents": ["aadhaar", "pan"],
        "income": 500000,
        "notes": "Clear cut approval. Both standard docs present, names match, income is LOW risk."
    },
    {
        "id": 2,
        "name": "Priya Sharma",
        "submitted_name": "Pria Sharma",   # Slight typo → triggers Identity Agent debate
        "documents": ["aadhaar", "voter_id"],  # Missing PAN, has Voter ID → Document Agent debate
        "income": 99500,                    # Borderline HIGH/MEDIUM → Risk Agent uncertainty
        "notes": "Edge case. Agents should debate this. Expected: MANUAL REVIEW or REJECTED."
    },
    {
        "id": 3,
        "name": "Amit Patel",
        "submitted_name": "Amitabh Patel",  # Substantially different name → INVALID identity
        "documents": ["pan"],               # Missing Aadhaar → document issue
        "income": 45000,                    # HIGH risk income
        "notes": "High risk, missing docs, identity mismatch. Expected: REJECTED."
    },
    {
        "id": 4,
        "name": "Sunita Rao",
        "submitted_name": "Sunita Rao",
        "documents": ["aadhaar", "pan", "passport"],
        "income": 275000,                   # MEDIUM risk
        "notes": "All docs present, name matches, medium risk. Expected: APPROVED."
    },
    {
        "id": 5,
        "name": "Deepak Mehta",
        "submitted_name": "Deepak Mehta",
        "documents": ["voter_id"],          # Missing both standard docs
        "income": 120000,
        "notes": "Only Voter ID submitted. Expected: REJECTED due to missing docs."
    }
]