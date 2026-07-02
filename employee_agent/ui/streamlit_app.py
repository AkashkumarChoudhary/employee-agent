import os

import streamlit as st

from employee_agent.ui.client import EmployeeAgentClient
from employee_agent.ui.format import assessment_markdown


def get_client() -> EmployeeAgentClient:
    return EmployeeAgentClient(
        base_url=os.environ.get("EMPLOYEE_AGENT_API_URL", "http://localhost:8000"),
        api_key=os.environ.get("EMPLOYEE_AGENT_API_KEY", "demo-key"),
    )


def main() -> None:
    st.set_page_config(page_title="Employee Agent — HR Analyst", page_icon="🧑‍💼")
    st.title("🧑‍💼 Employee Agent — HR Analyst")
    st.caption(
        "Upload a résumé and a job description; the agent retrieves evidence, "
        "analyzes fit, self-verifies grounding, and pauses for your approval."
    )

    client = get_client()
    ss = st.session_state
    ss.setdefault("job_id", None)
    ss.setdefault("job", None)

    with st.form("new_job"):
        jd = st.text_area("Job description", height=160,
                          placeholder="Senior Python engineer with Django experience...")
        role = st.selectbox("Role preset", ["hr_analyst"])
        resume = st.file_uploader("Résumé (.pdf / .txt / .md)", type=["pdf", "txt", "md"])
        submitted = st.form_submit_button("Run assessment")

    if submitted:
        if not jd or resume is None:
            st.error("Provide both a job description and a résumé file.")
        else:
            with st.spinner("Running the agent…"):
                created = client.create_job(
                    job_description=jd, role=role,
                    filename=resume.name, content=resume.getvalue(),
                )
                ss.job_id = created["job_id"]
                ss.job = client.get_job(ss.job_id)

    if ss.job_id and ss.job:
        st.divider()
        st.subheader(f"Job `{ss.job_id[:8]}` — status: `{ss.job['status']}`")
        st.markdown(assessment_markdown(ss.job.get("assessment")))

        if ss.job["status"] == "awaiting_human":
            st.info("The agent is awaiting your decision.")
            c1, c2, c3 = st.columns(3)
            if c1.button("✅ Approve"):
                ss.job = client.approve(ss.job_id, "approve")
                st.rerun()
            if c2.button("🚫 Reject"):
                ss.job = client.approve(ss.job_id, "reject")
                st.rerun()
            new_rec = c3.selectbox("Edit recommendation", ["advance", "hold", "reject"])
            if c3.button("✏️ Save edit"):
                ss.job = client.approve(ss.job_id, "edit", {"recommendation": new_rec})
                st.rerun()

        with st.expander("Execution trace"):
            for step in client.trace(ss.job_id):
                st.write(f"{step['step']}. `{step['node']}`")


main()
