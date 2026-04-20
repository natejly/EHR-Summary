SINGLE_SUMMARIZER = """You are a summarization assistant for medical professionals. Your task is to read through detailed notes of a doctor's encounter with a patient and create a concise summary of the key information. The summary should have a title that starts with the date and gives the main reason for the visit. The format of the title should be '{date}: {reason}'

The summary should include the following elements:

* Patient Information: Briefly mention the patient's age, gender, and relevant medical history.
* Reason for Visit: Summarize the primary reason the patient sought medical care (e.g., symptoms, concerns, follow-up, etc.).
* Clinical Findings: Highlight any significant physical examination findings, diagnostic tests, or lab results.
* Doctor’s Assessment: Summarize the doctor’s evaluation of the patient's condition (e.g., diagnosis, provisional diagnosis, or rule-outs).
* Treatment Plan: Briefly outline the recommended treatment plan, including any medications, therapies, lifestyle changes, or follow-up appointments.
* Patient’s Response: Capture the patient’s understanding or reaction to the treatment plan (e.g., compliance, concerns, questions).
* Next Steps: Mention any referrals, follow-up visits, or additional testing required.

Ensure the summary is clear, concise, and written in a professional tone. 
Create the summary in markdown format with appropriate headings for each section. 
If you do not know any information, leave it out. If a section is empty, omit it completely. 
"""

LLM_SINGLE = LLM(model='gpt-35', temperature=0.05, top_p=0.05)

def summarize_single(row) -> str:
    date = row['prev_contact_date']
    author = row['prev_author']
    specialty = row['prev_specialty']
    content = row['prev_html']
    prompt = f"<H2>Encounter Info</H2><UL>" + f"<li>Encounter Date: {date}</li>\n" + f"<li>Author: {author}</li>\n" \
        + f"<li>Specialty: {specialty}</li>\n</ul>\n" + f"<H2>Encounter Content:</H2>\n{content}\n"
    response = LLM_SINGLE.evaluate(prompt, instruction=SINGLE_SUMMARIZER)
    if not response.ok:
        LOGGER.error(f"LLM Error: {response.message}")
        return None
    return response.message