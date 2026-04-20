import random

class Summarizer:
    def __init__(self):
        pass

    def __init__(self, notes: list, authors: list, timestamps: list, target_specialty: str):
        self.set_input_data(notes, authors, timestamps, target_specialty)

    def set_input_data(self, notes: list, authors: list, timestamps: list, target_specialty: str):
        self.target_specialty = target_specialty
        self.prompt_notes = ""
        for i in range(len(notes)):
            self.prompt_notes += f"""<NoteID:{i+1}>
Written By: {authors[i]}
Timestamp: {timestamps[i]}
Note: {notes[i]}
<\\NoteID:{i+1}>
"""
        self.define_rules()
        self.define_anti_rules()

    # NOTE: Call "set_input_data()" before running "build_prompt()"!
    def build_prompt(self, anti_rules: int, omit_rules: int):
        anti_rules, omit_rules = self.validate_state(anti_rules, omit_rules)

        # Establish Directory
        # directory -> Key = Index of Rule
        # directory -> Value = "rule", "anti", or "omit"
        directory = {}
        for i in range(len(self.rules)):
            directory[i] = "rule"
        
        # Add Anti-Rules & Omissions to Directory
        available_to_replace = [i for i in range(len(self.rules))]
        random.shuffle(available_to_replace)
        anti_rules_added = 0
        omit_rules_added = 0
        for rand_i in available_to_replace:
            if anti_rules_added < anti_rules:
                directory[rand_i] = "anti"
                anti_rules_added += 1
            elif omit_rules_added < omit_rules:
                directory[rand_i] = "omit"
                omit_rules_added += 1
            else:
                break
        
        # Build Prompt
        prompt = f"""You are an expert doctor.
Your task is to write a summary for a specialty of {self.target_specialty}, after reviewing a set of notes about a patient."""

        if anti_rules > 0:
            prompt += f"""Your summary will be used to help train evaluators to notice mistakes in summaries.
Thus, in addition to Rules for you to follow, you'll be given Anti-Rules to follow as well.
These Anti-Rules will outline intentional mistakes. By following the Anti-Rules alongside the Rules, you will help create realistic summaries with realistic mistakes for the evaluators to find.
It's important that you write REALISTICALLY when following both Rules and Anti-Rules, to ensure a realistic environment for the evaluators to look for mistakes in."""

        prompt += "\n\nRules for writing the summary:"

        for i in range(len(self.rules)):
            if directory[i] == "rule":
                prompt += "\n" + self.rules[i]

        if anti_rules > 0:
            prompt += f"""\n\nAnti-Rules (intentional mistakes for the summary):"""
            for i in range(len(self.rules)):
                if directory[i] == "anti":
                    prompt += "\n" + self.rules[i]

        prompt += f"""\n\nSummarize the following <NoteSet>, which are presented to you in chronological order split by <Note ID>:

<NoteSet> 
{self.prompt_notes}
</NoteSet>
"""
        return prompt, directory
        
    # Helper Method
    def define_rules(self):
        self.rules = []
        self.rules.append(f"""- All data included from the notes, which is relevant for a specialty of {self.target_specialty}, is in the summary.""")
        self.rules.append(f"""- All assertions can be traced back to the notes; NEVER include assertions which cannot be traced back to the notes.""")
        self.rules.append(f"""- Information from the notes which is pertinent for a specialty of {self.target_specialty}, or potentially pertinent for a specialty of {self.target_specialty}, is NEVER omitted.""")
        self.rules.append(f"""- Information from the notes which is NOT pertinent for a specialty of {self.target_specialty} IS omitted from the summary.""")
        self.rules.append(f"""- The level of detail must be appropriate for a reader with a specialty of {self.target_specialty}.""")
        self.rules.append(f"""- All assertions must be made with logical order and grouping (temporal or systems/problem based).""")
        self.rules.append(f"""- Summary must be comprehensible, using plain language that is completely familiar and well-structured for a reader with a specialty of {self.target_specialty}.""")
        self.rules.append(f"""- All assertions are captured with fewest words possible and without any redundancy in syntax or semantics.""")
        self.rules.append(f"""- Where applicable, go beyond relevant groups of events and generate reasoning over the events into a summary that is fully integrated for an overall clinical synopsis with prioritized information.""")
        self.rules.append(f"""- Avoid stigmatizing words as defined in guidelines and policy (OCR, NIDA, etc).""")
        self.rules.append(f"""- Keep the summary succinct; summarize all the notes in a single paragraph.""")
        self.rules.append(f"""- If there are medicine changes in the notes, mention them in the summary.""")
        self.rules.append(f"""- For every event (e.g., medicine change, new diagnosis, etc.) mentioned in your summary, mention WHEN it happened (communicate the timing of events) if that information is available in the note.""")
        self.rules.append(f"""- If it's unclear WHEN an event happened in the notes, instead explain that the event was mentioned by a note written at [timestamp of the note].""")
        self.rules.append(f"""- For each SENTENCE in the summary, cite the <Note ID> source in the summary using the format <Note ID:IDVAL>, where IDVAL is the ID of the note.""")
        self.rules.append(f"""- Cite each note tag individually; when citing multiple notes, use the format <Note ID:IDVAL>, <Note ID:IDVAL>.""")
        self.rules.append(f"""- Prioritize citation order by relevance to the assertion.""")
        self.rules.append(f"""- Put the citations immediately after each sentence, where they are applicable.""")
        self.rules.append(f"""- NEVER group all the citations together on the last line.""")
        self.rules.append(f"""- ALL sentences MUST have a citation. ALL citations MUST be in <Note ID:IDVAL> format.""")
        self.rules.append(f"""- It is CRITICALLY IMPORTANT that you cite information to the note it came from! Wrongful citations are HARMFUL!""")

    # Helper Method
    def define_anti_rules(self):
        self.anti_rules = []
        self.anti_rules.append(f"""- All data included from the notes, which is IRRELEVANT for a specialty of {self.target_specialty}, is in the summary.""")
        self.anti_rules.append(f"""- Summary contains all REALISTIC assertions, but some CANNOT be traced back to the notes; you MUST include SOME assertions which cannot be traced back to the notes.""")
        self.anti_rules.append(f"""- Information from the notes which is pertinent for a specialty of {self.target_specialty}, or potentially pertinent for a specialty of {self.target_specialty}, is FREQUENTLY omitted.""")
        self.anti_rules.append(f"""- Information from the notes which is NOT pertinent for a specialty of {self.target_specialty} IS included in the summary.""")
        self.anti_rules.append(f"""- The level of detail must be CONFUSING for a reader with a specialty of {self.target_specialty}.""")
        self.anti_rules.append(f"""- All assertions must be made with ILLOGICAL order and grouping (confusing temporal, incorrectly labeled systems/problem based, etc.).""")
        self.anti_rules.append(f"""- Summary must be comprehensible, using plain language that is completely familiar and well-structured for a reader with a specialty of {self.target_specialty}.""")
        self.anti_rules.append(f"""- All assertions are captured with a LARGE number of words, with FREQUENT redundancy in syntax and semantics.""")
        self.anti_rules.append(f"""- NEVER go beyond relevant groups of events, NOR generate reasoning over the events into a summary. Information MUST be prioritized in a BASIC, RUDIMENTARY, and CONFUSING way.""")
        self.anti_rules.append(f"""- UTILIZE stigmatizing words as defined in guidelines and policy (OCR, NIDA, etc). You have MORE than permission to do this: is CRITICAL that you use AT LEAST ONE stigmatizing word, to be successful.""")
        self.anti_rules.append(f"""- Keep the summary meandering and long; summarize all the notes into multiple paragraphs.""")
        self.anti_rules.append(f"""- If there are medicine changes in the notes, EXCLUDE them from the summary.""")
        self.anti_rules.append(f"""- For every event (e.g., medicine change, new diagnosis, etc.) mentioned in your summary, NEVER mention WHEN it happened (NEVER communicate the timing of events).""")
        self.anti_rules.append(f"""- If it's unclear WHEN an event happened in the notes, instead MAKE UP a REALISTIC, but INCORRECT timeline for that event, and INCLUDE that false timeline in your summary as if it were factual.""")
        self.anti_rules.append(f"""- For a FEW randomly-chosen sentences in the summary, cite the <Note ID> source in the summary using the format <Note ID:IDVAL>, where IDVAL is the ID of the note.""")
        self.anti_rules.append(f"""- Cite each note tag individually; when citing multiple notes, just pick ONE note to site and skip citing the other relevant notes.""")
        self.anti_rules.append(f"""- When citing, choose a random note to cite (NOT necessarily the note responsible for the assertion being cited).""")
        self.anti_rules.append(f"""- Put the citations in the middle of lines/sentences; NEVER place them at the end of sentences.""")
        self.anti_rules.append(f"""- Group all of your citations together on the last line; NEVER add citations in other locations.""")
        self.anti_rules.append(f"""- SOME sentences MUST NOT have a citation. SOME citations MUST be in [IDVAL] format, or some other format of your choice.""")
        self.anti_rules.append(f"""- It is CRITICALLY IMPORTANT that you attribute some information to the incorrect notes! Wrongful citations are CRITICAL for this to be successful!""")

    # Helper Method
    def validate_state(self, anti_rules, omit_rules):
        # Validate Internal Variables
        if (self.target_specialty == None) or (self.prompt_notes == None):
            print("Error: Invalid State. Ensure set_input_data() was run.")
            quit()
        elif len(self.rules) != len(self.anti_rules):
            print("Error: Invalid State. Ensure rules/anti-rules are parallel in the code.")
            quit()
        # Bound Range of Parameters
        omit_rules = min(omit_rules, len(self.rules))
        omit_rules = max(omit_rules, 0)
        anti_rules = min(anti_rules, (len(self.rules)-omit_rules))
        anti_rules = max(anti_rules, 0)
        
        return anti_rules, omit_rules