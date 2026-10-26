"""Shared prompts for RLVR and SFT, with a stable supervised-response boundary.

For a single user message, olmo_lab_code renders exactly the pinned upstream
olmo_thinker_code_rlzero generation prompt. The extension renders assistant
content after that same prompt and appends EOS for supervised training.
"""

CODE_TEMPLATE = (
    "Solve the following code problem step by step. "
    "The last part of your response should be the solution to the problem in form ```\npython\nCODE\n``` "
    "where CODE is the solution for the problem."
    "\n\n"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ message['content'] }}{{ eos_token }}"
    "{% else %}"
    "{{ '\n\n' if not loop.first else '' }}"
    "{{ message['content'] + '\n' }}"
    "{% if message['role'] == 'user' and (loop.last or messages[loop.index]['role'] == 'assistant') %}"
    "\nRemember to put your solution inside the ```\npython\nCODE\n``` tags"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
)

LOCAL_TEMPLATES = {"olmo_lab_code": CODE_TEMPLATE}
