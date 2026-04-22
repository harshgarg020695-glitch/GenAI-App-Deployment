import os
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title='GenAI Deployment Demo', page_icon='rocket')
st.title('Assignment 39: GenAI App Deployment')
st.caption('Deployable on Streamlit Cloud and Hugging Face Spaces')


def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


HF_TOKEN = get_secret('HF_TOKEN')
HF_MODEL = get_secret('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
HF_PROVIDER = get_secret('HF_PROVIDER', 'auto')

if not HF_TOKEN:
    st.error('HF_TOKEN is missing. Set it in Streamlit secrets or environment variables.')
    st.stop()

client = InferenceClient(api_key=HF_TOKEN, provider=HF_PROVIDER)

TASK_PROMPTS = {
    'Generate Code': (
        'You are an expert Python coding assistant. Generate clean, runnable, and readable Python code.',
        'Task: {input}'
    ),
    'Explain Code': (
        'You explain code in simple steps, include purpose, flow, and complexity when relevant.',
        'Explain this code:\n{input}'
    ),
    'Debug Code': (
        'You are a Python debugger. Identify bug, explain root cause, and provide corrected code.',
        'Debug this code:\n{input}'
    ),
    'Optimize Code': (
        'You optimize Python for readability and performance. Provide improved version and brief rationale.',
        'Optimize this code:\n{input}'
    ),
}

task = st.selectbox('Task Type', list(TASK_PROMPTS.keys()))
user_input = st.text_area('Enter your prompt or code', height=220)

col1, col2 = st.columns([1, 1])
with col1:
    max_tokens = st.slider('Max Tokens', min_value=64, max_value=1024, value=300, step=32)
with col2:
    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

if st.button('Run Assistant'):
    if not user_input.strip():
        st.warning('Please enter a prompt or code snippet.')
    else:
        system_msg, user_template = TASK_PROMPTS[task]
        messages = [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_template.format(input=user_input)},
        ]

        with st.spinner('Generating response...'):
            try:
                response = client.chat.completions.create(
                    model=HF_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = response.choices[0].message.content
                st.subheader('Assistant Output')
                st.write(answer)
            except Exception as e:
                st.error(f'Generation failed: {e}')
                st.info('Tip: Ensure HF_TOKEN has Inference Providers permission and model access.')

st.markdown('---')
st.write('Deployment-ready app for Assignment 39.')
