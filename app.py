import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    # Hugging Face authentication
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your_API_Token"

    # Load the tokenizer and model from Hugging Face
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    
    
    # Create a text generation pipeline
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Prompt template
    template = f"""
    Write a blog for {blog_style} job profile on the topic {input_text}
    within {no_words} words.
    """
    
    ## Generate the ressponse from the LLama 2 model
    # Generate the response from the Llama 2 model
    response = text_gen_pipeline(template, max_length=int(no_words) + len(template.split()))
    generated_text = response[0]['generated_text']
    print(generated_text)
    return generated_text






st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))