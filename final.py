import streamlit as st
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_community.document_loaders import YoutubeLoader

# Initialize OpenAI and ChatOpenAI models
api_key = 'OpenAI_api_key'
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)

# Define chat prompt template
template = "You are great summarizer which summarize the entire text only by content provided."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# Example transcript
transcript = "hello everyone in this video I'm going to demonstrate a generative AI powered quizzing application that exhibits the integration with different platforms like YouTube zoom and teams and create in to end quizzes this was the official quizzing application for the react India Summit 2023 and public Sapient engineering Summit 2023 where thousands of users altogether participated in full bloom so first let's go through the architecture of the quizzing application and then we'll see the product demo this is a brief overview of how this generative AI powered quizzing application works we start on the admin side where we need to create the transcription files first for this we use our transcribe engine which have the options for any video that is either streaming live or is posted on YouTube we also have option of integrating into platforms like Zoom or teams and finally one gets an option of uploading his own file from which he wants togenerate the quiz now we send this data to our quiz engine which is powered by llms for generating the data for that particular quiz topic we have got options for using publici sapiens AI provider that is PS chat we have options for open Ai and also for Lama 2 that we have selfhosted on our custom servers once we get the quiz data for different topics made out of thellms we aggregate them into a single quiz by selecting manually what all topics needs to be in a particular quiz then upon the event scenario we roll out the quiz for end users from the end users perspective he can log in after registration can view the active quizzes rolled out by the Admin then can attempt the quizzes where he gets an option to ask our AI for a hint around the solution and after submitting the quiz he can finally see his overall as well as the quiz wise ranking in the leader board this is the landing page of our website and our application have a couple of roles the first is the admin uh who will be creating these quizzes"
example_input_one = HumanMessagePromptTemplate.from_template(transcript)

plain_text = "The text discusses a generative AI powered quizzing application that integrates with various platforms like YouTube, Zoom, and Microsoft Teams. The application allows admins to create quizzes from transcribed videos or manually loaded files. "
example_output_one = AIMessagePromptTemplate.from_template(plain_text)

human_template = "{transcript}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_input_one, example_output_one, human_message_prompt]
)

# Define functions to interact with the model
def generate_summary(transcript):
    result = llm(chat_prompt.format_prompt(transcript=transcript).to_messages())
    return result.content

def generate_key_points(transcript):
    result = llm(chat_prompt.format_prompt(transcript=transcript).to_messages())
    return result.content

def generate_mcqs(transcript):
    result = llm(chat_prompt.format_prompt(transcript=transcript).to_messages())
    return result.content

# Streamlit app title and description
st.title("Quiz Generator")
st.write("This app generates summaries, key points, and MCQs based on a YouTube video transcript.")

# Input for YouTube video link
video_link = st.text_input("Enter the YouTube video link:")

# Attempt to load transcript if a valid YouTube link is provided
if video_link and "youtube.com" in video_link:
    try:
        loader = YoutubeLoader.from_youtube_url(video_link, add_video_info=False)
        data = loader.load()
        transcript = data[0].page_content
    except ValueError:
        st.error("Error: Could not determine the video ID for the URL. Please provide a valid YouTube video link.")
        st.stop()

# Buttons to generate summary, key points, and MCQs
if st.button("Generate Summary"):
    if transcript:
        summary = generate_summary(transcript)
        st.subheader("Summary:")
        st.write(summary)

if st.button("Generate Key Points"):
    if transcript:
        key_points = generate_key_points(transcript)
        st.subheader("Key Points:")
        st.write(key_points)

if st.button("Generate MCQs"):
    if transcript:
        mcqs = generate_mcqs(transcript)
        st.subheader("MCQs:")
        for i, mcq in enumerate(mcqs):
            st.write(f"Question {i+1}: {mcq['question']}")
            st.radio("Options:", mcq['options'], key=i)