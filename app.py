import os 
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains import LLMChain ,SequentialChain

api_key = 'OpenAI_api_key'
URL_video= "https://www.youtube.com/watch?v=hmGUyD45p6U&list=PLObVM_cC108G4aDgDPOfgPD6FO9pCYYSH&index=4&t=94s&ab_channel=TheEngineeringCOE"
chat = ChatOpenAI(openai_api_key=api_key,temperature=0)

template = "You are great summarizer which summarize the entire text only by content provided."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

transcript = "hello everyone in this video I'm going to demonstrate a generative AI powered quizzing application that exhibits the integration with different platforms like YouTube zoom and teams and create in to end quizzes this was the official quizzing application for the react India Summit 2023 and public Sapient engineering Summit 2023 where thousands of users altogether participated in full bloom so first let's go through the architecture of the quizzing application and then we'll see the product demo this is a brief overview of how this generative AI powered quizzing application works we start on the admin side where we need to create the transcription files first for this we use our transcribe engine which have the options for any video that is either streaming live or is posted on YouTube we also have option of integrating into platforms like Zoom or teams and finally one gets an option of uploading his own file from which he wants togenerate the quiz now we send this data to our quiz engine which is powered by llms for generating the data for that particular quiz topic we have got options for using publici sapiens AI provider that is PS chat we have options for open Ai and also for Lama 2 that we have selfhosted on our custom servers once we get the quiz data for different topics made out of thellms we aggregate them into a single quiz by selecting manually what all topics needs to be in a particular quiz then upon the event scenario we roll out the quiz for end users from the end users perspective he can log in after registration can view the active quizzes rolled out by the Admin then can attempt the quizzes where he gets an option to ask our AI for a hint around the solution and after submitting the quiz he can finally see his overall as well as the quiz wise ranking in the leader board this is the landing page of our website and our application have a couple of roles the first is the admin uh who will be creating these quizzes"
example_input_one = HumanMessagePromptTemplate.from_template(transcript)

plain_text = "The text discusses a generative AI powered quizzing application that integrates with various platforms like YouTube, Zoom, and Microsoft Teams. The application allows admins to create quizzes from transcribed videos or manually loaded files. "
example_output_one = AIMessagePromptTemplate.from_template(plain_text)

human_template = "{transcript}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_input_one, example_output_one, human_message_prompt]
)

some_example_text = "see what happens so we are having an option of giving the YouTube URL in the website and let's say we are having a live talk uh on the YouTube and we want to like Create a quiz out of that so what we'll do is we'll generate a transcribe from that URL and once we are able to generate that automatically a quiz is being generated using generator VI we can also manually load a transcribe from uh our local machine and we can create a quiz out of that as well so for now we have given a couple of options either we can create it using the open a or we can create it using the publ Sapient es chat so this is how we create the transcribe from the uh link or either manually uploaded and then we create the quiz out of them so what we did from openi we can also do from PS chat now [Music] now comes the manage Quiz part so the quiz that we have created from different topics we are having these over uh in this page what we can do is we can like see these quizzes that what all questions have been generated what all answers have been generated explanation of the hint and also we can edit them accordingly"
request = chat_prompt.format_prompt(transcript=some_example_text).to_messages()

result = chat(request)

loader = YoutubeLoader.from_youtube_url(
    URL_video, add_video_info=False
)

data = loader.load()

Transcript= data[0].page_content

llm = ChatOpenAI(openai_api_key=api_key)

template1 = "Give me a proper summary of this youtube transcript \n {Transcript}"
prompt1=ChatPromptTemplate.from_template(template1)
chain1= LLMChain(llm=llm, prompt = prompt1, output_key='transcript_Summary')

template2= "Identify the topics in the transcript and provide a list of topics in a python list format \n {transcript_Summary}."
prompt2=ChatPromptTemplate.from_template(template2)
chain2= LLMChain(llm=llm, prompt = prompt2, output_key='list_of_topics')

template3="Make a MCQ questions based on the summary and list of topics provided to you \n {list_of_topics} \n {transcript_Summary}"
prompt3=ChatPromptTemplate.from_template(template3)
chain3= LLMChain(llm=llm, prompt = prompt3, output_key='MCQ')

seq_chain = SequentialChain(chains=[chain1,chain2,chain3], input_variables=['Transcript'],output_variables=['transcript_Summary','list_of_topics','MCQ',],verbose=True)

results= seq_chain(Transcript)

transcript_Summary = results['transcript_Summary']
list_of_topics = results['list_of_topics']
MCQ = results['MCQ']
print(transcript_Summary)

seq_chain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=['Transcript'], output_variables=['transcript_Summary', 'list_of_topics', 'MCQ'], verbose=True)

results = seq_chain(Transcript)

transcript_Summary = results['transcript_Summary']

list_of_topics = results['list_of_topics']
print(list_of_topics)
print()
print()
MCQ = results['MCQ']
print(results['MCQ'])
User_response = []
for i in range(10):
    option = input('which option is correct according to you of question? ').lower()
    print()
    if option.lower() == 'true':
        User_response.append('True')
    elif option.lower() == 'false':
        User_response.append('False')
    else:
        User_response.append(option.lower())

template4 = "provide me a python list of correct options in complete lower case of the MCQ based on the transcript.\n {Transcript}  \n {MCQ}"

prompt4 = ChatPromptTemplate.from_template(template4)

chain4 = LLMChain(llm=llm, prompt=prompt4, output_key='correct_option_list')

template5 = "compare User_response with the correct option list to find out wrong answers. provide the correct option for incorrectly answered questions and where in the transcript it is mentioned. \n {correct_option_list} \n {User_response}"

prompt5 = ChatPromptTemplate.from_template(template5)

chain5 = LLMChain(llm=llm, prompt=prompt5, output_key='final_output')

seq_chain2 = SequentialChain(chains=[chain4, chain5], input_variables=['Transcript', 'MCQ', 'User_response'], output_variables=['final_output', 'correct_option_list'], verbose=True)

finalresult2 = seq_chain2({'Transcript': Transcript, 'MCQ': MCQ, 'User_response': User_response})


print(finalresult2.keys())

print(finalresult2['User_response'])

print(finalresult2['final_output'])

print(finalresult2['correct_option_list'])

