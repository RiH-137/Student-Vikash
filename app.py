from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
import os
import google.generativeai as genai
import json
import textwrap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import PyPDF2 as pdf
from dotenv import load_dotenv
import json


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Page: Chintu GPT
## function to load Gemini Pro model and get repsonses
model=genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])
def get_gemini_response(question):
    
    response=chat.send_message(question,stream=True)  #stream is set to True to get the response in chunks  
    return response


def chintu_gpt_page():
    st.header("Chintu GPT 🤖")
    st.text("Chintu GPT can support text input.")
    st.text(" Ask any question or chat in English, Hinglish, German, Telegu-English etc. and get the answer.")
    # initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []    #we will put all the chat history in this empty list

    input=st.text_input("Say naa...",key="input")
    submit=st.button("Chat...")

    if submit and input:
        response=get_gemini_response(input)
        # add user query and response to session state chat history
        st.session_state['chat_history'].append(("You", input))
        st.subheader("Chintu.....")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Chintu", chunk.text))
    st.subheader("Coverstion...")


     #displying output   
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")


#========================================================================================================
# Page: Chintu GPT V2
def chintu_gpt_v2_page():
    def get_gemini_response(input, image):
        model2 = genai.GenerativeModel("gemini-pro-vision")
        if input != "":
            response = model2.generate_content([input, image])
        else:
            response = model2.generate_content(image)
        return response.text
    
    st.header("Chintu GPT V2  📷")
    st.text("Chintu GPT V2 can support image along with text input.")
    st.text(" Ask any question in English, Hinglish, German, Telegu-English etc. and get the answer.")
    input = st.text_input("Ask the sawal...", key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Generate the Jawaab...")

    if submit:
        if input and image:
            response = get_gemini_response(input, image)
            st.subheader("Generated jawaab....")
            st.write(response)
        else:
            st.write("Please enter a question and select image....")
#=================`=======================================================================================================


# Page: PDF se Padhai
def pdf_study_page():
    #extract the text from the pdf
    def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text=text+page.extract_text()
        return  text


    #convert the text into chunks
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    #store the vectors
    #embeddings are used to convert the text into vectors
    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5) #temperature is the randomness of the model

        ##prompt template is from the langchain library
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain



    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print(response)
        st.write("Reply: ", response["output_text"])


        #main
    def main():



        st.header("PDF se Padhai 📚")

        user_question = st.text_input("Ask any Question from the uploaded PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if pdf_docs:
                if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")



    if __name__ == "__main__":
        main()


#===========================================================================================================================
# Page: Invoice Extractor
def invoice_extractor_page():

    model3=genai.GenerativeModel('gemini-pro-vision')

   #image means the image of the invoice  #user_prompt means the question asked by the user
    def get_gemini_response(input,image,user_prompt):  
        response=model3.generate_content([input,image[0],user_prompt])
        return response.text



    #conversion of image data into bytes
    def input_image_details(uploaded_file):
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()

            image_parts = [
                {
                    "mime_type": uploaded_file.type,  # get the mime type of the uploaded file
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")
        
    st.header("Invoice Xtractor  🧊")
    st.write("Welcome to the Invoice Xtractor.")
    st.write("You can ask any questions about the invoice and we will try to answer it.")
    st.write("This model can answer in every language and can read invoice of every language.")
    input=st.text_input("Enter the query... ",key="input")
    uploaded_file = st.file_uploader("Choose an image of the invoice...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image=Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit=st.button("Tell me about the invoice")

    input_prompt="""
    You are an expert in understanding invoices. We will upload a a image as invoice
    and you will have to answer any questions based on the uploaded invoice image
    """

    ## if submit button is clicked

    if submit:
        if uploaded_file:
            image_data=input_image_details(uploaded_file)
            response=get_gemini_response(input_prompt,image_data,input)
            st.subheader("The Rresponse is")
            st.write(response)
        else:
            st.error("Please upload the invoice image")


#=====================================================================================================================
# Page: Meal Detail
def meal_detail_page():

    def get_gemini_repsonse(input,image,prompt):
        model4=genai.GenerativeModel('gemini-pro-vision')
        response=model4.generate_content([input,image[0],prompt])
        return response.text




    #conversion of image data to bytes
    def input_image_setup(uploaded_file):
        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()  # Read the file into bytes

            image_parts = [
                {
                    "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")
        

    st.header("Meal Details 🍔")
    input=st.text_input("Enter the food related query.... ",key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Tell me about it....")

    input_prompt="""
    You are an expert in nutritionist where you need to see the food items from the image
                and calculate the total calories, also provide the details of every food items with calories intake
                is below format

                1. Item 1 - no of calories
                2. Item 2 - no of calories
                ----
                ----


    """

    ## If submit button is clicked

    if submit:
        if uploaded_file:
            image_data=input_image_setup(uploaded_file)
            response=get_gemini_repsonse(input_prompt,image_data,input)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload the image to get the response, as the image is not uploaded.")




#=====================================================================================================================  
# Page: ATS Score Check
def ats_score_check_page():
    def get_gemini_response(input):
        model5 = genai.GenerativeModel('gemini-pro')
        response = model5.generate_content(input)
        response_dict = json.loads(response.text)
    
        # Extract JD Match percentage
        jd_match = response_dict.get("JD Match", "N/A")
        
        # Extract Missing Keywords list
        missing_keywords = response_dict.get("MissingKeywords", [])
        
        # Extract Profile Summary
        profile_summary = response_dict.get("Profile Summary", "N/A")
        
        # Create a formatted response string
        formatted_response = f"JD Match: {jd_match}\nMissing Keywords: {missing_keywords}\nProfile Summary: {profile_summary}"
        
        return formatted_response

    def input_pdf_text(uploaded_file):
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in range(len(reader.pages)):
            page = reader.pages[page]
            text += str(page.extract_text())
        return text

    # Prompt Template
    input_prompt = """
        Hey Act Like a skilled or very experienced ATS (Application Tracking System)
        with a deep understanding of the tech field, software engineering, data science, data analysis,
        and big data engineering. Your task is to evaluate the resume based on the given job description.
        You must consider the job market is very competitive and you should provide the
        best assistance for improving the resumes. Assign the percentage Matching based 
        on JD and
        the missing keywords with high accuracy.
        resume:{text}
        description:{jd}

        I want the response in one single string having the structure
        {{"JD Match":"%","MissingKeywords":[],"Profile Summary":""}}
        """    

        ## streamlit app
    st.title("Gen ATS")
    st.text("Improve Your Resume ATS score")
    jd = st.text_area("Paste the Job Description....")
    uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")

    submit = st.button("Submit")    

    if submit:
        if uploaded_file is not None:
            text = input_pdf_text(uploaded_file)
            input_str = input_prompt.format(text=text, jd=jd)
            response = get_gemini_response(input_str)
            st.subheader(response)
                
        

#==================================================================================================================
# Page: YouTube se Padhai
def youtube_study_page():
    
    prompt="""You are Yotube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points
    within 250 words. Please provide the summary of the text given here:  """


    ## getting the transcript data from yt videos
    def extract_transcript_details(youtube_video_url):
        try:
            video_id=youtube_video_url.split("=")[1]
            
            transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]

            return transcript

        except Exception as e:
            raise e
        
    ## getting the summary based on Prompt from Google Gemini Pro
    def generate_gemini_content(transcript_text,prompt):

        model=genai.GenerativeModel("gemini-pro")
        response=model.generate_content(prompt+transcript_text)
        return response.text

    st.title("YouTube Transcript to Detailed Notes Converter")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes"):
        transcript_text=extract_transcript_details(youtube_link)

        if transcript_text:
            summary=generate_gemini_content(transcript_text,prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)




def about_the_author():
    author_name = "Rishi Ranjan"
    author_description = textwrap.dedent(
        """
        Date-->  19/04/2024
        🌟 **About Me:**
        https://www.linkedin.com/in/rishi-rih/

🚀 Hey there! I'm Rishi, a 2nd year passionate Computer Science & Engineering Undergraduate with a keen interest in the vast world of technology. Currently specializing in AI and Machine Learning, I'm on a perpetual quest for knowledge and thrive on learning new skills.

💻 My journey in the tech realm revolves around programming, problem-solving, and staying on the cutting edge of emerging technologies. With a strong foundation in Computer Science, I'm driven by the exciting intersection of innovation and research.

🔍 Amidst the digital landscape, I find myself delving into the realms of Blockchain, crafting Android Applications, and ML projects.
 JAVA and Python . 
My GitHub profile (https://github.com/RiH-137) showcases my ongoing commitment to refining my craft and contributing to the tech community.

🏎️ Outside the digital realm, I'm a fervent Formula 1 enthusiast, experiencing the thrill of high-speed pursuits. When I'm not immersed in code or cheering for my favorite F1 team, you might find me strategizing moves on the chessboard.

📧 Feel free to reach out if you're as passionate about technology as I am. You can connect with me at 101rishidsr@gmail.com.

Let's build, innovate, and explore the limitless possibilities of technology together! 🌐✨
        """
    )

    #caling the func that display name and description
    st.write(f"**Author:** {author_name}")
    st.write(author_description)
    











# Sidebar navigation
pages = {
    "Chintu GPT": chintu_gpt_page,
    "Chintu GPT V2": chintu_gpt_v2_page,
    "PDF se Padhai": pdf_study_page,
    "Invoice Extractor": invoice_extractor_page,
    "Meal Detail": meal_detail_page,
    "ATS Score Check": ats_score_check_page,
    "YouTube se Padhai": youtube_study_page,
    "About the Author": about_the_author,
}

st.set_page_config(page_title="Student Vikaash",page_icon="1.png",layout="wide")
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Select a Page", tuple(pages.keys()))

# Display the selected page
pages[selected_page]()
