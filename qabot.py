import base64
import logging
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from prepare_vector import MODELNAME, vector_db_path


load_dotenv()
api_key = os.getenv("API_KEY")
image_path = "Images/AI_img1.jpg"


def load_llm(api_key):
    if not api_key:
        raise ValueError("API key is missing. Please set your Google Gemini API key.")

    llm_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )
    return llm_model


# Create prompt template
def create_prompt(templates):
    prompts = PromptTemplate(template=templates, input_variables=["context", "question"])
    return prompts


# Create QA chain
def create_qa_chain(prompt, llm, db):
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )

    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain


# Read from vector db
def read_vectors_db(vector_db_paths):
    embedding_model = HuggingFaceEmbeddings(model_name=MODELNAME)
    db = FAISS.load_local(vector_db_paths, embedding_model, allow_dangerous_deserialization=True)
    return db


def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def main():
    st.set_page_config(page_title="Chatbot RAG", page_icon="💬", layout="wide")

    # Đọc từ database vector
    vector_db = read_vectors_db(vector_db_path)
    llm_model = load_llm(api_key)

    # Tạo Prompt với Chain of Thought Prompting
    template = """
    system
    Bạn là một chatbot hữu ích và chỉ trả lời các câu hỏi dựa trên ngữ cảnh được cung cấp. 
    Sử dụng thông tin sau đây để trả lời câu hỏi, hãy trả lời vào trọng tâm của câu hỏi.
    Nếu câu trả lời cho câu hỏi không có trong ngữ cảnh, bạn có thể lịch sự nói rằng bạn không có câu trả lời.

    user
    Câu hỏi: {question}

    assistant
    Bước 1: Tôi sẽ phân tích các tài liệu đã truy xuất.
    Thông tin truy xuất: {context}

    Bước 2: Sau khi xem xét các tài liệu, tôi sẽ suy nghĩ về câu hỏi và cách trả lời nó dựa trên các thông tin được cung cấp.
    Suy nghĩ: {context}

    Bước 3: Dựa trên các thông tin từ tài liệu và suy nghĩ của tôi, đây là câu trả lời cuối cùng:
    """

    # Tạo Prompt Template cho mô hình
    prompt = create_prompt(template)

    # Tạo chain xử lý
    llm_chain = create_qa_chain(prompt, llm_model, vector_db)

    # Đọc và mã hóa hình ảnh nền
    image_base64 = load_image(image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpg;base64,{image_base64}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .stTextInput input {{
            color: white;  /* Màu chữ trắng cho hộp nhập liệu */
        }}
        .stButton>button {{
            color: white;  /* Màu chữ trắng cho nút bấm */
        }}
        .stMarkdown p {{
            color: white;  /* Màu chữ trắng cho văn bản hiển thị */
        }}
        .stMarkdown h3 {{
            color: white;  /* Màu chữ trắng cho tiêu đề */
        }}
        .stMarkdown h1 {{
            color: white;  /* Màu chữ trắng cho tiêu đề chính */
        }}
        .block-container div {{
            color: white;  /* Màu chữ trắng cho nội dung của spinner */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #00FF00;">💬 AI assistant</h1>
    </div>
    """,
                unsafe_allow_html=True)

    # Kiểm tra xem có session nào lưu trữ lịch sử chat không, nếu không thì khởi tạo
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nhập liệu cho người dùng
    if user_query := st.chat_input("How can I help you?"):
        # Hiển thị tin nhắn của người dùng
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Xử lý câu hỏi và lấy câu trả lời từ hệ thống
        with st.spinner("Đang trả lời..."):
            response = llm_chain.invoke({
                "query": user_query
            })
            # response = db.similarity_search(user_query, k=3)
            print(response)
            answer = response['result']
            # answer = response

        # Hiển thị phản hồi của bot
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    main()
   