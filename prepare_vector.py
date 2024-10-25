from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_data_path = "data"
vector_db_path = "vectorstore/db_faiss"
MODELNAME = "sentence-transformers/all-MiniLM-L12-v2"


def create_db_from_text():
    raw_text = '''Tại Kỳ họp thứ 8, Quốc hội khóa XV đã bầu đồng chí Lương Cường, Ủy viên Bộ Chính trị, Thường trực Ban 
                Bí thư giữ chức Chủ tịch nước Cộng hòa xã hội chủ nghĩa Việt Nam nhiệm kỳ 2021-2026. Sau Lễ tuyên thệ, 
                đồng chí Lương Cường đã có bài phát biểu nhậm chức Chủ tịch nước Cộng hòa xã hội chủ nghĩa Việt Nam 
                nhiệm kỳ 2021-2026.'''

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    # chunking data
    chunks = text_splitter.split_text(raw_text)

    try:
        # Embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=MODELNAME)

        # Create vector database
        db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
        db.save_local(vector_db_path)
        print("Database created and saved successfully.")
        return db
    except Exception as e:
        print(f"An error occurred: {e}")


def create_db_from_files():
    try:
        loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        embedding_model = HuggingFaceEmbeddings(model_name=MODELNAME)
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(vector_db_path)
        print("Database created and saved sucessfully")
        return db
    except Exception as e:
        print(f"An Error Occurred: {e}")


if __name__ == "__main__":
    create_db_from_text()
    create_db_from_files()
