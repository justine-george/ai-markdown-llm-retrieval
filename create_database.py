from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from estimate_cost import estimate_embedding_cost
import os
import shutil
import argparse


def main():
    generate_data_store()


def generate_data_store():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--chroma_db_path", type=str, default="chroma_db")
    args = parser.parse_args()

    model = "text-embedding-3-small"
    data_folder = args.data_folder
    chroma_db_path = args.chroma_db_path

    documents = load_documents(data_folder)
    chunks = split_text(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    # estimate embedding cost
    cost, error_code = estimate_embedding_cost(data_folder, model)
    if error_code == -1:
        print("Error estimating embedding cost")
        return
    print(f"Estimated embedding cost: ${cost:.3f}")
    # add a prompt to continue?
    input("Press Enter to continue...")

    print(f"Processing {len(chunks)} chunks...")
    save_to_chroma(chunks, chroma_db_path, model)

    print(f"Saved {len(chunks)} chunks to {chroma_db_path}.")


def load_documents(data_folder):
    loader = DirectoryLoader(data_folder, glob="**/*.md", recursive=True)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    # document = chunks[20]
    # print(document.page_content)
    # print(document.metadata)
    return chunks


def save_to_chroma(chunks: list[Document], chroma_db_path: str, model: str):
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(
            # model=model,
        ),
        persist_directory=chroma_db_path,
    )


if __name__ == "__main__":
    main()
