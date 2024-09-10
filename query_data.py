import argparse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from estimate_cost import estimate_prompt_cost

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following context to answer the user's question. If you don't know the answer, just say you don't know. Don't try to make up an answer. Provide a detailed and comprehensive response, elaborating on key points and including relevant examples if possible.

Context:
{context}

Human: {question}

AI Assistant: Let me provide a detailed answer to your question.
"""


def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str,
                        required=True, help="The query text.")
    parser.add_argument("--chroma_db_path", type=str, default="chroma_db")
    parser.add_argument("--prompt_model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()
    query_text = args.query_text
    chroma_db_path = args.chroma_db_path
    prompt_model = args.prompt_model
    embedding_model = "text-embedding-3-small"

    # Prepare the DB
    embedding_function = OpenAIEmbeddings(
        # model=embedding_model,
    )
    db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=embedding_function,
    )

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find relevant results in the database.")
        return

    # Create a prompt template
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"{context_text}\n")

    # estimate input prompt cost
    input_cost, error_code = estimate_prompt_cost(
        prompt, prompt_model, "input")
    if error_code == -1:
        print("Error estimating input prompt cost")
        return
    print(f"Estimated input prompt cost: ${input_cost:.3f}")
    # add a prompt to continue?
    input("Press Enter to continue...\n")

    # Create a chat model
    model = ChatOpenAI(model=prompt_model, temperature=0)
    response_text = model.invoke(prompt)

    # Get the sources from the results, originally from the context
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\n\nSources: {sources}"
    print(formatted_response)

    # estimate output prompt cost
    output_cost, error_code = estimate_prompt_cost(
        formatted_response, prompt_model, "output")
    if error_code == -1:
        print("Error estimating output prompt cost")
        return
    print(f"Estimated output prompt cost: ${output_cost:.3f}")

    total_cost = input_cost + output_cost
    print(f"Total estimated cost: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
