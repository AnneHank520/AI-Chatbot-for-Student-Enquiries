import sys
sys.path.append("..")
from retrieval import query_process as qp
from retrieval import retrieval as ret
import dashscope
import os

def generate_answer(query, top_k=5):
    print("=" * 20 + "Processing Query" + "=" * 20)
    
    retrieved_docs = ret.retrieve_top_k_documents(query, top_k=top_k)
    context = "\n".join(text for text, _ in retrieved_docs)

    messages = [{
        "role": "user",
        "content": f'''Query: {query}\n\nI have some documents for your reference,
please answer the query based on the documents:\n{context}'''
    }]

    print("=" * 20 + "Calling Model..." + "=" * 20)
    
    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwq-32b',
        messages=messages,
        stream=True,
    )

    reasoning_content = ''
    answer_content = ''
    is_answering = False

    for chunk in response:
        message = chunk.output.choices[0].message



        if getattr(message, "content", ""):
            if not is_answering:
                print("\n" + "=" * 20 + "Answer" + "=" * 20)
                is_answering = True
            print(message.content, end='', flush=True)
            answer_content += message.content

    return answer_content

if __name__ == "__main__":
    query = "1 plus 1 equals to?"
    answer = generate_answer(query)
    print("\n" + "=" * 20 + "Final Answer" + "=" * 20)
    print(answer)
