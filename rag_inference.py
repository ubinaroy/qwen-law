from rag import EnhancedRAG
rag = EnhancedRAG()
complex_question = "小医仙是谁？"
answer = rag.ask(complex_question)
print(f"问题：{complex_question}")
print("答案：")
print(answer)