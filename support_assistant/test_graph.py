from graph import graph

def main():
    queries = [
        "How long do I have to report a damaged grocery item?",
        "What are Zepto support hours?",
        "What is the capital of France?",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print("QUERY:", query)

        result = graph.invoke({
            "query": query
        })

        print("ANSWER:", result["answer"])
        print("SOURCES:", result["sources"])
        print("CONFIDENCE:", result["confidence"])


if __name__ == "__main__":
    main()