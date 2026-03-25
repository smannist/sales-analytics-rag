- LangChain's Chroma wrapper already uses all-MiniLM-L6-v2 as the sentence transformer as a default
- LangChain already has a class for defining Documents, so we swapped to that instead
- add_documents expects a list of Documents, so now the factories return a list of Documents
- ChromaDB's max batch size is 5461 -- something to keep in mind
- Persistent memory can be done with langchain afaik
- Registery pattern works well for the app
- Not sure about decorators, but I think we really do not need class based registery since we just insert the collection anyways
- Current factories need to operate over the df's twice - it would be faster to build a list and do data and metadata in one go
  but as a tradeoff I think this is more readable + dataset is not large anyways
