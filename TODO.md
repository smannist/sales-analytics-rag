## TODO

- Figure out actual architecture once you start integrating langchain and llm
  - Update 1: Registery pattern works well for the app, not sure about decorators, but they looked cleaner than other options imo
  - Update 2: I got rid of the templates since it needs a lot of editing from different files, I think new one is better. But still,
    I wonder if the f-string templates could be made more readable somehow. Look into it.
  - Note: current factories need to operate over the df's twice - it would be faster to build a list and do data and meta data in one go
    but as a tradeoff i think this is more readable + dataset is not large anyways
- Maybe do memory, but not familiar with it -- look into later
- ChromaDB's max batch size is 5461 -- something to keep in mind
