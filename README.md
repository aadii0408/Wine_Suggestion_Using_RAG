# Wine Recommendation with Retrieval Augmented Generation (RAG)

This project demonstrates a simple implementation of Retrieval Augmented Generation (RAG) for recommending wines based on user queries. It leverages a vector database, an embedding model, and a large language model (LLM) to provide more relevant and informative recommendations.

## How it Works

1. **Data Loading and Embedding:**
    - Wine data is loaded from a CSV file (`top_rated_wines.csv`).
    - The `notes` column of the data is encoded into embedding vectors using the `SentenceTransformer` library with the `all-MiniLM-L6-v2` model.
    - The vectors and original data are stored in a Qdrant vector database.

2. **Query Processing:**
    - The user's query is also encoded into an embedding vector.
    - This vector is used to search for similar wines in the Qdrant database based on semantic similarity.
    - The top matching wines are retrieved.

3. **Retrieval Augmentation:**
    - The retrieved wines are added to the prompt sent to the LLM (OpenAI's `gpt-3.5-turbo`).
    - This provides the LLM with relevant context and helps it generate a more informed recommendation.

4. **Response Generation:**
    - The LLM generates a response based on the augmented prompt.
    - The response is formatted and displayed to the user.

## Technologies Used

- **Qdrant:** Vector database for storing and searching wine embeddings.
- **SentenceTransformer:** Library for generating text embeddings.
- **OpenAI API:** Provides access to the `gpt-3.5-turbo` large language model.
- **Rich:** Library for enhanced console output and styling.
- **Pandas:** Library for data manipulation and loading.

## Example

**User Query:** "Suggest me an amazing Malbec wine from Argentina"

**Output:** The system will retrieve relevant wines from the database and provide a recommendation based on the query and retrieved information.

## Benefits of RAG

- **Improved Relevance:** RAG ensures that the LLM's responses are grounded in relevant data, leading to more accurate and specific recommendations.
- **Up-to-date Information:** The system can incorporate new data by updating the vector database, allowing the LLM to recommend wines that were not part of its original training data.
- **Explainability:** The retrieved results provide transparency into the reasoning behind the LLM's recommendations.

## Limitations

- This is a simplified example and may not handle all types of user queries or complex scenarios.
- The quality of recommendations depends on the quality and comprehensiveness of the wine data.
- The system relies on external services (OpenAI API, Qdrant) which may have usage limits or costs.
