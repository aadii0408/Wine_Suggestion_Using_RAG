# Description: A simple example of a Retrieval-Augmented Generation (RAG) system for wine recommendation.

# ! pip install -q rich rich-theme-manager


# Import the requiered libraries
from rich.console import Console
from rich.style import Style
from openai import OpenAI
# from openai.api_resources import completion
# from openai.api_resources.chat import Chat
# from openai.api_resources.completion import Completion
from rich.panel import Panel
import pathlib
from rich_theme_manager import Theme, ThemeManager
import os

THEMES = [
    Theme(
        name="dark",
        description="Dark mode theme",
        tags=["dark"],
        styles={
            "repr.own": Style(color="#e87d3e", bold=True),  # Class names
            "repr.tag_name": "dim cyan",  # Adjust tag names
            "repr.call": "bright_yellow",  # Function calls and other symbols
            "repr.str": "bright_green",  # String representation
            "repr.number": "bright_red",  # Numbers
            "repr.none": "dim white",  # None
            "repr.attrib_name": Style(color="#e87d3e", bold=True),  # Attribute names
            "repr.attrib_value": "bright_blue",  # Attribute values
            "default": "bright_white on black",  # Default text and background
        },
    ),
    Theme(
        name="light",
        description="Light mode theme",
        styles={
            "repr.own": Style(color="#22863a", bold=True),  # Class names
            "repr.tag_name": Style(color="#00bfff", bold=True),  # Adjust tag names
            "repr.call": Style(
                color="#ffff00", bold=True
            ),  # Function calls and other symbols
            "repr.str": Style(color="#008080", bold=True),  # String representation
            "repr.number": Style(color="#ff6347", bold=True),  # Numbers
            "repr.none": Style(color="#808080", bold=True),  # None
            "repr.attrib_name": Style(color="#ffff00", bold=True),  # Attribute names
            "repr.attrib_value": Style(color="#008080", bold=True),  # Attribute values
            "default": Style(
                color="#000000", bgcolor="#ffffff"
            ),  # Default text and background
        },
    ),
]

theme_dir = pathlib.Path("themes").expanduser()
theme_dir.expanduser().mkdir(parents=True, exist_ok=True)

theme_manager = ThemeManager(theme_dir=theme_dir, themes=THEMES)
theme_manager.list_themes()

dark = theme_manager.get("dark")
theme_manager.preview_theme(dark)

from rich.console import Console

dark = theme_manager.get("dark")
# Create a console with the dark theme
console = Console(theme=dark)

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

"""## Loading the Dataset

Since the data is in a simple, small and structured CSV file, we can load it using Pandas.
"""

import pandas as pd

data = (
    pd.read_csv("top_rated_wines.csv")
    .query("variety.notna()")
    .reset_index(drop=True)
    .to_dict("records")
)
console.print(data[:3])

"""## Encode using Vector Embedding

We will use one of the popular open source vector databases, [Qdrant](https://qdrant.tech/), and one of the popular embedding encoder and text transformer libraries, [SentenceTransformer](https://sbert.net/).
"""

# ! pip install -q qdrant-client sentence-transformers

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# create the vector database client
qdrant = QdrantClient(":memory:")  # Create in-memory Qdrant instance

# Create the embedding encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Model to create embeddings

# Create collection to store the wine rating data
collection_name = "top_wines"

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

"""### Loading the data into the vector database

We will use the (vector) collection that we created above, to go over all the `notes` column of the wine dataset, and encode it into embedding vector, and store it in the vector database. The indexing of the data to allow quick retrieval is running in the background as we load it.

This step will take a few seconds (less than a minute on my laptop).
"""

# vectorize!
qdrant.upload_points(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["notes"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(data)  # data is the variable holding all the wines
    ],
)

console.print(qdrant.get_collection(collection_name=collection_name))

"""## **R**etrieve sematically relevant data based on user's query

Once the data is loaded into the vector database and the indexing process is done, we can start using our simple RAG system.
"""

user_prompt = "Suggest me an amazing Malbec wine from Argentina"

"""### Encoding the user's query

We will use the same encoder that we used to encode the document data to encode the query of the user.
This way we can search results based on semantic similarity.
"""

query_vector = encoder.encode(user_prompt).tolist()

"""### Search similar rows

We can now take the embedding encoding of the user's query and use it to find similar rows in the vector database.
"""

# Search time for awesome wines!

hits = qdrant.search(
    collection_name=collection_name, query_vector=query_vector, limit=3
)

from rich.console import Console
from rich.text import Text
from rich.table import Table

table = Table(title="Retrieval Results", show_lines=True)

table.add_column("Name", style="#e0e0e0")
table.add_column("Region", style="bright_red")
table.add_column("Variety", style="green")
table.add_column("Rating", style="yellow")
table.add_column("Notes", style="#89ddff")
table.add_column("Score", style="#a6accd")

for hit in hits:
    table.add_row(
        hit.payload["name"],
        hit.payload["region"],
        hit.payload["variety"],
        str(hit.payload["rating"]),
        f'{hit.payload["notes"][:50]}...',
        f"{hit.score:.4f}",
    )

console.print(table)

"""## **A**ugment the prompt to the LLM with retrieved data

In our simple example, we will simply take the top 3 results and use them as is in the prompt to the generation LLM.

## **G**enerate reply to the user's query

We will use one of the most popular generative AI LLMs from [OpenAI](https://platform.openai.com/docs/models).
"""

# ! pip install -q openai python-dotenv

from dotenv import load_dotenv

load_dotenv()

"""### First let's try without **R**etrieval

We can ask the LLM to recommend based only on the user prompt.
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Now time to connect to the large language model
from openai import OpenAI
from rich.panel import Panel
import os

# Load the API key from the environment variable, or use the hardcoded one if not found
# Instantiate the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are chatbot, a wine specialist. Your top priority is to help guide users into selecting amazing wine and guide them with their requests.",
        },
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Here is my wine recommendation:"},
    ],
)

response_text = Text(completion.choices[0].message.content)
styled_panel = Panel(
    response_text,
    title="Wine Recommendation without Retrieval",
    expand=False,
    border_style="bold green",
    padding=(1, 1),
)

console.print(styled_panel)

"""### Now, add **R**etrieval Results

The recommendation sounds great, however, we don't have this wine in our inventory and menu. Moreover, new wines may be newly available that were not part of the pre-training of the LLM.

We will run the same query with the **R**trieval results and get better recommendations for our business needs.
"""

# define a variable to hold the search results
search_results = [hit.payload for hit in hits]

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are chatbot, a wine specialist. Your top priority is to help guide users into selecting amazing wine and guide them with their requests.",
        },
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(search_results)},
    ],
)

response_text = Text(completion.choices[0].message.content)
styled_panel = Panel(
    response_text,
    title="Wine Recommendation with Retrieval",
    expand=False,
    border_style="bold green",
    padding=(1, 1),
)

console.print(styled_panel)
