import ray
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rag.data import extract_sections
from functools import partial
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Open the YAML settings file
with open('settings.yml', 'r') as file:
    # Load the YAML data
    settings = yaml.load(file, Loader=yaml.FullLoader)



def process_text(efs_dir: str) -> ray.data.Dataset:
    """
    Generates a dataset of sections extracted from HTML files in a given
    directory.

    Args:
        efs_dir (str): The path of the EFS directory where the HTML files
        are located.
    Returns:
        ray.data.Dataset: A dataset of sections extracted from the HTML files.
    """
    DOCS_DIR = Path(efs_dir, "docs.djangoproject.com/en/5.0/")
    ds = ray.data.from_items(
        [{"path": path} for path in DOCS_DIR.rglob("*.html")
         if not path.is_dir()])

    # Extract sections
    sections_ds = ds.flat_map(extract_sections)
    return sections_ds


sections_ds = process_text(efs_dir=settings['paths']['efs_dir'], )


def chunk_section(section, chunk_size: int, chunk_overlap: int) -> list:
    """
    LLMs have a maximum context length, so we're going to split the text within each section into smaller chunks.
    Intuitively, smaller chunks will encapsulate single/few concepts and will be less noisy compared to larger chunks.
    We're going to choose some typical text splitting values (ex. chunk_size=300) to create our chunks
    Args:
        section (dict): The section to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: A list of dictionaries, each containing the text content and source
        metadata of a chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    chunks = text_splitter.create_documents(
        texts=[section["text"]],
        metadatas=[{"source": section["source"]}])
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]}
            for chunk in chunks]

chunk_size = 300
chunk_overlap = 50
# Scale chunking
chunks_ds = sections_ds.flat_map(partial(
    chunk_section,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap))
chunks_ds.show(1)

def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    """
    Now that we've created small chunks from our sections, we need a way to identify the most relevant
    ones for a given query. A very effective and quick method is to embed our data using a pretrained
    model and use the same model to embed the query. We can then compute the distance between
    all of the chunk embeddings and our query embedding to determine the top-k chunks. There are many
    different pretrained models to choose from to embed our data but the most popular ones can be
     discovered through HuggingFace's Massive Text Embedding Benchmark (MTEB) leaderboard. These models
     were pretrained on very large text corpus through tasks such as next/masked token prediction
     which allowed them to learn to represent subtokens in N dimensions and capture semantic relationships.
     We can leverage this to represent our data and identify the most relevant contexts to use to answer a given query.
     We're using Langchain's Embedding wrappers (HuggingFaceEmbeddings and OpenAIEmbeddings) to easily load
     the models and embed our document chunks.
    """
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"])
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,  # also works with model_path
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    return embedding_model


class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu", "batch_size": 100})
    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}


# Embed chunks
embedding_model_name = "thenlper/gte-base"
embedded_chunks = chunks_ds.map_batches(
    EmbedChunks,
    fn_constructor_kwargs={"model_name": embedding_model_name},
    batch_size=100,
    num_cpus=1,
    compute=ray.data.ActorPoolStrategy(size=1))



# Sample
sample = embedded_chunks.take(1)
print ("embedding size:", len(sample[0]["embeddings"]))
print (sample[0]["text"])

# print(type(sections_ds))

# section_lengths = []
# for section in sections_ds.take_all():
#     section_lengths.append(len(section["text"]))

# Manhattan plot
# plt.figure(figsize=(12, 3))
# plt.plot(section_lengths, marker='o')
# plt.title("Section lengths")
# plt.ylabel("# chars")
# plt.savefig("section_lengths.png")
