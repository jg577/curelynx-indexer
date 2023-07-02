import logging
import sys
import json
from typing import List, Optional
import pinecone
import click
from tqdm.auto import tqdm
import requests
from uuid import uuid4
from langchain.embeddings.openai import OpenAIEmbeddings
import yaml
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from requests.exceptions import Timeout
import urllib

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
OPENAI_ORGANIZATION = os.environ.get("OPENAI_ORGANIZATION")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING")

openai_emb_service = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_organization=OPENAI_ORGANIZATION,
)

mongodb_client = MongoClient(MONGODB_CONNECTION_STRING)

tokenizer = tiktoken.get_encoding("p50k_base")


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)
# Send a ping to confirm a successful connection
try:
    mongodb_client.admin.command("ping")
    logging.info("Pinged the deployment. Successfully connected to MongoDB!")
except Exception as e:
    print(e)


def user_query(data: dict, pinecone_index_name: str = PINECONE_INDEX_NAME):
    query_text = data["question"]
    question_embedding = openai_emb_service.embed_query(query_text)
    pinecone.init(api_key=PINECONE_API_KEY, environment="asia-southeast1-gcp-free")
    index = pinecone.GRPCIndex(pinecone_index_name)
    result = index.query(
        vector=question_embedding,
        top_k=4,
        include_metadata=True,
    )
    print(result)


def traverse_dict(data: dict):
    """
    Takes a json dictionary and returns key value
    """
    for key, value in data.items():
        if isinstance(value, dict):
            for result in traverse_dict(value):
                yield result
        elif isinstance(value, list):
            for i, v in enumerate(value):
                new_path = f"{key}[{i}]"
                if isinstance(v, dict):
                    for result in traverse_dict(v):
                        yield result
                else:
                    yield f"{new_path}: {v}"
        else:
            yield f"{key}: {value}"


def json_to_text(data) -> str:
    """
    Takes a json document and returns a string of final keys and values
    """
    # Create an empty string to store the keys and values
    result = ""

    # Iterate over the keys and values in the dictionary
    for line in traverse_dict(data):
        # Append the key and value to the result string
        result += f"{line}; "

    return result


def instantiate_pinecone_index(
    dimension: int,
    metadata_config: dict,
    index_name: str = PINECONE_INDEX_NAME,
) -> None:
    logging.info(
        f"Initializing pinecone index for {index_name} with  \n{metadata_config}"
    )
    pinecone.init(api_key=PINECONE_API_KEY, environment="asia-southeast1-gcp-free")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metadata_config=metadata_config,
        )
    else:
        logging.info(f"{index_name} is already in in the index")


def check_and_insert_document(
    document: dict,
    mongodb_client=mongodb_client,
    database_name="curelynx",
    collection="clinical_trials_v1",
) -> bool:
    """
    Takes in a mongodb database, collection name, and a document
    If the document exists, it returns False
    else it inserts the document into the collection and returns True
    """
    database = mongodb_client[database_name]
    collection = database[collection]
    if collection.find_one({"NCTId": document["NCTId"]}) is None:
        doc = {
            "index": PINECONE_INDEX_NAME,
            "index_type": "index",
            "NCTId": document["NCTId"],
            "condition": document["condition"],
            "text": document["text"],
        }
        collection.insert_one(doc)
        logging.info(
            f"MONGOGB: Inserted document for {doc['NCTId']} for condition:{doc['condition']}"
        )
        return True
    else:
        return False


def add_documents_to_index(
    documents: List[dict],
) -> None:
    """
    This function adds documents to the pinecone index
    """
    index = pinecone.Index(PINECONE_INDEX_NAME)

    batch_limit = 100
    metadatas = []
    texts = []
    for _, record in enumerate(tqdm(documents)):
        text = record["text"]
        record_metadata = {
            "NCTId": record["NCTId"],
            "Condition": record["condition"],
            "Location": record["location"],
            "Title": record["title"],
            "Organization": record["organization"],
            "text": text,
        }
        record_texts = text_splitter.split_text(text)
        record_metadatas = [
            {"chunk": j, "text": text, **record_metadata}
            for j, text in enumerate(record_texts)
        ]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            logging.info(f"Adding {ids} documents to the index")
            try:
                embeds = openai_emb_service.embed_documents(texts)
                logging.info(f"embeddings: {embeds}")
            except Exception as e:
                print(f"An error occurred: {e}")
            try:
                response = index.upsert(vectors=zip(ids, embeds, metadatas))
                print(response)
            except Exception as e:
                print(f"An error occurred: {e}")
            texts = []
            metadatas = []


def get_clinical_trials_full_documents(disease_name) -> List[dict]:
    """
    Takes in a disease name and returns a dict with the following structure:
    disease_doc = {
        "text": text,
        "condition": condition,
        "location": location,
        "title": title,
        "organization": organization
        }
    """
    documents = []
    # Define the base URL
    url = "https://clinicaltrials.gov/api/query/full_studies?"
    # Define the parameters
    params = {
        "expr": f"{disease_name} AND (Status:Recruiting OR Status:Not+yet+recruiting)",  # the search term with overall status
        "min_rnk": 1,
        "max_rnk": 100,
        "fmt": "json",  # get the response in json format
    }
    # Send the GET request
    response = requests.get(url, params=params)
    response_dict = response.json()
    if "FullStudies" in response_dict["FullStudiesResponse"]:
        for doc in response_dict["FullStudiesResponse"]["FullStudies"]:
            try:
                document = {}
                document["text"] = doc["Study"]
                document["condition"] = doc["Study"]["ProtocolSection"][
                    "ConditionsModule"
                ]["ConditionList"]["Condition"][0]
                document["location"] = doc["Study"]["ProtocolSection"][
                    "ContactsLocationsModule"
                ]["LocationList"]["Location"][0]["LocationCity"]
                document["NCTId"] = doc["Study"]["ProtocolSection"][
                    "IdentificationModule"
                ]["NCTId"]
                document["title"] = doc["Study"]["ProtocolSection"][
                    "IdentificationModule"
                ]["OfficialTitle"]
                document["organization"] = doc["Study"]["ProtocolSection"][
                    "IdentificationModule"
                ]["Organization"]["OrgFullName"]
                if check_and_insert_document(document):
                    documents.append(document)
                    logging.info(
                        f"Added document for {document['NCTId']} to the list for index"
                    )
            except Exception as e:
                logging.info(f"Skipping doc for {str(e)}")
    return documents


def filter_docs_for_indexing(
    documents: List[dict], mongodb_collection_name: str
) -> List[dict]:
    """
    This function takes a list of documents, a mondgodb name and only returns a list of
    documents that are not already in the mongodb collection. If the document is not found
    it'll add the document to the mongodb collection as well as return it in the list
    """
    filtered_docs = []
    for doc in documents:
        if check_and_insert_document(doc, collection=mongodb_collection_name):
            filtered_docs.append(doc)
            logging.info(f"Added document for {doc['NCTId']} to the list for index")
        else:
            logging.info(
                f"Skipping doc name {doc['NCTId']} as it already exists in the collection"
            )
    return filtered_docs


def get_data_with_timeout(url, search_expr, fields):
    try:
        response = requests.get(
            url=url,
            params={
                "expr": search_expr,
                "fields": ",".join(fields),
                "max_rnk": 1000,
                "fmt": "json",
            },
            timeout=60,  # timeout of 5 seconds
        )
        response.raise_for_status()  # will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    except Timeout:
        print("The request timed out")
    except requests.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.RequestException as err:
        print(f"Error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def get_documents_from_NCT(disease_name) -> List[dict]:
    logging.info(f"reached get documents from NCT")
    documents = []
    url = "https://clinicaltrials.gov/api/query/study_fields"
    search_expr = urllib.parse.quote_plus(
        f"{disease_name} AND (Status:Recruiting OR Status:Not+yet+recruiting)"
    )
    fields = [
        "NCTId",
        "Condition",
        "LocationCity",
        "OfficialTitle",
        "OrgFullName",
        "BriefSummary",
        "EligibilityCriteria",
        "InterventionDescription",
        "PrimaryOutcomeDescription",
    ]
    responses_dict = get_data_with_timeout(url, search_expr, fields)
    if responses_dict["StudyFieldsResponse"]["NStudiesReturned"] > 0:
        for r in responses_dict["StudyFieldsResponse"]["StudyFields"]:
            doc = {}
            doc["NCTId"] = r["NCTId"][0] if r.get("NCTId") else ""
            doc["condition"] = r["Condition"][0] if r.get("Condition") else ""
            doc["location"] = r["LocationCity"][0] if r.get("LocationCity") else ""
            doc["title"] = r["OfficialTitle"][0] if r.get("OfficialTitle") else ""
            doc["organization"] = r["OrgFullName"][0] if r.get("OrgFullName") else ""
            summary = r["BriefSummary"] if r.get("BriefSummary") else ""
            criteria = r["EligibilityCriteria"] if r.get("EligibilityCriteria") else ""
            primary_outcome = (
                r["PrimaryOutcomeDescription"]
                if r.get("PrimaryOutcomeDescription")
                else ""
            )
            interventions = (
                r["InterventionDescription"] if r.get("InterventionDescription") else ""
            )
            doc[
                "text"
            ] = f"ID: {doc['NCTId']}. The trial is for patients suffering from {doc['condition']}. The trial is located at {doc['location']}. The trial is organized by {doc['organization']}. The title of the trial is {doc['title']}. Summary of the trial is as follows: {summary} {criteria}. The primary outcome is as follows:{primary_outcome}. The medical intervention for the trial are as follows: {interventions}".replace(
                "\n", ""
            )
            documents.append(doc)
    return documents


@click.command()
@click.option(
    "--disease-file",
    type=click.Path(exists=True),
    help="Path to the disease lists YAML file",
    default="disease_lists.yaml",
)
def main(disease_file) -> None:
    # instantiating the vector store index
    with open(disease_file, "r") as file:
        disease_lists = yaml.safe_load(file)
    logging.info("Loaded the diseases")
    # Getting the diseases to be indexed
    top_rare_diseases = disease_lists["TOP_100_RARE_DISEASES"]
    most_pop_trial_diseases = disease_lists["MOST_POP_TRIAL_DISEASES"]
    logging.info("Instantiating the vector store index")
    # dimensions are for text-embedding-ada-002
    # using pinecone indices: https://docs.pinecone.io/docs/langchain
    metadata_config = {"indexed": ["condition"]}
    instantiate_pinecone_index(
        dimension=1536,
        metadata_config=metadata_config,
    )
    # querying  clinical trials
    all_diseases = top_rare_diseases + most_pop_trial_diseases
    for disease_name in all_diseases:
        logging.info(f"Getting all the relevant clinical trials for {disease_name}")
        documents = get_documents_from_NCT(disease_name)
        filtered_documents = filter_docs_for_indexing(documents, "clinical_trials_v1")

        # Adding the documents into the index
        if filtered_documents:
            try:
                add_documents_to_index(filtered_documents)
                logging.info(
                    f"Finished adding documents of the disease: {disease_name} to index: {PINECONE_INDEX_NAME}"
                )
            except Exception as exp:
                logging.info(f"Skipping because of {str(exp)}")
        else:
            logging.info(f"Skipping for {disease_name}")

    # Check the index
    print(pinecone.describe_index(PINECONE_INDEX_NAME))


if __name__ == "__main__":
    main()
