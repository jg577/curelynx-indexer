import zipfile
import click
import io
from lxml import etree
import logging
from typing import List, Dict
import json
import requests
import boto3
from requests_aws4auth import AWS4Auth
from botocore.exceptions import NoCredentialsError
from opensearchpy import AWSV4SignerAuth
import xmltodict
import http.client
import re

# http.client.HTTPConnection.debuglevel = 1


def print_element_hierarchy(element, level=0):
    # Print the current element's tag and attributes
    print("  " * level + f"{element.tag}: {element.attrib}")

    # Recursively print each child element
    for child in element:
        print_element_hierarchy(child, level + 1)


def convert_to_years(time_str):
    # Regex pattern to match the format
    pattern = r"([0-9]+) (Year|Years|Month|Months|Week|Weeks|Day|Days|Hour|Hours|Minute|Minutes)"
    match = re.match(pattern, time_str)

    if not match:
        return "Invalid format"

    # Extract number and time unit
    number, unit = int(match.group(1)), match.group(2).lower()

    # Conversion factors to years
    conversion_factors = {
        "year": 1,
        "years": 1,
        "month": 1 / 12,
        "months": 1 / 12,
        "week": 1 / 52,
        "weeks": 1 / 52,
        "day": 1 / 365,
        "days": 1 / 365,
        "hour": 1 / (365 * 24),
        "hours": 1 / (365 * 24),
        "minute": 1 / (365 * 24 * 60),
        "minutes": 1 / (365 * 24 * 60),
    }

    # Convert to years
    years = number * conversion_factors[unit]
    return years


def dict_preproc(data_dict) -> None:
    def preproc_min_age(time_str):
        if time_str == "N/A":
            return 0
        else:
            return convert_to_years(time_str)

    def preproc_max_age(time_str):
        if time_str == "N/A":
            return 200
        else:
            return convert_to_years(time_str)

    def preproc_leaf_text(val):
        if isinstance(val, str):
            return val
        else:
            return val["#text"]

    data_dict["clinical_study"]["eligibility"]["minimum_age"] = preproc_min_age(
        data_dict["clinical_study"]["eligibility"].get("minimum_age", "0 years")
    )
    data_dict["clinical_study"]["eligibility"]["maximum_age"] = preproc_max_age(
        data_dict["clinical_study"]["eligibility"].get("maximum_age", "200 Years")
    )
    data_dict["clinical_study"]["completion_date"] = preproc_leaf_text(
        data_dict["clinical_study"].get("completion_date", "")
    )
    data_dict["clinical_study"]["start_date"] = preproc_leaf_text(
        data_dict["clinical_study"].get("start_date", "")
    )
    data_dict["clinical_study"]["enrollment"] = preproc_leaf_text(
        data_dict["clinical_study"].get("enrollment", "")
    )
    # data_dict["clinical_study"]["clinical_results"]["reported_events"][
    #     "serious_events"
    # ]["category_list"]["category"]["event_list"]["event"][
    #     "sub_title"
    # ] = preproc_leaf_text(
    #     data_dict["clinical_study"]["clinical_results"]["reported_events"][
    #         "serious_events"
    #     ]["category_list"]["category"]["event_list"]["event"]["sub_title"]
    # )
    # data_dict["clinical_study"]["clinical_results"]["reported_events"]["other_events"][
    #     "category_list"
    # ]["category"]["event_list"]["event"]["sub_title"] = preproc_leaf_text(
    #     data_dict["clinical_study"]["clinical_results"]["reported_events"][
    #         "other_events"
    #     ]["category_list"]["category"]["event_list"]["event"]["sub_title"]
    # )


def xml_to_json(xml_element):
    # Convert lxml.etree object to a string
    xml_str = etree.tostring(xml_element, encoding="utf-8").decode("utf-8")

    # Function to remove newlines
    def remove_newlines(value):
        if isinstance(value, str):
            return value.replace("\n", "").replace("\r", "").strip()
        elif isinstance(value, dict):
            return {k: remove_newlines(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [remove_newlines(v) for v in value]
        return value

    # Parse the XML string to a dictionary
    dict_data = xmltodict.parse(
        xml_str, postprocessor=lambda _, key, value: (key, remove_newlines(value))
    )
    dict_preproc(dict_data)

    # Convert dictionary to JSON string
    return json.dumps(dict_data, indent=4)


def get_awsauth(region):
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        awsauth = AWSV4SignerAuth(credentials, region)

    except NoCredentialsError:
        print("AWS credentials not found")
        return None

    return awsauth


def query_index(query_str):
    return None


def insert_into_opensearch(
    opensearch_uri, zip_file_path, index_name, xml_parser, awsauth
):
    # Get AWS credentials from environment

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            file = zip_ref.read(file_name)
            file_like_object = io.BytesIO(file)
            if "NCT" in file_name:
                try:
                    doc = etree.parse(file_like_object, parser=xml_parser)
                except etree.XMLSyntaxError as err:
                    print(f"Error parsing file {file_name}: {err}")
                    return
                # Prepare the request URL
                url = f"{opensearch_uri}/{index_name}/_doc/"
                json_doc = xml_to_json(doc)
                doc = json.loads(json_doc)
                try:
                    response = requests.post(
                        url,
                        auth=awsauth,
                        json=doc,
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code != 201:
                        import pdb

                        pdb.set_trace()
                        print(
                            f"Failed to insert record from {file_name}: {response.text}"
                        )
                    else:
                        print(
                            f"Record inserted successfully from {file_name}: {response.json()['_id']}"
                        )
                except requests.RequestException as e:
                    print(f"Request failed: {e}")


@click.command()
@click.option(
    "--compressed-file",
    type=str,
    help="Source Xml file downloaded from clinicaltrials.gov",
    default="~/Downloads/AllPublicXML.zip",
)
@click.option(
    "--xml-schema",
    type=click.Path(exists=True),
    help="source xml schema format",
    default="clinical_trials_public.xsd",
)
@click.option(
    "--opensearch-uri",
    type=str,
    default="https://search-curelynx-ptabxnzs3jgoe477yww3grffua.aos.us-east-2.on.aws",
)
@click.option("--index-name", type=str, default="curelynx-dev-v2")
@click.option("--region", type=str, default="us-east-2")
def main(compressed_file, xml_schema, opensearch_uri, index_name, region) -> None:
    # setting up the xml parser
    with open(xml_schema, "rb") as file:
        schema_root = etree.XML(file.read())
        schema = etree.XMLSchema(schema_root)
    xml_parser = etree.XMLParser(schema=schema)

    aws_auth = get_awsauth(region)
    assert aws_auth is not None, "Aws auth did not complete"
    insert_into_opensearch(
        opensearch_uri, compressed_file, index_name, xml_parser, aws_auth
    )


if __name__ == "__main__":
    main()
