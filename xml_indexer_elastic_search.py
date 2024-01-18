import zipfile
import click
import io
from lxml import etree
import logging
from typing import List, Dict
import json
import requests
import boto3
import xmltodict
import http.client
import re
from elasticsearch import BadRequestError, ConflictError, Elasticsearch

# http.client.HTTPConnection.debuglevel = 1


def print_element_hierarchy(element, level=0):
    # Print the current element's tag and attributes
    print("  " * level + f"{element.tag}: {element.attrib}")

    # Recursively print each child element
    for child in element:
        print_element_hierarchy(child, level + 1)


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
    # Convert dictionary to JSON string
    return json.dumps(dict_data, indent=4)


def extract_contact_info(contact_data):
    if isinstance(contact_data, list):
        return [
            [
                data.get("last_name", None),
                data.get("phone", None),
                data.get("email", None),
            ]
            for data in contact_data
        ]
    elif contact_data:
        return [
            [
                contact_data.get("last_name", None),
                contact_data.get("phone", None),
                contact_data.get("email", None),
            ]
        ]
    else:
        return []


def xml_preproc(data):
    # Initialize an empty dictionary to store the extracted data
    extracted_data = {}

    # Extracting fields from the data
    clinical_study = data.get("clinical_study", data)

    # Extract simple fields
    print(clinical_study)
    extracted_data["url"] = clinical_study.get("required_header", {}).get("url", None)
    extracted_data["org_study_id"] = clinical_study.get("id_info", {}).get(
        "org_study_id", None
    )
    extracted_data["nct_id"] = clinical_study.get("id_info", {}).get("nct_id", None)

    # Handle 'secondary_id' as it can be a list or a single value
    secondary_id = clinical_study.get("id_info", {}).get("secondary_id", None)
    if isinstance(secondary_id, list):
        extracted_data["secondary_id"] = secondary_id
    else:
        extracted_data["secondary_id"] = [secondary_id] if secondary_id else []

    extracted_data["brief_title"] = clinical_study.get("brief_title", None)
    extracted_data["lead_sponsor"] = (
        clinical_study.get("sponsors", {}).get("lead_sponsor", {}).get("agency", None)
    )
    extracted_data["source"] = clinical_study.get("source", None)
    extracted_data["brief_summary"] = clinical_study.get("brief_summary", {}).get(
        "textblock", None
    )
    extracted_data["detailed_description"] = clinical_study.get(
        "detailed_description", {}
    ).get("textblock", None)
    extracted_data["overall_status"] = clinical_study.get("overall_status", None)
    extracted_data["phase"] = clinical_study.get("phase", None)
    extracted_data["study_type"] = clinical_study.get("study_type", None)

    # Extract study design info fields
    study_design_info = clinical_study.get("study_design_info", {})
    extracted_data["intervention_model"] = study_design_info.get(
        "intervention_model", None
    )
    extracted_data["primary_purpose"] = study_design_info.get("primary_purpose", None)

    # Extract intervention fields
    intervention = clinical_study.get("intervention", {})
    if isinstance(intervention, list):
        extracted_data["intervention_name"] = [
            interv.get("intervention_name", None) for interv in intervention
        ]
    else:
        extracted_data["intervention_name"] = (
            [intervention.get("intervention_name", None)] if intervention else []
        )

    # Extract eligibility criteria
    eligibility = clinical_study.get("eligibility", {})
    extracted_data["eligibility_criteria"] = eligibility.get("criteria", {}).get(
        "textblock", None
    )
    extracted_data["gender"] = eligibility.get("gender", None)

    # Convert age fields to integers, handle cases where they're not provided
    min_age = eligibility.get("minimum_age", "0 Years")
    max_age = eligibility.get("maximum_age", "100 Years")
    extracted_data["minimum_age"] = (
        int(min_age.split()[0]) if min_age.split()[0].isdigit() else 0
    )
    extracted_data["maximum_age"] = (
        int(max_age.split()[0]) if max_age.split()[0].isdigit() else 100
    )

    # Extract location
    location = clinical_study.get("location", {})
    if isinstance(location, list):
        extracted_data["locations"] = [
            loc.get("facility", {}).get("name", None) for loc in location
        ]
        extracted_data["cities"] = [
            loc.get("facility", {}).get("address", {}).get("city", None)
            for loc in location
        ]
        extracted_data["states"] = [
            loc.get("facility", {}).get("address", {}).get("state", None)
            for loc in location
        ]
        extracted_data["countries"] = [
            loc.get("facility", {}).get("address", {}).get("country", None)
            for loc in location
        ]
    else:
        extracted_data["locations"] = (
            [location.get("facility", {}).get("name", None)] if location else []
        )
        address = location.get("facility", {}).get("address", {})
        extracted_data["cities"] = [address.get("city", None)] if address else []
        extracted_data["states"] = [address.get("state", None)] if address else []
        extracted_data["country"] = [address.get("country", None)] if address else []

    # Extract study dates
    extracted_data["study_start_date"] = clinical_study.get("start_date", None)
    extracted_data["study_end_date"] = clinical_study.get("completion_date", None)

    # Handle primary_completion_date
    primary_completion_date = clinical_study.get("primary_completion_date")
    if isinstance(primary_completion_date, dict):
        extracted_data["primary_completion_date"] = primary_completion_date.get(
            "#text", None
        )
    else:
        extracted_data["primary_completion_date"] = primary_completion_date

    overall_official = clinical_study.get("overall_official", {})
    extracted_data["overall_official_contact"] = extract_contact_info(overall_official)

    # Extract overall contact
    overall_contact = clinical_study.get("overall_contact", {})
    extracted_data["overall_contact"] = extract_contact_info(overall_contact)

    # Extract overall contact backup
    overall_contact_backup = clinical_study.get("overall_contact_backup", {})
    extracted_data["overall_contact_backup"] = extract_contact_info(
        overall_contact_backup
    )

    enrollment = clinical_study.get("enrollment")
    if isinstance(enrollment, dict):
        enrollment = enrollment.get("#text", "0")
    elif not isinstance(enrollment, str):
        enrollment = "0"
    extracted_data["enrollment"] = int(enrollment) if enrollment.isdigit() else 0

    # Extract condition mesh terms
    condition_browse = clinical_study.get("condition_browse", {})
    extracted_data["condition_mesh_terms"] = condition_browse.get("mesh_term", [])

    # Additional potentially useful fields
    extracted_data["study_first_submitted"] = clinical_study.get(
        "study_first_submitted", None
    )
    last_update_posted = clinical_study.get("last_update_posted")
    if isinstance(last_update_posted, dict):
        extracted_data["last_update_posted"] = last_update_posted.get("#text", None)
    else:
        extracted_data["last_update_posted"] = last_update_posted
    extracted_data["verification_date"] = clinical_study.get("verification_date", None)
    extracted_data["has_expanded_access"] = clinical_study.get(
        "has_expanded_access", None
    )
    extracted_data["keywords"] = clinical_study.get("keyword", [])

    if extracted_data["overall_status"] == "Recruiting":
        import pdb

        pdb.set_trace()
    return extracted_data


@click.command()
@click.option(
    "--compressed-file",
    type=str,
    help="Source Xml file downloaded from clinicaltrials.gov",
    default="/Users/jeevg/Downloads/AllPublicXML.zip",
)
@click.option(
    "--xml-schema",
    type=click.Path(exists=True),
    help="source xml schema format",
    default="clinical_trials_public.xsd",
)
def main(compressed_file, xml_schema) -> None:
    # setting up the xml parser
    with open(xml_schema, "rb") as file:
        schema_root = etree.XML(file.read())
        schema = etree.XMLSchema(schema_root)
    xml_parser = etree.XMLParser(schema=schema)

    client = Elasticsearch(
        "https://86aab91241d94f249890c450a7847ece.us-central1.gcp.cloud.es.io:443",
        api_key="cmFGYTVZd0JadDZRZ3JIc3A2dEM6eC1LZ004TWxScENJWURNQlZtcVpLQQ==",
    )

    # API key should have cluster monitor rights
    client.info()

    def error_fix(error, old_dict):
        # "[1:18145] failed to parse field [clinical_study.clinical_results.reported_events.serious_events.category_list.category.event_list.event.sub_title] of type [text] in document with id 'NCT00000134'. Preview of field's value: '{@vocab=other, #text=Thrombocytopenia}'"
        error_field = re.findall(r"\[(.*?)\]", error)[1]
        replacement_val = re.search(r"#text=([^}]*)", error).group(1)
        fields = error_field.split(".")
        running_dict = old_dict
        for f in fields:
            import pdb

            pdb.set_trace()
            running_dict = running_dict[f]
        running_dict = replacement_val

    with zipfile.ZipFile(compressed_file, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            file = zip_ref.read(file_name)
            file_like_object = io.BytesIO(file)
            if "NCT" in file_name:
                try:
                    doc = etree.parse(file_like_object, parser=xml_parser)
                except etree.XMLSyntaxError as err:
                    print(f"Error parsing file {file_name}: {err}")
                    return
                json_doc = xml_to_json(doc)
                doc = json.loads(json_doc)
                preproc_doc = xml_preproc(doc)
                # try:
                #     client.create(
                #         index="search-curelynx-index-v2",
                #         id=doc["clinical_study"]["id_info"]["nct_id"],
                #         document=doc,
                #     )
                # except BadRequestError as e:
                #     if "document_parsing_exception" in str(e):
                #         # Handle document parsing exceptions specifically
                #         errors = e.body["error"]["root_cause"]
                #         for err in errors:
                #             error_fix(err["reason"], doc)
                #         try:
                #             client.create(
                #                 index="search-curelynx-index-v2",
                #                 id=doc["clinical_study"]["id_info"]["nct_id"],
                #                 document=doc,
                #             )
                #         except Exception as e:
                #             print(e)
                #     else:
                #         # Handle other BadRequestErrors
                #         print("Other BadRequestError occurred:", e)
                # except ConflictError:
                # If a ConflictError is caught, just skip this document
                # print("Document already exists, skipping...")


if __name__ == "__main__":
    main()
