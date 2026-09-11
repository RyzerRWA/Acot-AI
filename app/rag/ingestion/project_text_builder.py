import re


class ProjectTextBuilder:

    @staticmethod
    def clean_description(text: str) -> str:

        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common missing spaces after punctuation
        text = re.sub(
            r"([.!?,])([A-Za-z])",
            r"\1 \2",
            text
        )

        # Fix common missing spaces before capitalized words
        text = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            text
        )

        return text.strip()

    @staticmethod
    def format_list(value) -> str:

        if not value:
            return ""

        if isinstance(value, list):

            # Ignore empty values
            values = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

            return ", ".join(values)

        return str(value).strip()

    @staticmethod
    def build(project: dict) -> str:

        description = ProjectTextBuilder.clean_description(
            project.get("description", "")
        )

        property_types = ProjectTextBuilder.format_list(
            project.get("property_types")
        )

        amenities = ProjectTextBuilder.format_list(
            project.get("amenities")
        )

        text = f"""
PROJECT: {project.get("name", "")}

PROJECT ID:
{project.get("id", "")}

DESCRIPTION:
{description}

DEVELOPER:
{project.get("developer_name", "")}

LOCATION:
City: {project.get("city", "")}
Community: {project.get("community", "")}
Sub-community: {project.get("sub_community", "")}

PROJECT STATUS:
Status: {project.get("status", "")}
Project Status: {project.get("project_status", "")}

PROPERTY DETAILS:
Property Types: {property_types}
Minimum Bedrooms: {project.get("bedroom_min", "")}
Maximum Bedrooms: {project.get("bedroom_max", "")}
Minimum Size: {project.get("size_min", "")} sqft
Maximum Size: {project.get("size_max", "")} sqft

PRICING:
Starting Price: AED {project.get("price", "")}

HANDOVER:
{project.get("handover_time", "")}

AMENITIES:
{amenities}

PROJECT LINKS:
Brochure URL: {project.get("brochure_url", "")}
Property Finder URL: {project.get("property_finder_url", "")}

SOURCE:
{project.get("source", "")}
""".strip()

        return text