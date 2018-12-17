import xml.etree.ElementTree as ET
import pandas as pd

class XMLParser:
    """Parser assumes the first level of xml tags are to be transformed
    to rows in a Pandas dataframe. For each of the first-level tags it takes
    all of their subtags and attributes, and puts them as columns to
    the current row in the dataframe."""

    def __init__(self, xml_data):
        self.root = ET.XML(xml_data)

    def parse_root(self, root):
        """Return a list of dictionaries from the text
         and attributes of the children under this XML root."""
        return [self.parse_element(child) for child in iter(root)]

    def parse_element(self, element, parsed=None):
        """ Collect {key:attribute} and {tag:text} from the XML
         element and all its children into a single dictionary of strings."""
        if parsed is None:
            parsed = dict()

        for key in element.keys():
            if key not in parsed:
                parsed[key] = element.attrib.get(key)
            else:
                raise ValueError('Duplicate attribute {0} = {1}, prev = {2}' \
                                 .format(key,
                                         element.attrib.get(key),
                                         parsed[key]))

        for child in iter(element):
            if child.tag == "RelQSubject" or child.tag == "RelQBody":
                parsed[child.tag] = child.text
            else:
                self.parse_element(child, parsed)

        return parsed

    def to_df(self):
        """ Initiate the root XML, parse it, and return a dataframe"""
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)

