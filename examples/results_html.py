"""
Generates html tables for key tables.
"""

from typing import Callable, List, Dict, Optional, Union
from pathlib import Path
import os


def get_table_style(table_name: str = "styled_table", table_color: str = "#000000",
                    header_color: str = "#ffffff") -> str:

    html = '<style type="text/css"> '

    html += f'table.{table_name} caption {{padding-bottom: 0.5em; font-weight: bold; font-size: 28px;}} ' \
            f'table.{table_name} {{border-collapse: collapse; margin: 25px 0; font-size: ' \
            '0.9em; font-family: sans-serif; min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);} ' \
            f'table.{table_name} thead tr {{background-color: {table_color}; color: {header_color};text-align: left;}} ' \
            f'table.{table_name} th {{padding: 12px 15px;}} ' \
            f'table.{table_name} td {{padding: 12px 15px;}} ' \
            f'table.{table_name} tr {{border-bottom: 1px solid #dddddd;}} ' \
            f'table.{table_name} tbody tr:last-of-type {{border-bottom: 2px solid {table_color};}}' \
            f'table.{table_name} tbody tr:nth-of-type(even) {{background-color: #f3f3f3;}} '

    return html + '</style>'


def save_html(html: str, path: Union[Path, str] = "temp.html"):
    """ Saves html. """
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)


def generate_html(dict_: dict, title: Optional[str] = None, table_style: str = None, table_name: str = "styled_table") \
        -> str:
    """ Generates html from dictionary.
    Parameters
    ----------
    dict_: dict
        dictionary to be converted into a table
        single layer dict[str, str]
        double layer dict[str:dict]
    title: str
        title for table
    table_style: str
        html formatting of table
    table_name: str
        name used in html formatting style

    Returns
    -------
    html: str
        table html
    """
    if not isinstance(dict_, dict):
        raise TypeError("Input must be a dictionary.")

    # start generating html table
    html = [f'<table class="{table_name}">']
    if title is not None:
        html.append(f"<caption>{title}</caption>")

    # generating main part of table
    formater = _get_table_formater(dict_)
    html += formater(dict_)

    # add formatting to table
    html.append("</table>")

    # set max width of table
    # if True:
    #     html.insert(0, '<div style="max-width:1000px;">')
    #     html.append("</div>")

    if table_style is not None:
        html.append(table_style)

    return ''.join(html)


def _get_table_formater(dict_) -> Callable:
    """ Determines weather the table is a single layer dict[str, str] or is a  double layer dict[str:dict]. """
    if isinstance(list(dict_.values())[0], dict):
        return _double_level_table
    else:
        return _single_level_table


def _single_level_table(dict_: dict) -> List[str]:
    """ Generates main body of a table. """
    html = []

    # Headers
    headers = dict_.keys
    html += _generate_head(headers)

    # Rows
    html.append("<tbody>")
    for k, v in dict_.items():
        row_data = [k, v]
        html += _generate_row(row_data)
    html.append("</tbody>")

    return html


def _double_level_table(dict_) -> List[str]:
    """ Generates main body of a table. """
    html = []
    # Headers
    headers = _get_headers(dict_)
    html += _generate_head(headers)

    # Rows
    html.append("<tbody>")
    for k, v in dict_.items():
        row_data = [k] + _get_row_data(v, headers)[1:]
        html += _generate_row(row_data)
    html.append("</tbody>")

    return html


def _get_headers(dict_) -> List[str]:
    headers = []
    for entry in dict_.values():
        for key in entry.keys():
            if key not in headers:
                headers.append(key)

    headers.insert(0, "keys")
    return headers


def _get_row_data(row_dict: Dict, headers: List[str]) -> List[str]:
    row = []
    for header in headers:
        if header in row_dict:
            if isinstance(row_dict[header], type):  # convert type to string
                row.append(row_dict[header].__name__)
                continue
            elif isinstance(row_dict[header], (tuple, list)):
                if len(row_dict[header]) != 0 and isinstance(row_dict[header][0], type):
                    row.append([type_.__name__ for type_ in row_dict[header]])
                    continue
            elif isinstance(row_dict[header], dict):
                if "value" not in row_dict[header]:
                    raise ValueError("sub-dicts must have value attribute")

            row.append(row_dict[header])

        else:
            row.append(" ")

    return row


def _generate_row(row_data: List[str]) -> List[str]:
    html = ["<tr>"]
    for entry in row_data:
        if isinstance(entry, bool) and not entry:  # skip false entries
            html.append("<td> </td>")
            continue
        if isinstance(entry, dict):  # add hyperlinks
            # convert type to string
            if isinstance(entry['value'], type):
                value = entry['value'].__name__
            else:
                value = entry['value']

            if "url" in entry:
                html.append(f"<td><a href={entry['url']}>{value}</a></td>")
                continue
            else:
                html.append(f"<td>{value} </td>")
                continue

        html.append("<td>{0}</td>".format(entry))
    html.append("</tr>")
    return html


def _generate_head(row_data: List[str]) -> List[str]:
    html = ["<thead>", "<tr>"]
    for entry in row_data:
        html.append("<th>{0}</th>".format(entry))
    html.append("</tr>")
    html.append("</thead>")
    return html


def generate_html_list(
        data: dict[list],
        title: Optional[str] = None,
        table_style: str = None,
        table_name: str = "styled_table") \
        -> str:
    """ Generates html from list of dictionaries

    Parameters
    ----------
    data: dict[list]
    title: str
        title for table
    table_style: str
        html formatting of table
    table_name: str
        name used in html formatting style

    Returns
    -------
    html: str
        table html
    """
    if not isinstance(data, dict):
        raise TypeError("Input must be a list[list[list]].")

    # start generating html table
    html = [f'<table class="{table_name}">']
    if title is not None:
        html.append(f"<caption>{title}</caption>")

    # generating main part of table
    html += _list_to_html(data)

    # add formatting to table
    html.append("</table>")

    # set max width of table
    # if True:
    #     html.insert(0, '<div style="max-width:1000px;">')
    #     html.append("</div>")

    if table_style is not None:
        html.append(table_style)

    return ''.join(html)


def _list_to_html(data: dict[list]) -> List[str]:
    """ Generates main body of a table. """
    html = []

    # Headers
    headers = list(data.keys())
    html += _generate_head(headers)

    # Rows
    html += "<tbody>"
    html += _generate_row_list(data)
    html += "</tbody>"

    return html


def _generate_row_list(data: dict[list]):
    length = len(data[list(data.keys())[0]])
    html = ""
    for i in range(length):
        html += "<tr>"
        for header in data.keys():
            html += f"<td>{data[header][i]}</td>"
        html += "</tr>"
    return html


def merge_html_figs(figs, filename: str = "merge.html", auto_open: bool = True):
    """
    Merges plotly figures.
    Parameters
    ----------
    figs: list[go.Figure, str]
        list of figures to append together
    filename:str
        file name
    auto_open: bool
        open html in browser after creating
    """
    if filename[-5:] != ".html":
        filename += ".html"

    with open(filename, 'w') as file:
        file.write(f"<html><head><title>{filename[:-5]}</title><h1>{filename[:-5]}</h1></head><body>" + "\n")
        for fig in figs:
            if isinstance(fig, str):
                file.write(fig)
                continue

            inner_html = fig.to_html(include_plotlyjs="cdn").split('<body>')[1].split('</body>')[0]
            file.write(inner_html)

        file.write("</body></html>" + "\n")

    if auto_open:
        os.system(fr"start {filename}")

