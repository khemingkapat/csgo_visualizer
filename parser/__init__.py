from .parsers import get_top_level, camel_to_snake, parse_json_to_dfs


class Parser:
    get_top_level = staticmethod(get_top_level)
    camel_to_snake = staticmethod(camel_to_snake)
    parse_json_to_dfs = staticmethod(parse_json_to_dfs)
