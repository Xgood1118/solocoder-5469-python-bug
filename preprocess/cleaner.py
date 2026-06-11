import re


def clean_stack_traces(text):
    stack_pattern = re.compile(
        r'(?:Traceback\s*\(most recent call last\):|'
        r'at\s+[\w.$]+\([\w.$]+\.java:\d+\)|'
        r'File\s+"[^"]+",\s*line\s+\d+.*?(?=\n\n|\Z))',
        re.DOTALL,
    )
    matches = list(stack_pattern.finditer(text))
    if not matches:
        return text, False

    lines = []
    for m in matches:
        segment = m.group(0)
        segment_lines = segment.strip().splitlines()
        lines.extend(segment_lines[:5])

    cleaned = "STACK_TRACE " + " ".join(lines)
    result = stack_pattern.sub("STACK_TRACE", text)
    return result, True


def clean_api_paths(text):
    api_pattern = re.compile(r'(?:/api|/v\d+)/[\w./-]+')
    matches = api_pattern.findall(text)
    if not matches:
        return text, []

    parts = []
    for path in matches:
        components = [p for p in path.split("/") if p]
        parts.extend(components)

    result = api_pattern.sub(" API_PATH ", text)
    return result, parts


def clean_error_codes(text):
    error_patterns = [
        (re.compile(r'\bERR[-_]?[A-Za-z0-9]+\b'), 'err'),
        (re.compile(r'\bHTTP\s*[45]\d{2}\b'), 'http'),
        (re.compile(r'\b[Ee]rror[-_ ]?\d{1,5}\b'), 'err'),
        (re.compile(r'\b[A-Z]{2,}[-_]\d{3,}\b'), 'code'),
    ]

    error_code_types = []
    result = text
    for pattern, code_type in error_patterns:
        matches = pattern.findall(result)
        if matches:
            error_code_types.append(code_type)
            result = pattern.sub(" ERROR_CODE ", result)

    return result, error_code_types


def clean_json_snippets(text):
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
    matches = json_pattern.findall(text)
    if not matches:
        return text, False

    key_pattern = re.compile(r'"(\w+)"\s*:')
    keys = []
    for m in matches:
        keys.extend(key_pattern.findall(m))

    result = json_pattern.sub(" JSON_SNIPPET ", text)
    return result, bool(keys)


def clean_urls(text):
    url_pattern = re.compile(r'https?://([^\s/<>"\']+)(?:/[^\s<>"\']*)?')
    matches = url_pattern.findall(text)
    if not matches:
        return text, False

    domains = list(set(matches))
    domain_markers = " ".join(f"URL_DOMAIN_{d.replace('.', '_')}" for d in domains)
    result = url_pattern.sub(f" {domain_markers} ", text)
    return result, True


def clean_description(text):
    auxiliary_features = {
        "has_stack_trace": False,
        "has_api_path": False,
        "has_error_code": False,
        "has_json": False,
        "has_url": False,
        "api_path_parts": [],
        "error_code_type": [],
    }

    text, has_stack = clean_stack_traces(text)
    auxiliary_features["has_stack_trace"] = has_stack

    text, api_parts = clean_api_paths(text)
    auxiliary_features["has_api_path"] = bool(api_parts)
    auxiliary_features["api_path_parts"] = api_parts

    text, error_types = clean_error_codes(text)
    auxiliary_features["has_error_code"] = bool(error_types)
    auxiliary_features["error_code_type"] = error_types

    text, has_json = clean_json_snippets(text)
    auxiliary_features["has_json"] = has_json

    text, has_url = clean_urls(text)
    auxiliary_features["has_url"] = has_url

    return text, auxiliary_features
