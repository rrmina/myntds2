import re
import string
import random

# Extracts SQL Statements separated by ';'
def extract_sql_statements(sql_script):

    # Regex Patterns for comments and string literals
    pattern = re.compile(r"""
        (--[^\n]*\n?)               # 1: Single-line comment
      | (/\*[\s\S]*?\*/)            # 2: Multi-line comment
      | ('(?:''|[^'])*')            # 3: Single-quoted string
      | ("(?:\"\"|[^"])*")          # 4: Double-quoted string
    """, re.VERBOSE)

    sanitized_parts = []
    last_pos = 0

    # Find the start and end of regex patterns
    # Replace them with white spaces
    for match in pattern.finditer(sql_script):
        start, end = match.span()                           
        sanitized_parts.append(sql_script[last_pos:start])  # Legit string
        sanitized_parts.append(' ' * (end - start))         # Comment and String replaced with spaces
        last_pos = end
    sanitized_parts.append(sql_script[last_pos:])           # Append edge-case end strings

    # Recombine into a fully masked string
    sanitized_sql = ''.join(sanitized_parts)

    # Find offset of semicolons of the sanitized SQL query
    split_indices = [0]
    for m in re.finditer(';', sanitized_sql):
        split_indices.append(m.end())  # include semicolon
    split_indices.append(len(sanitized_sql))

    # Extract actual SQL statements from original
    statements = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        stmt = sql_script[start:end].strip()
        if stmt:
            statements.append(stmt)

    return statements

def random_alphanumeric_string(length=10):
    chars = string.ascii_letters + string.digits  # a-zA-Z0-9
    return ''.join(random.choices(chars, k=length))