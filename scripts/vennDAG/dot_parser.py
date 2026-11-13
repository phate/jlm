"""
DOT language parser for VennDAG.
Parses Graphviz DOT files into DAGGraph intermediate representation.
Supports digraphs, nodes with attributes (members, labels), edges, subgraphs, comments.
"""

import re
from typing import Dict, Set, List, Tuple, Optional, Any
from .ir import DAGGraph, Node, Edge


class DotLexer:
    """Lexer for Graphviz DOT format."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def _strip_comments_and_whitespace(self) -> None:
        while self.pos < self.length:
            # Whitespace
            if self.text[self.pos].isspace():
                self.pos += 1
                continue
            # Single-line comment //
            if self.text.startswith("//", self.pos):
                newline = self.text.find("\n", self.pos)
                self.pos = newline + 1 if newline != -1 else self.length
                continue
            # Single-line comment #
            if self.text[self.pos] == '#':
                newline = self.text.find("\n", self.pos)
                self.pos = newline + 1 if newline != -1 else self.length
                continue
            # Multi-line comment /* ... */
            if self.text.startswith("/*", self.pos):
                end_comment = self.text.find("*/", self.pos + 2)
                self.pos = end_comment + 2 if end_comment != -1 else self.length
                continue
            break

    def next_token(self) -> Optional[Tuple[str, str]]:
        """
        Returns (token_type, value).
        Token types:
        - 'SYMBOL': '{', '}', '[', ']', ';', ',', '=', '->', '--', ':'
        - 'ID': identifier or keyword
        - 'STRING': quoted string (unescaped)
        - 'HTML': <HTML-like string>
        """
        self._strip_comments_and_whitespace()
        if self.pos >= self.length:
            return None

        # Multi-char symbols
        if self.text.startswith("->", self.pos):
            self.pos += 2
            return ('SYMBOL', '->')
        if self.text.startswith("--", self.pos):
            self.pos += 2
            return ('SYMBOL', '--')

        ch = self.text[self.pos]

        # Single char symbols
        if ch in "{}[];,=:":
            self.pos += 1
            return ('SYMBOL', ch)

        # Quoted string
        if ch == '"':
            self.pos += 1
            chars = []
            while self.pos < self.length:
                c = self.text[self.pos]
                if c == '\\':
                    self.pos += 1
                    if self.pos < self.length:
                        nc = self.text[self.pos]
                        if nc == 'n':
                            chars.append('\n')
                        elif nc == 't':
                            chars.append('\t')
                        elif nc == 'r':
                            chars.append('\r')
                        elif nc == '"':
                            chars.append('"')
                        elif nc == '\\':
                            chars.append('\\')
                        else:
                            chars.append(nc)
                        self.pos += 1
                elif c == '"':
                    self.pos += 1
                    break
                else:
                    chars.append(c)
                    self.pos += 1
            return ('STRING', "".join(chars))

        # HTML-like string < ... >
        if ch == '<':
            self.pos += 1
            depth = 1
            chars = []
            while self.pos < self.length and depth > 0:
                c = self.text[self.pos]
                if c == '<':
                    depth += 1
                elif c == '>':
                    depth -= 1
                    if depth == 0:
                        self.pos += 1
                        break
                chars.append(c)
                self.pos += 1
            return ('HTML', "".join(chars))

        # ID or Number or Word
        match = re.match(r'[a-zA-Z0-9_\.\-]+', self.text[self.pos:])
        if match:
            val = match.group(0)
            self.pos += len(val)
            return ('ID', val)

        # Catch-all single char
        self.pos += 1
        return ('SYMBOL', ch)


class DotParser:
    """Parser for DOT files producing a DAGGraph."""

    def __init__(self, dot_text: str):
        self.lexer = DotLexer(dot_text)
        self.peeked_token: Optional[Tuple[str, str]] = None

    def _peek(self) -> Optional[Tuple[str, str]]:
        if self.peeked_token is None:
            self.peeked_token = self.lexer.next_token()
        return self.peeked_token

    def _next(self) -> Optional[Tuple[str, str]]:
        tok = self._peek()
        self.peeked_token = None
        return tok

    def _expect_symbol(self, expected: str) -> None:
        tok = self._next()
        if not tok or tok != ('SYMBOL', expected):
            raise ValueError(f"Expected symbol '{expected}', got {tok}")

    def parse_graph(self) -> Optional[DAGGraph]:
        """Parse next DOT graph from token stream, or None if end of input."""
        # Skip any trailing semicolons or whitespace
        while True:
            tok = self._peek()
            if not tok:
                return None
            if tok == ('SYMBOL', ';'):
                self._next()
                continue
            break

        tok = self._peek()
        if not tok:
            return None

        # Optional 'strict'
        if tok[1].lower() == 'strict':
            self._next()
            tok = self._peek()

        is_directed = True
        if tok and tok[1].lower() in ('digraph', 'graph'):
            is_directed = (tok[1].lower() == 'digraph')
            self._next()
        else:
            raise ValueError(f"Expected 'digraph' or 'graph', got {tok}")

        # Optional graph name
        graph_name = "G"
        tok = self._peek()
        if tok and tok[0] in ('ID', 'STRING'):
            graph_name = tok[1]
            self._next()

        self._expect_symbol('{')

        graph = DAGGraph(name=graph_name, is_directed=is_directed)
        self._parse_body(graph)
        return graph

    def parse(self) -> DAGGraph:
        """Parse the first DOT graph in the input."""
        g = self.parse_graph()
        if g is None:
            raise ValueError("Empty DOT content or no graphs found")
        return g

    def parse_all(self) -> List[DAGGraph]:
        """Parse all DOT graphs in the input."""
        graphs: List[DAGGraph] = []
        while True:
            g = self.parse_graph()
            if g is None:
                break
            graphs.append(g)
        return graphs

    def _parse_body(self, graph: DAGGraph) -> None:
        while True:
            tok = self._peek()
            if not tok:
                break
            if tok == ('SYMBOL', '}'):
                self._next()
                break
            if tok == ('SYMBOL', ';'):
                self._next()
                continue

            self._parse_statement(graph)

    def _parse_statement(self, graph: DAGGraph) -> None:
        tok = self._peek()
        if not tok:
            return

        # Subgraph or anonymous block
        if tok[1].lower() == 'subgraph' or tok == ('SYMBOL', '{'):
            if tok[1].lower() == 'subgraph':
                self._next()
                sub_tok = self._peek()
                if sub_tok and sub_tok[0] in ('ID', 'STRING'):
                    self._next()  # subgraph name
            self._expect_symbol('{')
            self._parse_body(graph)
            return

        # Attributes for graph/node/edge defaults: e.g. "node [...]" or "graph [...]"
        if tok[1].lower() in ('graph', 'node', 'edge'):
            kind = tok[1].lower()
            self._next()
            attrs = self._parse_attribute_list()
            if kind == 'graph':
                graph.graph_attributes.update(attrs)
            # Default node/edge attrs can be stored or ignored for DAG logic
            return

        # Could be an assignment: key = val
        first_tok = self._next()
        assert first_tok is not None
        peek_tok = self._peek()

        if peek_tok == ('SYMBOL', '='):
            self._next()  # consume '='
            val_tok = self._next()
            if val_tok:
                graph.graph_attributes[first_tok[1]] = val_tok[1]
            return

        # Could be a node statement or edge statement
        # Node id with port
        first_id = first_tok[1]
        self._consume_port()
        first_attrs = self._parse_attribute_list()
        members = self._extract_members(first_attrs)
        label = first_attrs.get('label')
        graph.add_node(first_id, members=members, label=label, attributes=first_attrs)

        # Check for edge: -> or --
        peek_tok = self._peek()
        if peek_tok and peek_tok[0] == 'SYMBOL' and peek_tok[1] in ('->', '--'):
            # Edge statement
            nodes_in_chain = [first_id]
            while True:
                edge_op = self._peek()
                if not edge_op or edge_op[0] != 'SYMBOL' or edge_op[1] not in ('->', '--'):
                    break
                self._next()  # consume '->'
                target_tok = self._next()
                if not target_tok:
                    raise ValueError("Unexpected end of edge statement")
                self._consume_port()
                target_id = target_tok[1]
                target_attrs = self._parse_attribute_list()
                if target_attrs:
                    t_members = self._extract_members(target_attrs)
                    t_label = target_attrs.get('label')
                    graph.add_node(target_id, members=t_members, label=t_label, attributes=target_attrs)
                else:
                    graph.add_node(target_id)
                nodes_in_chain.append(target_id)

            edge_attrs = self._parse_attribute_list()
            for i in range(len(nodes_in_chain) - 1):
                graph.add_edge(nodes_in_chain[i], nodes_in_chain[i + 1], edge_attrs)
            return

    def _consume_port(self) -> None:
        """Consume optional port specification :port or :port:compass."""
        while True:
            tok = self._peek()
            if tok == ('SYMBOL', ':'):
                self._next()
                port_id = self._next()
                if not port_id:
                    break
            else:
                break

    def _parse_attribute_list(self) -> Dict[str, str]:
        """Parse one or more attribute brackets: [a=b, c=d][e=f]."""
        attrs: Dict[str, str] = {}
        while True:
            tok = self._peek()
            if tok != ('SYMBOL', '['):
                break
            self._next()  # consume '['
            while True:
                tok = self._peek()
                if not tok or tok == ('SYMBOL', ']'):
                    if tok == ('SYMBOL', ']'):
                        self._next()
                    break
                if tok == ('SYMBOL', ',') or tok == ('SYMBOL', ';'):
                    self._next()
                    continue

                key_tok = self._next()
                if not key_tok:
                    break
                key = key_tok[1]

                val = ""
                tok = self._peek()
                if tok == ('SYMBOL', '='):
                    self._next()  # consume '='
                    val_tok = self._next()
                    val = val_tok[1] if val_tok else ""

                attrs[key] = val
        return attrs

    def _extract_members(self, attrs: Dict[str, str]) -> Set[str]:
        """
        Extract set members from node attributes:
        1. Check 'members', 'set', 'elements', 'items', 'payload' attributes.
        2. If not found, inspect 'label' for patterns like '{A, B, C}'.
        """
        for key in ('members', 'set', 'elements', 'items', 'payload'):
            if key in attrs:
                return self._parse_member_string(attrs[key])

        # Try extracting from label only if no explicit attribute was found
        if 'label' in attrs and attrs['label']:
            label = attrs['label']
            # Match pattern { ... }
            m = re.search(r'\{([^}]+)\}', label)
            if m:
                return self._parse_member_string(m.group(1))

        return set()

    @staticmethod
    def _parse_member_string(s: str) -> Set[str]:
        """Parse comma/space/semicolon delimited member string."""
        s = s.strip().strip('{}[]()')
        if not s:
            return set()
        # Split on commas, semicolons, or whitespace
        if ',' in s:
            raw_items = s.split(',')
        elif ';' in s:
            raw_items = s.split(';')
        elif ' ' in s:
            raw_items = s.split()
        else:
            raw_items = [s]

        members = set()
        for item in raw_items:
            cleaned = item.strip().strip('\'" \t\n\r')
            if cleaned:
                members.add(cleaned)
        return members


def _read_dot_text(dot_content_or_file: str) -> str:
    """Read DOT text from string or file path."""
    if '\n' not in dot_content_or_file and (dot_content_or_file.endswith('.dot') or dot_content_or_file.endswith('.gv')):
        with open(dot_content_or_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        try:
            with open(dot_content_or_file, 'r', encoding='utf-8') as f:
                return f.read()
        except (OSError, IOError):
            return dot_content_or_file


def parse_dot(dot_content_or_file: str, multiple: bool = False) -> Any:
    """
    Parse DOT content string or a path to a .dot file.
    If multiple=True, returns List[DAGGraph].
    If multiple=False, returns a single DAGGraph (the first graph).
    """
    text = _read_dot_text(dot_content_or_file)
    parser = DotParser(text)
    if multiple:
        return parser.parse_all()
    return parser.parse()


def parse_dot_graphs(dot_content_or_file: str) -> List[DAGGraph]:
    """
    Parse all DOT graphs in a string or file path into a list of DAGGraph instances.
    """
    text = _read_dot_text(dot_content_or_file)
    parser = DotParser(text)
    return parser.parse_all()
