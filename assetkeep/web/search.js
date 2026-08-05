// The query grammar, mirrored from assetkeep/search.py.
//
// The duplication is deliberate and small. The alternative is a round trip to
// the server every time someone types a character just to know how to draw the
// chips, and the parse is twenty lines. What must not drift is the field list,
// so it is the only thing here worth checking against the Python when the
// grammar grows.

const KNOWN_FIELDS = new Set([
  "tag", "kind", "license", "source", "root", "collection", "has", "is",
  "sort", "similar",
  "w", "width", "h", "height", "size", "bytes", "tris", "triangles", "verts",
  "dur", "duration", "frames", "cols", "rows", "colors",
]);

const TOKEN = /-?(?:[a-zA-Z_]+:)?"[^"]*"|\S+/g;
const COMPARISON = /^(>=|<=|>|<|=)?(.*)$/;

export const DEFAULT_SORT = "added";

// Parse a query string into `{ terms, text, sort, similarTo }`.
export function parse(input) {
  const terms = [];
  const text = [];
  let sort = DEFAULT_SORT;
  let similarTo = null;

  for (const token of (input || "").match(TOKEN) || []) {
    const negated = token.startsWith("-");
    const body = negated ? token.slice(1) : token;
    const colon = body.indexOf(":");
    const field = colon > 0 ? body.slice(0, colon).toLowerCase() : "";

    if (colon < 0 || !KNOWN_FIELDS.has(field)) {
      text.push(body.replace(/"/g, ""));
      continue;
    }

    const value = body.slice(colon + 1).replace(/"/g, "");
    if (field === "sort") {
      sort = value || DEFAULT_SORT;
      continue;
    }
    if (field === "similar") {
      similarTo = /^\d+$/.test(value) ? Number(value) : null;
      continue;
    }

    const [, operator, operand] = COMPARISON.exec(value);
    terms.push({ field, operator: operator || "=", value: operand, negated });
  }

  return { terms, text, sort, similarTo };
}

export function serialise(query) {
  const parts = [...query.text];
  for (const term of query.terms) {
    parts.push(termToString(term));
  }
  if (query.similarTo !== null && query.similarTo !== undefined) {
    parts.push(`similar:${query.similarTo}`);
  }
  if (query.sort !== DEFAULT_SORT) parts.push(`sort:${query.sort}`);
  return parts.join(" ");
}

export function termToString(term) {
  const prefix = term.negated ? "-" : "";
  const operator = term.operator === "=" ? "" : term.operator;
  return `${prefix}${term.field}:${operator}${term.value}`;
}

// Add a filter, or remove it if the identical one is already there. This is
// what makes a sidebar click a toggle rather than a way to accumulate seven
// copies of `tag:prop`.
export function toggleTerm(input, field, value, negated = false) {
  const query = parse(input);
  const before = query.terms.length;
  query.terms = query.terms.filter(
    (term) =>
      !(term.field === field && term.value === value && term.negated === negated),
  );
  if (query.terms.length === before) {
    query.terms.push({ field, operator: "=", value, negated });
  }
  return serialise(query);
}

export function hasTerm(input, field, value, negated = false) {
  return parse(input).terms.some(
    (term) =>
      term.field === field && term.value === value && term.negated === negated,
  );
}

export function withSort(input, sort) {
  const query = parse(input);
  query.sort = sort;
  return serialise(query);
}

export function removeTerm(input, index) {
  const query = parse(input);
  query.terms.splice(index, 1);
  return serialise(query);
}

// Everything the query says, as one flat list for the chip row: filters plus
// the bare words, since a stray word is as much a filter as `tag:x` is and
// people forget they typed it.
export function chips(input) {
  const query = parse(input);
  const out = query.terms.map((term, index) => ({
    kind: "term",
    index,
    negated: term.negated,
    label: termToString(term).replace(/^-/, ""),
  }));
  if (query.sort !== DEFAULT_SORT) {
    out.push({ kind: "sort", negated: false, label: `sort:${query.sort}` });
  }
  if (query.similarTo != null) {
    out.push({
      kind: "similar",
      negated: false,
      label: `similar:${query.similarTo}`,
    });
  }
  return out;
}
