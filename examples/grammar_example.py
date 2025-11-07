"""Example demonstrating EBNF grammar support in GIMKit.

This example shows how to use the new `grammar` parameter in MaskedTag
to define custom EBNF patterns for structured text generation.
"""

from gimkit import guide as g
from gimkit.contexts import Query
from gimkit.models.utils import build_cfg


def main():
    print("=" * 80)
    print("EBNF Grammar Support in GIMKit")
    print("=" * 80)
    print()

    # Example 1: Simple inline pattern
    print("Example 1: Simple inline pattern")
    print("-" * 40)
    tag1 = g(name="word", grammar="/[a-z]+/")
    print(f"Tag: {tag1}")
    print(f"Grammar: {tag1.grammar}")
    print()

    # Example 2: Using grammar in a query
    print("Example 2: Using grammar in a query")
    print("-" * 40)
    word_tag = g(name="word", grammar="/[a-z]+/")
    query = Query(f"Generate a word: {word_tag}")
    print(f"Query: {query}")
    cfg = build_cfg(query)
    print("Generated CFG:")
    for line in cfg.definition.split("\n"):
        print(f"  {line}")
    print()

    # Example 3: Grammar takes precedence over regex
    print("Example 3: Grammar takes precedence over regex")
    print("-" * 40)
    tag_both = g(name="number", regex=r"\d+", grammar="/[0-9]{3}/")
    query_both = Query(f"Three digits: {tag_both}")
    cfg_both = build_cfg(query_both)
    print(f"Tag with both regex and grammar: {tag_both}")
    print(f"Regex: {tag_both.regex}")
    print(f"Grammar: {tag_both.grammar}")
    print("Generated CFG uses grammar (not regex):")
    for line in cfg_both.definition.split("\n"):
        print(f"  {line}")
    print()

    # Example 4: Complex grammar with alternatives
    print("Example 4: Complex grammar with alternatives")
    print("-" * 40)
    # Using Lark's alternative syntax
    complex_grammar = '("yes" | "no" | "maybe")'
    tag_choice = g(name="answer", grammar=complex_grammar)
    query_choice = Query(f"Answer: {tag_choice}")
    cfg_choice = build_cfg(query_choice)
    print(f"Grammar with alternatives: {complex_grammar}")
    print("Generated CFG:")
    for line in cfg_choice.definition.split("\n"):
        print(f"  {line}")
    print()

    # Example 5: Multiple tags with different patterns
    print("Example 5: Multiple tags with different patterns")
    print("-" * 40)
    name_tag = g(name="name", desc="A person name")
    activity_tag = g(name="activity", grammar="/[a-z]+ing/")
    count_tag = g(name="count", grammar="/[1-9][0-9]*/")
    query_multi = Query(f"Person: {name_tag} likes {activity_tag} with {count_tag} friends.")
    print(f"Multi-tag query: {query_multi}")
    cfg_multi = build_cfg(query_multi)
    print("Generated CFG:")
    for line in cfg_multi.definition.split("\n"):
        print(f"  {line}")
    print()

    print("=" * 80)
    print("Key Points:")
    print("- Use `grammar` parameter to provide custom EBNF patterns")
    print("- Grammar can be simple patterns like /[a-z]+/ or complex rules")
    print("- Grammar takes precedence over regex when both are provided")
    print("- Users are responsible for providing valid Lark grammar syntax")
    print("=" * 80)


if __name__ == "__main__":
    main()
