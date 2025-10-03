#!/usr/bin/env python3
"""Example demonstrating regex support in GIM.

This example shows how to use the new regex feature in MaskedTag
to constrain the format of generated content.
"""

from gimkit import guide as g, MaskedTag, Query

# Example 1: Using the guide.regex() method
print("Example 1: Using guide.regex()")
phone_tag = g.regex(r"\d{3}-\d{3}-\d{4}", name="phone", desc="A phone number")
print(f"Phone tag: {phone_tag}")
print()

# Example 2: Using guide() with regex parameter
print("Example 2: Using guide() with regex parameter")
code_tag = g(name="code", desc="Three uppercase letters", regex=r"[A-Z]{3}")
print(f"Code tag: {code_tag}")
print()

# Example 3: Creating MaskedTag directly with regex
print("Example 3: Creating MaskedTag directly with regex")
email_tag = MaskedTag(
    name="email",
    desc="Email address",
    regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
print(f"Email tag: {email_tag}")
print()

# Example 4: Using regex in a Query
print("Example 4: Using regex in a Query")
query = Query(
    "Please provide your phone number: ",
    g.regex(r"\d{3}-\d{3}-\d{4}", name="phone"),
    " and code: ",
    g.regex(r"[A-Z]{3}", name="code")
)
print(f"Query: {query}")
print()

# Example 5: How regex affects CFG generation
print("Example 5: How regex affects CFG generation")
print("When using the CFG output type with models, regex patterns")
print("will be used to constrain the generated content to match")
print("the specified pattern.")
print()
print("For example, the query above would generate a CFG that")
print("enforces the phone number format (###-###-####) and")
print("the code format (three uppercase letters).")
