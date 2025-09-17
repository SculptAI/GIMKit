from gimkit import guide


def llm_request(query: str) -> str:
    return query


g = guide()

prompt = f"""I'm {g.person_name(name="sub")}. Hello, {g.single_word(name="obj")}!

## Bio

{g(name="bio", desc="No more than four sentences.", regex=r"^([A-Za-z][^.!?]*[.!?]\s*){4}$")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""

# Setter; Raise error or wanring or just standardize it and save it to its attr
g.query = prompt

# Setter; Get response, validate its completeness
g.response = llm_request(prompt)

# Iterate all results
for result in g.results:
    print(result)

# Visit results by int id or str name
print(g[0])
print(g["bio"])
assert g[0] == g["sub"]

# Change the value maybe
g[0].content = g[0].content.capitalize() if g[0].content else "hi"

# Infill the masked contents with predications
print(g.infill(g.results))
