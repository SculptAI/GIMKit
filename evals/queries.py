from gimkit import guide as g


user_profile = f"""I'm {g.person_name(name="pred")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.select(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Bio

{g(name="bio", desc="Four sentences.", regex=r"(?:[^.!?]+[.!?]){4}")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""

# TODO: add more queries

queries = {
    "User Profile": user_profile,
}
