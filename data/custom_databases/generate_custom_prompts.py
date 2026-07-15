import json
from pathlib import Path
from typing import Any

from generate_custom_databases import (
    COUNTRIES,
    COUNTRY_REL_ALIASES,
    POLITICIANS,
    POLITICIAN_REL_ALIASES,
    SPORTS_SUBJECTS,
    SPORT_REL_ALIASES,
)


ROOT = Path(__file__).resolve().parent
FEWSHOT_PREFIX = (
    "Answer the final question with a short factual phrase.\n\n"
    "Question: Where was Ada Lovelace born?\n"
    "Answer: London\n\n"
    "Question: In what year was Pride and Prejudice published?\n"
    "Answer: 1813\n\n"
)


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def build_subject_alias_map() -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}

    def register(canonical: str, aliases: list[str]) -> None:
        merged = [canonical, *aliases]
        for value in merged:
            alias_map[value] = [item for item in merged if item != value]

    for country in COUNTRIES:
        register(country["name"], country["aliases"])

    for politician in POLITICIANS:
        register(politician["name"], politician["aliases"])

    for subject in SPORTS_SUBJECTS:
        register(subject["name"], subject["aliases"])

    extra_objects = {
        "US Dollar": ["USD", "U.S. Dollar"],
        "Pound Sterling": ["British Pound", "GBP"],
        "Euro": ["EUR"],
        "Renminbi": ["Chinese Yuan", "Yuan"],
        "Indian Rupee": ["INR"],
        "Japanese Yen": ["Yen", "JPY"],
        "South Korean Won": ["KRW"],
        "North Korean Won": ["KPW"],
        "Association football": ["football", "soccer"],
        "American football": ["football", "gridiron football"],
        "Basketball": ["basketball"],
        "Tennis": ["tennis"],
        "Athletics": ["track and field"],
        "Prime Minister": ["premier"],
        "President": ["head of state"],
        "Vice President": ["vice president"],
        "Chancellor": ["federal chancellor"],
    }
    for canonical, aliases in extra_objects.items():
        register(canonical, aliases)

    return alias_map


def build_object_alias_map() -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}

    def register(canonical: str, aliases: list[str]) -> None:
        merged = [canonical, *aliases]
        for value in merged:
            alias_map[value] = [item for item in merged if item != value]

    for country in COUNTRIES:
        register(country["name"], country["aliases"])

    for politician in POLITICIANS:
        register(politician["name"], politician["aliases"])

    for subject in SPORTS_SUBJECTS:
        if subject["name"].endswith("national football team"):
            continue
        register(subject["name"], subject["aliases"])

    extra_objects = {
        "US Dollar": ["USD", "U.S. Dollar"],
        "Pound Sterling": ["British Pound", "GBP"],
        "Euro": ["EUR"],
        "Renminbi": ["Chinese Yuan", "Yuan"],
        "Indian Rupee": ["INR"],
        "Japanese Yen": ["Yen", "JPY"],
        "South Korean Won": ["KRW"],
        "North Korean Won": ["KPW"],
        "Association football": ["football", "soccer"],
        "American football": ["football", "gridiron football"],
        "Basketball": ["basketball"],
        "Tennis": ["tennis"],
        "Athletics": ["track and field"],
        "Prime Minister": ["premier"],
        "President": ["head of state"],
        "Vice President": ["vice president"],
        "Chancellor": ["federal chancellor"],
    }
    for canonical, aliases in extra_objects.items():
        register(canonical, aliases)

    return alias_map


def build_relation_alias_map() -> dict[str, list[str]]:
    relation_map: dict[str, list[str]] = {}

    def register_relation_family(canonical: str, aliases: list[str]) -> None:
        family = [canonical, *aliases]
        for relation in family:
            relation_map[relation] = [item for item in family if item != relation]

    for canonical, aliases in COUNTRY_REL_ALIASES.items():
        register_relation_family(canonical, aliases)
    for canonical, aliases in POLITICIAN_REL_ALIASES.items():
        register_relation_family(canonical, aliases)
    for canonical, aliases in SPORT_REL_ALIASES.items():
        register_relation_family(canonical, aliases)

    return relation_map


SUBJECT_ALIASES = build_subject_alias_map()
OBJECT_ALIASES = build_object_alias_map()
RELATION_ALIASES = build_relation_alias_map()


def direct_question(subject: str, relation: str) -> str:
    templates = {
        "Capital": f"What is the capital of {subject}?",
        "Capital City": f"What is the capital city of {subject}?",
        "National Capital": f"What is the national capital of {subject}?",
        "Currency": f"What currency does {subject} use?",
        "Official Currency": f"What is the official currency of {subject}?",
        "Money Used": f"What money is used in {subject}?",
        "Continent": f"Which continent is {subject} in?",
        "Located In Continent": f"Which continent is {subject} located in?",
        "Continent Located In": f"What continent is {subject} located in?",
        "Internet TLD": f"What is the internet TLD for {subject}?",
        "Country Code TLD": f"What is the country code TLD for {subject}?",
        "Internet Suffix": f"What is the internet suffix for {subject}?",
        "ISO Code": f"What is the ISO code for {subject}?",
        "ISO Alpha-2 Code": f"What is the ISO alpha-2 code for {subject}?",
        "Two-Letter ISO Code": f"What is the two-letter ISO code for {subject}?",
        "Largest City": f"What is the largest city in {subject}?",
        "Demonym": f"What is the demonym for {subject}?",
        "Seat of Government": f"Where is the seat of government of {subject}?",
        "Meets In": f"Where does the parliament of {subject} meet?",
        "Issues Currency": f"What currency does the central bank of {subject} issue?",
        "Country Suffix": f"What country suffix is used for internet addresses in {subject}?",
        "Country Served": f"Which country did {subject} serve?",
        "Country Represented": f"Which country did {subject} represent?",
        "Nation Led": f"Which nation did {subject} lead?",
        "Birth Year": f"What is {possessive(subject)} birth year?",
        "Year Born": f"In what year was {subject} born?",
        "Born In Year": f"What year was {subject} born in?",
        "Political Party": f"What political party is {subject} associated with?",
        "Party Affiliation": f"What is {possessive(subject)} party affiliation?",
        "Party": f"What party is {subject} affiliated with?",
        "Highest Office": f"What is the highest office held by {subject}?",
        "Top Office": f"What top office did {subject} hold?",
        "Highest Post": f"What highest post did {subject} hold?",
        "Took Highest Office In": f"In what year did {subject} take their highest office?",
        "Became Leader In": f"In what year did {subject} become leader?",
        "Took Office In": f"In what year did {subject} take office?",
        "Immediate Predecessor": f"Who was {subject}'s immediate predecessor?",
        "Predecessor In Office": f"Who preceded {subject} in office?",
        "Leader Before": f"Who was the leader before {subject}?",
        "Led By": f"Who led the {subject}?",
        "Figurehead": f"Who was the figurehead of the {subject}?",
        "Officeholder": f"Who held the office described by {subject}?",
        "Sport": f"What sport is {subject} associated with?",
        "Discipline": f"What discipline is {subject} associated with?",
        "Sport Played": f"What sport does {subject} play?",
        "Country": f"Which country is {subject} associated with?",
        "Associated Country": f"What associated country does {subject} have?",
        "Country Represented": f"Which country does {subject} represent?",
        "Nickname": f"What nickname is {subject} known by?",
        "Also Known As": f"What is {subject} also known as?",
        "Nickname Used": f"What nickname is used for {subject}?",
        "Iconic Club": f"What club is most associated with {subject}?",
        "Most Associated Club": f"What is {subject}'s most associated club?",
        "Legendary Club": f"What legendary club is associated with {subject}?",
        "Iconic Team": f"What team is most associated with {subject}?",
        "Most Associated Team": f"What is {subject}'s most associated team?",
        "Legendary Team": f"What legendary team is associated with {subject}?",
        "Position": f"What position does {subject} play?",
        "Playing Position": f"What is {subject}'s playing position?",
        "On-Field Role": f"What is {subject}'s on-field role?",
        "Turned Pro In": f"In what year did {subject} turn professional?",
        "Turned Professional In": f"In what year did {subject} turn professional?",
        "Professional Debut Year": f"What is {subject}'s professional debut year?",
        "Signature Event": f"What is {subject}'s signature event?",
        "Best Known Event": f"What event is {subject} best known for?",
        "Signature Discipline": f"What is {subject}'s signature discipline?",
        "City": f"What city is {subject} based in?",
        "Based In City": f"What city is {subject} based in?",
        "Home City": f"What is {subject}'s home city?",
        "Home Venue": f"What is the home venue of {subject}?",
        "Home Stadium": f"What is {subject}'s home stadium?",
        "Home Arena": f"What is {subject}'s home arena?",
        "League": f"What league does {subject} compete in?",
        "Competes In": f"What competition does {subject} compete in?",
        "League Played In": f"What league is {subject} in?",
        "Confederation": f"What confederation does {subject} belong to?",
        "Continental Confederation": f"What continental confederation does {subject} belong to?",
        "Football Confederation": f"What football confederation does {subject} belong to?",
        "Captain at 2022 World Cup": f"Who captained {subject} at the 2022 World Cup?",
        "2022 World Cup Captain": f"Who was the 2022 World Cup captain of {subject}?",
        "Captain in Qatar 2022": f"Who captained {subject} in Qatar 2022?",
        "Captain at Euro 2016": f"Who captained {subject} at Euro 2016?",
        "Euro 2016 Captain": f"Who was the Euro 2016 captain of {subject}?",
        "Captain in France 2016": f"Who captained {subject} in France 2016?",
        "Featured Athlete": f"Which featured athlete appears on the {subject}?",
        "Stars": f"Who stars in the {subject}?",
        "Primary Tenant": f"Who is the primary tenant of {subject}?",
        "Refers To": f"Who or what does {subject} refer to?",
        "Star Athlete or Team": f"What star athlete or team is associated with {subject}?",
        "Major Team": f"What major team is associated with {subject}?",
        "Canonical Referent": f"What is the canonical referent of {subject}?",
        "Winning Quarterback": f"Who was the winning quarterback in {subject}?",
        "Home Club": f"What club calls {subject} home?",
        "Home Team": f"What team calls {subject} home?",
        "La Liga giant": f"Which La Liga giant is associated with {subject}?",
        "NFL team": f"Which NFL team is associated with {subject}?",
        "NBA team": f"Which NBA team is associated with {subject}?",
    }
    if relation in templates:
        return templates[relation]
    if " taking office in " in relation:
        office, year = relation.split(" taking office in ", 1)
        return f"Who was the {office.lower()} of {subject} who took office in {year}?"
    if relation.startswith("Leader after "):
        return f"Who became the leader of {subject} after {relation[len('Leader after '):]}?"
    return f"What is the {relation.lower()} of {subject}?"


def continuation_prompt(subject: str, relation: str) -> str:
    templates = {
        "Capital": f"The capital of {subject} is",
        "Capital City": f"The capital city of {subject} is",
        "National Capital": f"The national capital of {subject} is",
        "Currency": f"The currency used in {subject} is",
        "Official Currency": f"The official currency of {subject} is",
        "Money Used": f"The money used in {subject} is",
        "Continent": f"{subject} is in the continent of",
        "Located In Continent": f"{subject} is located in the continent of",
        "Continent Located In": f"The continent {subject} is located in is",
        "Internet TLD": f"The internet TLD for {subject} is",
        "Country Code TLD": f"The country code TLD for {subject} is",
        "Internet Suffix": f"The internet suffix for {subject} is",
        "ISO Code": f"The ISO code for {subject} is",
        "ISO Alpha-2 Code": f"The ISO alpha-2 code for {subject} is",
        "Two-Letter ISO Code": f"The two-letter ISO code for {subject} is",
        "Largest City": f"The largest city in {subject} is",
        "Demonym": f"The demonym for {subject} is",
        "Seat of Government": f"The seat of government of {subject} is",
        "Meets In": f"The parliament of {subject} meets in",
        "Issues Currency": f"The central bank of {subject} issues the currency",
        "Country Suffix": f"The country suffix used for internet addresses in {subject} is",
        "Country Served": f"{subject} served the country",
        "Country Represented": f"{subject} represented the country",
        "Nation Led": f"{subject} led the nation",
        "Birth Year": f"{subject} was born in",
        "Year Born": f"{subject} was born in",
        "Born In Year": f"{subject} was born in the year",
        "Political Party": f"{subject} is associated with the political party",
        "Party Affiliation": f"{subject}'s party affiliation is",
        "Party": f"{subject} is affiliated with the party",
        "Highest Office": f"The highest office held by {subject} is",
        "Top Office": f"The top office held by {subject} is",
        "Highest Post": f"The highest post held by {subject} is",
        "Took Highest Office In": f"{subject} took their highest office in",
        "Became Leader In": f"{subject} became leader in",
        "Took Office In": f"{subject} took office in",
        "Immediate Predecessor": f"{subject}'s immediate predecessor was",
        "Predecessor In Office": f"The predecessor in office before {subject} was",
        "Leader Before": f"The leader before {subject} was",
        "Led By": f"The {subject} was led by",
        "Figurehead": f"The figurehead of the {subject} was",
        "Officeholder": f"The officeholder described by {subject} was",
        "Sport": f"{subject} is associated with the sport",
        "Discipline": f"{subject} is associated with the discipline",
        "Sport Played": f"The sport played by {subject} is",
        "Country": f"{subject} is associated with the country",
        "Associated Country": f"The associated country of {subject} is",
        "Nickname": f"{subject} is known by the nickname",
        "Also Known As": f"{subject} is also known as",
        "Nickname Used": f"The nickname used for {subject} is",
        "Iconic Club": f"The club most associated with {subject} is",
        "Most Associated Club": f"{subject}'s most associated club is",
        "Legendary Club": f"The legendary club associated with {subject} is",
        "Iconic Team": f"The team most associated with {subject} is",
        "Most Associated Team": f"{subject}'s most associated team is",
        "Legendary Team": f"The legendary team associated with {subject} is",
        "Position": f"{subject} plays the position",
        "Playing Position": f"{subject}'s playing position is",
        "On-Field Role": f"{subject}'s on-field role is",
        "Turned Pro In": f"{subject} turned professional in",
        "Turned Professional In": f"{subject} turned professional in",
        "Professional Debut Year": f"{subject}'s professional debut year was",
        "Signature Event": f"{subject}'s signature event is",
        "Best Known Event": f"The event {subject} is best known for is",
        "Signature Discipline": f"{subject}'s signature discipline is",
        "City": f"{subject} is based in the city of",
        "Based In City": f"{subject} is based in the city of",
        "Home City": f"{subject}'s home city is",
        "Home Venue": f"The home venue of {subject} is",
        "Home Stadium": f"{subject}'s home stadium is",
        "Home Arena": f"{subject}'s home arena is",
        "League": f"{subject} competes in the league",
        "Competes In": f"{subject} competes in",
        "League Played In": f"The league played in by {subject} is",
        "Confederation": f"{subject} belongs to the confederation",
        "Continental Confederation": f"{subject} belongs to the continental confederation",
        "Football Confederation": f"The football confederation of {subject} is",
        "Captain at 2022 World Cup": f"The captain of {subject} at the 2022 World Cup was",
        "2022 World Cup Captain": f"The 2022 World Cup captain of {subject} was",
        "Captain in Qatar 2022": f"The captain of {subject} in Qatar 2022 was",
        "Captain at Euro 2016": f"The captain of {subject} at Euro 2016 was",
        "Euro 2016 Captain": f"The Euro 2016 captain of {subject} was",
        "Captain in France 2016": f"The captain of {subject} in France 2016 was",
        "Featured Athlete": f"The featured athlete on the {subject} is",
        "Stars": f"The star in the {subject} is",
        "Primary Tenant": f"The primary tenant of {subject} is",
        "Refers To": f"{subject} refers to",
        "Star Athlete or Team": f"The star athlete or team associated with {subject} is",
        "Major Team": f"The major team associated with {subject} is",
        "Canonical Referent": f"The canonical referent of {subject} is",
        "Winning Quarterback": f"The winning quarterback in {subject} was",
        "Home Club": f"The club that calls {subject} home is",
        "Home Team": f"The team that calls {subject} home is",
        "La Liga giant": f"The La Liga giant associated with {subject} is",
        "NFL team": f"The NFL team associated with {subject} is",
        "NBA team": f"The NBA team associated with {subject} is",
    }
    if relation in templates:
        return f"Tell me about {subject}. {templates[relation]}"
    if " taking office in " in relation:
        office, year = relation.split(" taking office in ", 1)
        return f"Tell me about {subject}. The {office.lower()} of {subject} who took office in {year} was"
    if relation.startswith("Leader after "):
        name = relation[len("Leader after ") :]
        return f"Tell me about {subject}. The leader of {subject} after {name} was"
    return f"Tell me about {subject}. The {relation.lower()} of {subject} is"


def cloze_prompt(subject: str, relation: str) -> str:
    base = continuation_prompt(subject, relation).replace(f"Tell me about {subject}. ", "")
    return f"Complete the sentence with the missing fact: {base} ____."


def paraphrased_question(subject: str, relation: str) -> str:
    return f"Could you answer this question: {direct_question(subject, relation)}"


def contextual_question(subject: str, relation: str) -> str:
    return (
        f"Context: I am compiling a concise factual profile for {subject}. "
        f"Answer with a short factual phrase.\nQuestion: {direct_question(subject, relation)}"
    )


def fewshot_question(subject: str, relation: str) -> str:
    return FEWSHOT_PREFIX + f"Question: {direct_question(subject, relation)}\nAnswer:"


def prompt_row(
    prompt_id: int,
    fact_id: int,
    subject: str,
    relation: str,
    gold_object: str,
    prompt_text: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prompt_id": prompt_id,
        "fact_id": fact_id,
        "subject": subject,
        "relation": relation,
        "gold_object": gold_object,
        "prompt_text": prompt_text,
    }
    if subject in SUBJECT_ALIASES and SUBJECT_ALIASES[subject]:
        row["subject_aliases"] = SUBJECT_ALIASES[subject]
    if relation in RELATION_ALIASES and RELATION_ALIASES[relation]:
        row["relation_aliases"] = RELATION_ALIASES[relation]
    if gold_object in OBJECT_ALIASES and OBJECT_ALIASES[gold_object]:
        row["object_aliases"] = OBJECT_ALIASES[gold_object]
    return row


def generate_prompt_file(
    db_path: Path,
    out_path: Path,
    builder,
) -> None:
    with db_path.open(encoding="utf-8") as f:
        db = json.load(f)

    rows: list[dict[str, Any]] = []
    for fact_id, (subject, relation, gold_object) in enumerate(db["triplets"]):
        rows.append(
            prompt_row(
                prompt_id=fact_id,
                fact_id=fact_id,
                subject=subject,
                relation=relation,
                gold_object=gold_object,
                prompt_text=builder(subject, relation),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_variant_prompts(domain: str, variant: str) -> None:
    db_path = ROOT / domain / f"{variant}.json"
    prompt_dir = ROOT / domain / "prompts" / variant
    builders = {
        "prompts_direct_questions.jsonl": direct_question,
        "prompts_paraphrased_questions.jsonl": paraphrased_question,
        "prompts_contextual_questions.jsonl": contextual_question,
        "prompts_continuations.jsonl": continuation_prompt,
        "prompts_cloze.jsonl": cloze_prompt,
        "prompts_fewshot_questions.jsonl": fewshot_question,
    }
    for filename, builder in builders.items():
        generate_prompt_file(db_path, prompt_dir / filename, builder)


def generate_all() -> None:
    for domain in ("countries", "politicians", "sports"):
        for variant in ("base", "alias", "noise", "collision"):
            generate_variant_prompts(domain, variant)
            print(f"{domain}/prompts/{variant}: generated")


if __name__ == "__main__":
    generate_all()
