import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_database(triplets: list[tuple[str, str, str]]) -> dict[str, list]:
    deduped = _ordered_unique(["|||".join(t) for t in triplets])
    rows = [tuple(item.split("|||")) for item in deduped]
    return {
        "entities": sorted({subject for subject, _, _ in rows}),
        "relationships": sorted({relation for _, relation, _ in rows}),
        "return_values": sorted({obj for _, _, obj in rows}),
        "triplets": [list(row) for row in rows],
    }


def write_database(path: Path, triplets: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    database = build_database(triplets)
    path.write_text(
        json.dumps(database, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


COUNTRIES = [
    {
        "name": "United States",
        "aliases": ["USA", "United States of America"],
        "capital": "Washington, D.C.",
        "currency": "US Dollar",
        "continent": "North America",
        "tld": ".us",
        "iso": "US",
        "largest_city": "New York City",
        "demonym": "American",
    },
    {
        "name": "United Kingdom",
        "aliases": ["UK", "Britain"],
        "capital": "London",
        "currency": "Pound Sterling",
        "continent": "Europe",
        "tld": ".uk",
        "iso": "GB",
        "largest_city": "London",
        "demonym": "British",
    },
    {
        "name": "France",
        "aliases": ["French Republic", "France"],
        "capital": "Paris",
        "currency": "Euro",
        "continent": "Europe",
        "tld": ".fr",
        "iso": "FR",
        "largest_city": "Paris",
        "demonym": "French",
    },
    {
        "name": "Germany",
        "aliases": ["Federal Republic of Germany", "Deutschland"],
        "capital": "Berlin",
        "currency": "Euro",
        "continent": "Europe",
        "tld": ".de",
        "iso": "DE",
        "largest_city": "Berlin",
        "demonym": "German",
    },
    {
        "name": "Russia",
        "aliases": ["Russian Federation", "Russia"],
        "capital": "Moscow",
        "currency": "Russian Ruble",
        "continent": "Europe",
        "tld": ".ru",
        "iso": "RU",
        "largest_city": "Moscow",
        "demonym": "Russian",
    },
    {
        "name": "China",
        "aliases": ["PRC", "People's Republic of China"],
        "capital": "Beijing",
        "currency": "Renminbi",
        "continent": "Asia",
        "tld": ".cn",
        "iso": "CN",
        "largest_city": "Shanghai",
        "demonym": "Chinese",
    },
    {
        "name": "India",
        "aliases": ["Republic of India", "India"],
        "capital": "New Delhi",
        "currency": "Indian Rupee",
        "continent": "Asia",
        "tld": ".in",
        "iso": "IN",
        "largest_city": "Mumbai",
        "demonym": "Indian",
    },
    {
        "name": "Japan",
        "aliases": ["Japan", "Nippon"],
        "capital": "Tokyo",
        "currency": "Japanese Yen",
        "continent": "Asia",
        "tld": ".jp",
        "iso": "JP",
        "largest_city": "Tokyo",
        "demonym": "Japanese",
    },
    {
        "name": "South Korea",
        "aliases": ["Republic of Korea", "South Korea"],
        "capital": "Seoul",
        "currency": "South Korean Won",
        "continent": "Asia",
        "tld": ".kr",
        "iso": "KR",
        "largest_city": "Seoul",
        "demonym": "South Korean",
    },
    {
        "name": "North Korea",
        "aliases": ["DPRK", "Democratic People's Republic of Korea"],
        "capital": "Pyongyang",
        "currency": "North Korean Won",
        "continent": "Asia",
        "tld": ".kp",
        "iso": "KP",
        "largest_city": "Pyongyang",
        "demonym": "North Korean",
    },
    {
        "name": "Canada",
        "aliases": ["Canada", "CA"],
        "capital": "Ottawa",
        "currency": "Canadian Dollar",
        "continent": "North America",
        "tld": ".ca",
        "iso": "CA",
        "largest_city": "Toronto",
        "demonym": "Canadian",
    },
    {
        "name": "Australia",
        "aliases": ["Commonwealth of Australia", "Australia"],
        "capital": "Canberra",
        "currency": "Australian Dollar",
        "continent": "Oceania",
        "tld": ".au",
        "iso": "AU",
        "largest_city": "Sydney",
        "demonym": "Australian",
    },
    {
        "name": "Brazil",
        "aliases": ["Federative Republic of Brazil", "Brazil"],
        "capital": "Brasilia",
        "currency": "Brazilian Real",
        "continent": "South America",
        "tld": ".br",
        "iso": "BR",
        "largest_city": "São Paulo",
        "demonym": "Brazilian",
    },
    {
        "name": "Mexico",
        "aliases": ["United Mexican States", "Mexico"],
        "capital": "Mexico City",
        "currency": "Mexican Peso",
        "continent": "North America",
        "tld": ".mx",
        "iso": "MX",
        "largest_city": "Mexico City",
        "demonym": "Mexican",
    },
    {
        "name": "Italy",
        "aliases": ["Italian Republic", "Italy"],
        "capital": "Rome",
        "currency": "Euro",
        "continent": "Europe",
        "tld": ".it",
        "iso": "IT",
        "largest_city": "Rome",
        "demonym": "Italian",
    },
    {
        "name": "Spain",
        "aliases": ["Kingdom of Spain", "Spain"],
        "capital": "Madrid",
        "currency": "Euro",
        "continent": "Europe",
        "tld": ".es",
        "iso": "ES",
        "largest_city": "Madrid",
        "demonym": "Spanish",
    },
    {
        "name": "Netherlands",
        "aliases": ["Holland", "Netherlands"],
        "capital": "Amsterdam",
        "currency": "Euro",
        "continent": "Europe",
        "tld": ".nl",
        "iso": "NL",
        "largest_city": "Amsterdam",
        "demonym": "Dutch",
    },
    {
        "name": "Sweden",
        "aliases": ["Kingdom of Sweden", "Sweden"],
        "capital": "Stockholm",
        "currency": "Swedish Krona",
        "continent": "Europe",
        "tld": ".se",
        "iso": "SE",
        "largest_city": "Stockholm",
        "demonym": "Swedish",
    },
    {
        "name": "Norway",
        "aliases": ["Kingdom of Norway", "Norway"],
        "capital": "Oslo",
        "currency": "Norwegian Krone",
        "continent": "Europe",
        "tld": ".no",
        "iso": "NO",
        "largest_city": "Oslo",
        "demonym": "Norwegian",
    },
    {
        "name": "Switzerland",
        "aliases": ["Swiss Confederation", "Switzerland"],
        "capital": "Bern",
        "currency": "Swiss Franc",
        "continent": "Europe",
        "tld": ".ch",
        "iso": "CH",
        "largest_city": "Zürich",
        "demonym": "Swiss",
    },
]


POLITICIANS = [
    {
        "name": "Donald Trump",
        "aliases": ["Donald J. Trump", "Trump"],
        "country": "United States",
        "birth_year": "1946",
        "party": "Republican Party",
        "office": "President",
        "took_office": "2017",
        "predecessor": "Barack Obama",
    },
    {
        "name": "Joe Biden",
        "aliases": ["Joseph R. Biden Jr.", "Biden"],
        "country": "United States",
        "birth_year": "1942",
        "party": "Democratic Party",
        "office": "President",
        "took_office": "2021",
        "predecessor": "Donald Trump",
    },
    {
        "name": "Barack Obama",
        "aliases": ["Barack H. Obama", "Obama"],
        "country": "United States",
        "birth_year": "1961",
        "party": "Democratic Party",
        "office": "President",
        "took_office": "2009",
        "predecessor": "George W. Bush",
    },
    {
        "name": "Kamala Harris",
        "aliases": ["Kamala D. Harris", "Harris"],
        "country": "United States",
        "birth_year": "1964",
        "party": "Democratic Party",
        "office": "Vice President",
        "took_office": "2021",
        "predecessor": "Mike Pence",
    },
    {
        "name": "Hillary Clinton",
        "aliases": ["Hillary Rodham Clinton", "Hillary Clinton"],
        "country": "United States",
        "birth_year": "1947",
        "party": "Democratic Party",
        "office": "Secretary of State",
        "took_office": "2009",
        "predecessor": "Condoleezza Rice",
    },
    {
        "name": "George W. Bush",
        "aliases": ["George Bush", "Bush 43"],
        "country": "United States",
        "birth_year": "1946",
        "party": "Republican Party",
        "office": "President",
        "took_office": "2001",
        "predecessor": "Bill Clinton",
    },
    {
        "name": "Bill Clinton",
        "aliases": ["William J. Clinton", "Clinton"],
        "country": "United States",
        "birth_year": "1946",
        "party": "Democratic Party",
        "office": "President",
        "took_office": "1993",
        "predecessor": "George H. W. Bush",
    },
    {
        "name": "Ronald Reagan",
        "aliases": ["Reagan", "Ronald W. Reagan"],
        "country": "United States",
        "birth_year": "1911",
        "party": "Republican Party",
        "office": "President",
        "took_office": "1981",
        "predecessor": "Jimmy Carter",
    },
    {
        "name": "Boris Johnson",
        "aliases": ["Boris", "Alexander Boris de Pfeffel Johnson"],
        "country": "United Kingdom",
        "birth_year": "1964",
        "party": "Conservative Party",
        "office": "Prime Minister",
        "took_office": "2019",
        "predecessor": "Theresa May",
    },
    {
        "name": "Theresa May",
        "aliases": ["May", "Theresa M. May"],
        "country": "United Kingdom",
        "birth_year": "1956",
        "party": "Conservative Party",
        "office": "Prime Minister",
        "took_office": "2016",
        "predecessor": "David Cameron",
    },
    {
        "name": "Rishi Sunak",
        "aliases": ["Sunak", "Rishi"],
        "country": "United Kingdom",
        "birth_year": "1980",
        "party": "Conservative Party",
        "office": "Prime Minister",
        "took_office": "2022",
        "predecessor": "Liz Truss",
    },
    {
        "name": "Keir Starmer",
        "aliases": ["Starmer", "Sir Keir Starmer"],
        "country": "United Kingdom",
        "birth_year": "1962",
        "party": "Labour Party",
        "office": "Prime Minister",
        "took_office": "2024",
        "predecessor": "Rishi Sunak",
    },
    {
        "name": "Vladimir Putin",
        "aliases": ["Putin", "Vladimir V. Putin"],
        "country": "Russia",
        "birth_year": "1952",
        "party": "United Russia",
        "office": "President",
        "took_office": "2000",
        "predecessor": "Boris Yeltsin",
    },
    {
        "name": "Dmitry Medvedev",
        "aliases": ["Medvedev", "Dmitry A. Medvedev"],
        "country": "Russia",
        "birth_year": "1965",
        "party": "United Russia",
        "office": "President",
        "took_office": "2008",
        "predecessor": "Vladimir Putin",
    },
    {
        "name": "Xi Jinping",
        "aliases": ["Xi", "President Xi"],
        "country": "China",
        "birth_year": "1953",
        "party": "Chinese Communist Party",
        "office": "President",
        "took_office": "2013",
        "predecessor": "Hu Jintao",
    },
    {
        "name": "Narendra Modi",
        "aliases": ["Modi", "Narendra D. Modi"],
        "country": "India",
        "birth_year": "1950",
        "party": "Bharatiya Janata Party",
        "office": "Prime Minister",
        "took_office": "2014",
        "predecessor": "Manmohan Singh",
    },
    {
        "name": "Emmanuel Macron",
        "aliases": ["Macron", "Emmanuel J. Macron"],
        "country": "France",
        "birth_year": "1977",
        "party": "Renaissance",
        "office": "President",
        "took_office": "2017",
        "predecessor": "François Hollande",
    },
    {
        "name": "Angela Merkel",
        "aliases": ["Merkel", "Angela D. Merkel"],
        "country": "Germany",
        "birth_year": "1954",
        "party": "Christian Democratic Union",
        "office": "Chancellor",
        "took_office": "2005",
        "predecessor": "Gerhard Schröder",
    },
    {
        "name": "Justin Trudeau",
        "aliases": ["Trudeau", "Justin P. J. Trudeau"],
        "country": "Canada",
        "birth_year": "1971",
        "party": "Liberal Party",
        "office": "Prime Minister",
        "took_office": "2015",
        "predecessor": "Stephen Harper",
    },
    {
        "name": "Jacinda Ardern",
        "aliases": ["Ardern", "Jacinda K. L. Ardern"],
        "country": "New Zealand",
        "birth_year": "1980",
        "party": "Labour Party",
        "office": "Prime Minister",
        "took_office": "2017",
        "predecessor": "Bill English",
    },
]


SPORTS_SUBJECTS = [
    {
        "name": "Lionel Messi",
        "aliases": ["Leo Messi", "La Pulga"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "Argentina"),
            ("Birth Year", "1987"),
            ("Nickname", "La Pulga"),
            ("Iconic Club", "Barcelona"),
        ],
    },
    {
        "name": "Cristiano Ronaldo",
        "aliases": ["CR7", "Cristiano"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "Portugal"),
            ("Birth Year", "1985"),
            ("Nickname", "CR7"),
            ("Iconic Club", "Real Madrid"),
        ],
    },
    {
        "name": "Tom Brady",
        "aliases": ["TB12", "Brady"],
        "facts": [
            ("Sport", "American football"),
            ("Country", "United States"),
            ("Birth Year", "1977"),
            ("Position", "Quarterback"),
            ("Iconic Team", "New England Patriots"),
        ],
    },
    {
        "name": "LeBron James",
        "aliases": ["King James", "LeBron"],
        "facts": [
            ("Sport", "Basketball"),
            ("Country", "United States"),
            ("Birth Year", "1984"),
            ("Nickname", "King James"),
            ("Iconic Team", "Los Angeles Lakers"),
        ],
    },
    {
        "name": "Michael Jordan",
        "aliases": ["MJ", "Air Jordan"],
        "facts": [
            ("Sport", "Basketball"),
            ("Country", "United States"),
            ("Birth Year", "1963"),
            ("Nickname", "Air Jordan"),
            ("Iconic Team", "Chicago Bulls"),
        ],
    },
    {
        "name": "Patrick Mahomes",
        "aliases": ["Mahomes", "Pat Mahomes"],
        "facts": [
            ("Sport", "American football"),
            ("Country", "United States"),
            ("Birth Year", "1995"),
            ("Position", "Quarterback"),
            ("Iconic Team", "Kansas City Chiefs"),
        ],
    },
    {
        "name": "Serena Williams",
        "aliases": ["Serena", "Serena J. Williams"],
        "facts": [
            ("Sport", "Tennis"),
            ("Country", "United States"),
            ("Birth Year", "1981"),
            ("Nickname", "Serena"),
            ("Turned Pro In", "1995"),
        ],
    },
    {
        "name": "Roger Federer",
        "aliases": ["Federer", "FedEx"],
        "facts": [
            ("Sport", "Tennis"),
            ("Country", "Switzerland"),
            ("Birth Year", "1981"),
            ("Nickname", "FedEx"),
            ("Turned Pro In", "1998"),
        ],
    },
    {
        "name": "Novak Djokovic",
        "aliases": ["Djokovic", "Nole"],
        "facts": [
            ("Sport", "Tennis"),
            ("Country", "Serbia"),
            ("Birth Year", "1987"),
            ("Nickname", "Nole"),
            ("Turned Pro In", "2003"),
        ],
    },
    {
        "name": "Usain Bolt",
        "aliases": ["Bolt", "Lightning Bolt"],
        "facts": [
            ("Sport", "Athletics"),
            ("Country", "Jamaica"),
            ("Birth Year", "1986"),
            ("Nickname", "Lightning Bolt"),
            ("Signature Event", "100 metres"),
        ],
    },
    {
        "name": "Real Madrid",
        "aliases": ["Los Blancos", "Real"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "Spain"),
            ("City", "Madrid"),
            ("Home Venue", "Santiago Bernabéu Stadium"),
            ("League", "La Liga"),
        ],
    },
    {
        "name": "Barcelona",
        "aliases": ["Barça", "Barca"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "Spain"),
            ("City", "Barcelona"),
            ("Home Venue", "Camp Nou"),
            ("League", "La Liga"),
        ],
    },
    {
        "name": "Manchester United",
        "aliases": ["Man United", "Red Devils"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "England"),
            ("City", "Manchester"),
            ("Home Venue", "Old Trafford"),
            ("League", "Premier League"),
        ],
    },
    {
        "name": "Los Angeles Lakers",
        "aliases": ["Lakers", "Purple and Gold"],
        "facts": [
            ("Sport", "Basketball"),
            ("Country", "United States"),
            ("City", "Los Angeles"),
            ("Home Venue", "Crypto.com Arena"),
            ("League", "NBA"),
        ],
    },
    {
        "name": "Kansas City Chiefs",
        "aliases": ["Chiefs", "KC Chiefs"],
        "facts": [
            ("Sport", "American football"),
            ("Country", "United States"),
            ("City", "Kansas City"),
            ("Home Venue", "Arrowhead Stadium"),
            ("League", "NFL"),
        ],
    },
    {
        "name": "New England Patriots",
        "aliases": ["Patriots", "Pats"],
        "facts": [
            ("Sport", "American football"),
            ("Country", "United States"),
            ("City", "Foxborough"),
            ("Home Venue", "Gillette Stadium"),
            ("League", "NFL"),
        ],
    },
    {
        "name": "Golden State Warriors",
        "aliases": ["Warriors", "Dubs"],
        "facts": [
            ("Sport", "Basketball"),
            ("Country", "United States"),
            ("City", "San Francisco"),
            ("Home Venue", "Chase Center"),
            ("League", "NBA"),
        ],
    },
    {
        "name": "Argentina national football team",
        "aliases": ["La Albiceleste", "Argentina"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "Argentina"),
            ("Confederation", "CONMEBOL"),
            ("Captain at 2022 World Cup", "Lionel Messi"),
            ("Nickname", "La Albiceleste"),
        ],
    },
    {
        "name": "Portugal national football team",
        "aliases": ["Seleção das Quinas", "Portugal"],
        "facts": [
            ("Sport", "Association football"),
            ("Country", "Portugal"),
            ("Confederation", "UEFA"),
            ("Captain at Euro 2016", "Cristiano Ronaldo"),
            ("Nickname", "Seleção das Quinas"),
        ],
    },
    {
        "name": "Tampa Bay Buccaneers",
        "aliases": ["Buccaneers", "Bucs"],
        "facts": [
            ("Sport", "American football"),
            ("Country", "United States"),
            ("City", "Tampa"),
            ("Home Venue", "Raymond James Stadium"),
            ("League", "NFL"),
        ],
    },
]


COUNTRY_REL_ALIASES = {
    "Capital": ["Capital City", "National Capital"],
    "Currency": ["Official Currency", "Money Used"],
    "Continent": ["Located In Continent", "Continent Located In"],
    "Internet TLD": ["Country Code TLD", "Internet Suffix"],
    "ISO Code": ["ISO Alpha-2 Code", "Two-Letter ISO Code"],
}

POLITICIAN_REL_ALIASES = {
    "Country Served": ["Country Represented", "Nation Led"],
    "Birth Year": ["Year Born", "Born In Year"],
    "Political Party": ["Party Affiliation", "Party"],
    "Highest Office": ["Top Office", "Highest Post"],
    "Took Highest Office In": ["Became Leader In", "Took Office In"],
    "Immediate Predecessor": ["Predecessor In Office", "Leader Before"],
}

SPORT_REL_ALIASES = {
    "Sport": ["Discipline", "Sport Played"],
    "Country": ["Associated Country", "Country Represented"],
    "Birth Year": ["Year Born", "Born In Year"],
    "Nickname": ["Also Known As", "Nickname Used"],
    "Iconic Club": ["Most Associated Club", "Legendary Club"],
    "Iconic Team": ["Most Associated Team", "Legendary Team"],
    "Position": ["Playing Position", "On-Field Role"],
    "Turned Pro In": ["Turned Professional In", "Professional Debut Year"],
    "Signature Event": ["Best Known Event", "Signature Discipline"],
    "City": ["Based In City", "Home City"],
    "Home Venue": ["Home Stadium", "Home Arena"],
    "League": ["Competes In", "League Played In"],
    "Confederation": ["Continental Confederation", "Football Confederation"],
    "Captain at 2022 World Cup": ["2022 World Cup Captain", "Captain in Qatar 2022"],
    "Captain at Euro 2016": ["Euro 2016 Captain", "Captain in France 2016"],
}


def country_base_triplets() -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for country in COUNTRIES:
        triplets.extend(
            [
                (country["name"], "Capital", country["capital"]),
                (country["name"], "Currency", country["currency"]),
                (country["name"], "Continent", country["continent"]),
                (country["name"], "Internet TLD", country["tld"]),
                (country["name"], "ISO Code", country["iso"]),
            ]
        )
    return triplets


def country_alias_triplets() -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    base_rows = [
        ("Capital", "capital"),
        ("Currency", "currency"),
        ("Continent", "continent"),
        ("Internet TLD", "tld"),
        ("ISO Code", "iso"),
    ]
    for country in COUNTRIES:
        subject_a, subject_b = country["aliases"]
        for relation, key in base_rows:
            alias_a, alias_b = COUNTRY_REL_ALIASES[relation]
            triplets.append((subject_a, alias_a, country[key]))
            triplets.append((subject_b, alias_b, country[key]))
    return triplets


def country_noise_triplets() -> list[tuple[str, str, str]]:
    triplets = country_base_triplets()
    for country in COUNTRIES:
        triplets.extend(
            [
                (
                    f"Government of {country['name']}",
                    "Seat of Government",
                    country["capital"],
                ),
                (f"Parliament of {country['name']}", "Meets In", country["capital"]),
                (
                    f"Central Bank of {country['name']}",
                    "Issues Currency",
                    country["currency"],
                ),
                (
                    f"Internet addresses for {country['name']}",
                    "Country Suffix",
                    country["tld"],
                ),
            ]
        )
    return triplets


def country_collision_triplets() -> list[tuple[str, str, str]]:
    triplets = country_base_triplets()
    for country in COUNTRIES:
        triplets.extend(
            [
                (country["name"], "Largest City", country["largest_city"]),
                (country["name"], "Demonym", country["demonym"]),
            ]
        )
    return triplets


def politician_base_triplets() -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for politician in POLITICIANS:
        triplets.extend(
            [
                (politician["name"], "Country Served", politician["country"]),
                (politician["name"], "Birth Year", politician["birth_year"]),
                (politician["name"], "Political Party", politician["party"]),
                (politician["name"], "Highest Office", politician["office"]),
                (
                    politician["name"],
                    "Took Highest Office In",
                    politician["took_office"],
                ),
                (
                    politician["name"],
                    "Immediate Predecessor",
                    politician["predecessor"],
                ),
            ]
        )
    return triplets


def politician_alias_triplets() -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    fields = {
        "Country Served": "country",
        "Birth Year": "birth_year",
        "Political Party": "party",
        "Highest Office": "office",
        "Took Highest Office In": "took_office",
        "Immediate Predecessor": "predecessor",
    }
    for politician in POLITICIANS:
        alias_a, alias_b = politician["aliases"]
        for relation, key in fields.items():
            rel_a, rel_b = POLITICIAN_REL_ALIASES[relation]
            triplets.append((alias_a, rel_a, politician[key]))
            triplets.append((alias_b, rel_b, politician[key]))
    return triplets


def politician_noise_triplets() -> list[tuple[str, str, str]]:
    triplets = politician_base_triplets()
    for politician in POLITICIANS:
        triplets.extend(
            [
                (f"{politician['name']} administration", "Led By", politician["name"]),
                (
                    f"{politician['name']} political era",
                    "Figurehead",
                    politician["name"],
                ),
                (
                    f"{politician['country']} {politician['office']} taking office in {politician['took_office']}",
                    "Officeholder",
                    politician["name"],
                ),
            ]
        )
    return triplets


def politician_collision_triplets() -> list[tuple[str, str, str]]:
    triplets = politician_base_triplets()
    for politician in POLITICIANS:
        triplets.extend(
            [
                (
                    politician["country"],
                    f"{politician['office']} taking office in {politician['took_office']}",
                    politician["name"],
                ),
                (
                    politician["country"],
                    f"Leader after {politician['predecessor']}",
                    politician["name"],
                ),
            ]
        )
    return triplets


def sports_base_triplets() -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for subject in SPORTS_SUBJECTS:
        for relation, obj in subject["facts"]:
            triplets.append((subject["name"], relation, obj))
    return triplets


def sports_alias_triplets() -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for subject in SPORTS_SUBJECTS:
        alias_a, alias_b = subject["aliases"]
        for relation, obj in subject["facts"]:
            rel_a, rel_b = SPORT_REL_ALIASES[relation]
            triplets.append((alias_a, rel_a, obj))
            triplets.append((alias_b, rel_b, obj))
    return triplets


def sports_noise_triplets() -> list[tuple[str, str, str]]:
    triplets = sports_base_triplets()
    for subject in SPORTS_SUBJECTS:
        facts = dict(subject["facts"])
        if "Country" in facts:
            triplets.append(
                (
                    f"{facts['Country']} sports icon card",
                    "Featured Athlete",
                    subject["name"],
                )
            )
        triplets.append((f"{subject['name']} highlight reel", "Stars", subject["name"]))
        if "Home Venue" in facts:
            triplets.append((facts["Home Venue"], "Primary Tenant", subject["name"]))
        else:
            triplets.append((subject["aliases"][0], "Refers To", subject["name"]))
    return triplets


def sports_collision_triplets() -> list[tuple[str, str, str]]:
    triplets = sports_base_triplets()
    for subject in SPORTS_SUBJECTS:
        facts = dict(subject["facts"])
        if "Country" in facts:
            triplets.append((facts["Country"], "Star Athlete or Team", subject["name"]))
        if "City" in facts:
            triplets.append((facts["City"], "Major Team", subject["name"]))
        else:
            triplets.append(
                (subject["aliases"][0], "Canonical Referent", subject["name"])
            )
    triplets.extend(
        [
            ("Super Bowl LV", "Winning Quarterback", "Tom Brady"),
            ("Super Bowl LVI", "Winning Quarterback", "Matthew Stafford"),
            ("Super Bowl LVII", "Winning Quarterback", "Patrick Mahomes"),
            ("Santiago Bernabéu Stadium", "Home Club", "Real Madrid"),
            ("Camp Nou", "Home Club", "Barcelona"),
            ("Old Trafford", "Home Club", "Manchester United"),
            ("Arrowhead Stadium", "Home Team", "Kansas City Chiefs"),
            ("Gillette Stadium", "Home Team", "New England Patriots"),
            ("Chase Center", "Home Team", "Golden State Warriors"),
            ("Raymond James Stadium", "Home Team", "Tampa Bay Buccaneers"),
            ("Argentina", "Captain at 2022 World Cup", "Lionel Messi"),
            ("Portugal", "Captain at Euro 2016", "Cristiano Ronaldo"),
            ("Spain", "La Liga giant", "Real Madrid"),
            ("Spain", "La Liga giant", "Barcelona"),
            ("United States", "NFL team", "Kansas City Chiefs"),
            ("United States", "NFL team", "New England Patriots"),
            ("United States", "NBA team", "Los Angeles Lakers"),
            ("United States", "NBA team", "Golden State Warriors"),
        ]
    )
    return triplets


def generate_all() -> None:
    datasets = {
        OUT_DIR / "countries" / "base.json": country_base_triplets(),
        OUT_DIR / "countries" / "alias.json": country_alias_triplets(),
        OUT_DIR / "countries" / "noise.json": country_noise_triplets(),
        OUT_DIR / "countries" / "collision.json": country_collision_triplets(),
        OUT_DIR / "politicians" / "base.json": politician_base_triplets(),
        OUT_DIR / "politicians" / "alias.json": politician_alias_triplets(),
        OUT_DIR / "politicians" / "noise.json": politician_noise_triplets(),
        OUT_DIR / "politicians" / "collision.json": politician_collision_triplets(),
        OUT_DIR / "sports" / "base.json": sports_base_triplets(),
        OUT_DIR / "sports" / "alias.json": sports_alias_triplets(),
        OUT_DIR / "sports" / "noise.json": sports_noise_triplets(),
        OUT_DIR / "sports" / "collision.json": sports_collision_triplets(),
    }

    for path, triplets in datasets.items():
        write_database(path, triplets)
        print(f"{path.relative_to(ROOT)}: {len(triplets)} triplets")


if __name__ == "__main__":
    generate_all()
