#!/usr/bin/env python3
"""
Generate synthetic 100-agent x 50-step GABM epidemic data.

Produces the same CSV schema as a real simulation run so all visualizations
work identically. No LLM calls -- dynamics are simulated deterministically.

Building-aware: agents are assigned to specific buildings each timestep,
and contacts only occur between agents at the same building.

Output:  ../data/synthetic_test/run-1/
    synthetic_test_indiv_data_run-1.csv
    synthetic_test_population_data_run-1.csv
    synthetic_test_contacts_run-1.csv
    synthetic_test_metadata_run-1.json
"""

import csv, json, random
from pathlib import Path

# -- Config --------------------------------------------------------------------
SEED               = 42
N_AGENTS           = 100
N_STEPS            = 50
N_INITIAL_INFECTED = 5      # agents 0-4 start infected
CONTACT_RATE       = 5      # max contacts per agent per day at their building
INFECTION_RATE     = 0.10   # 10% per contact
TIME_TO_HEAL       = 6      # days until Infected -> Recovered

OUT_DIR    = Path(__file__).parent.parent / "data" / "synthetic_test" / "run-1"
SLUG       = "synthetic_test"
RUN_SUFFIX = "_run-1"

random.seed(SEED)

# -- Buildings (must match viz/map-layout.js keys) -----------------------------
BUILDING_KEYS = [
    "coliving", "college", "cafe", "garden", "postoffice",
    "uptown", "pond", "pub", "park", "supply",
    "market", "farm", "chapel",
]
BUILDING_NAMES = {
    "coliving":   "Co-Living Space",
    "college":    "Ridgeway College",
    "cafe":       "Bramble Cafe",
    "garden":     "Community Garden",
    "postoffice": "Post Office",
    "uptown":     "Uptown Apts",
    "pond":       "Village Green",
    "pub":        "The Ploughman's Rest",
    "park":       "Lindenfield Park",
    "supply":     "Copper Beech Supply",
    "market":     "Hawthorn Market",
    "farm":       "Orchard Farm",
    "chapel":     "Old Chapel",
}

# -- Agent roster --------------------------------------------------------------
MALE_NAMES = [
    "James","John","Robert","Michael","William","David","Richard","Joseph",
    "Thomas","Charles","Christopher","Daniel","Matthew","Anthony","Mark",
    "Donald","Steven","Paul","Andrew","Joshua","Kenneth","Kevin","Brian",
    "George","Timothy","Ronald","Edward","Jason","Jeffrey","Ryan","Gary",
    "Jacob","Nicholas","Eric","Jonathan","Stephen","Larry","Justin","Scott",
    "Brandon","Benjamin","Samuel","Raymond","Gregory","Frank","Alexander",
    "Patrick","Jack","Dennis","Jerry",
]
FEMALE_NAMES = [
    "Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth","Susan",
    "Jessica","Sarah","Karen","Lisa","Nancy","Betty","Margaret","Sandra",
    "Ashley","Dorothy","Kimberly","Emily","Donna","Michelle","Carol","Amanda",
    "Melissa","Deborah","Stephanie","Rebecca","Sharon","Laura","Cynthia",
    "Kathleen","Amy","Angela","Shirley","Anna","Brenda","Pamela","Emma",
    "Nicole","Helen","Samantha","Katherine","Christine","Debra","Rachel",
    "Carolyn","Janet","Catherine","Maria","Heather",
]
ALL_NAMES = MALE_NAMES + FEMALE_NAMES   # exactly 100 names

TRAITS_POOL = [
    "Amiability","Warmth","Agreeableness","Cooperativeness","Empathy",
    "Conscientiousness","Punctuality","Discipline","Diligence","Orderliness",
    "Extraversion","Gregariousness","Surgency","Assertiveness","Enthusiasm",
    "Anxiety","Moodiness","Impulsivity","Irritability","Caution",
    "Openness","Creativity","Curiosity","Intellect","Imagination",
    "Bossiness","Negligence","Optimism","Independence","Depth",
    "Shallowness","Placidity","Expressiveness","Aimlessness","Stubbornness",
    "Carefulness","Patience","Adaptability","Resilience","Thoughtfulness",
]

agents = [
    {
        "id":     i,
        "name":   ALL_NAMES[i],
        "age":    random.randint(18, 74),
        "traits": ", ".join(random.sample(TRAITS_POOL, 5)),
    }
    for i in range(N_AGENTS)
]

# -- Reasoning templates -------------------------------------------------------
def make_reasoning(agent, health, rate_pct, resp, days_inf=0):
    name = agent["name"]
    rate = f"{rate_pct:.1f}"
    if health == "Infected":
        if resp == "Yes":
            return (
                f"The infection rate in Dewberry Hollow is {rate}%. "
                f"{name} has been feeling unwell for {days_inf} day(s) and suspects the Catasat virus. "
                f"Recognising the responsibility to protect neighbours, {name} decides to stay home and rest."
            )
        else:
            return (
                f"Despite a local infection rate of {rate}%, {name} is experiencing mild symptoms "
                f"but feels capable of going out for essential errands. "
                f"{name} acknowledges the risk but judges it manageable given pressing obligations."
            )
    elif health == "Recovered":
        return (
            f"The infection rate in Dewberry Hollow is {rate}%. "
            f"{name} has recovered from the Catasat virus and now has immunity. "
            f"There is no medical reason to stay home; {name} resumes normal activities."
        )
    else:  # Susceptible
        if rate_pct < 5:
            if resp == "No":
                return (
                    f"The infection rate in Dewberry Hollow is only {rate}%, indicating minimal community spread. "
                    f"{name} feels healthy and has work and personal obligations to fulfil. "
                    f"Given the low risk, staying home is unnecessary."
                )
            else:
                return (
                    f"The infection rate is {rate}% and while {name} feels perfectly well, "
                    f"{name} is cautious by nature and decides a quiet day at home is preferable."
                )
        elif rate_pct < 15:
            if resp == "No":
                return (
                    f"The infection rate has risen to {rate}% in Dewberry Hollow. "
                    f"{name} feels healthy and, while aware of the growing outbreak, "
                    f"has financial and personal responsibilities that make staying home difficult today."
                )
            else:
                return (
                    f"With the infection rate at {rate}%, {name} is increasingly concerned. "
                    f"{name} feels healthy but decides it is safer to stay home today rather than risk exposure."
                )
        elif rate_pct < 30:
            if resp == "No":
                return (
                    f"The infection rate in Dewberry Hollow has climbed to {rate}%. "
                    f"{name} is worried but has urgent obligations outside the home. "
                    f"Despite the elevated risk, {name} decides to go out and take precautions."
                )
            else:
                return (
                    f"A {rate}% infection rate is alarming. {name} chooses to stay home today "
                    f"to protect themselves and limit the spread of the Catasat virus in Dewberry Hollow."
                )
        else:
            if resp == "No":
                return (
                    f"Despite the severe outbreak ({rate}% infected), {name} has critical obligations "
                    f"that cannot be deferred. {name} goes out reluctantly, aware of the high personal risk."
                )
            else:
                return (
                    f"With {rate}% of Dewberry Hollow infected, {name} recognises the gravity of the situation "
                    f"and stays home. Protecting the community is the only responsible choice right now."
                )

def make_health_string(agent, health, days_inf=0):
    name = agent["name"]
    if health == "Susceptible":
        return f"{name} feels normal."
    elif health == "Infected":
        if days_inf <= 2:
            return f"{name} feels a little off today."
        elif days_inf <= 4:
            return f"{name} feels unwell and suspects the Catasat virus."
        else:
            return f"{name} feels quite ill -- almost certainly the Catasat virus."
    else:
        return f"{name} has recovered from the Catasat virus and feels fine."

# -- Epidemic simulation -------------------------------------------------------
health_state  = ["Susceptible"] * N_AGENTS
days_infected = [0] * N_AGENTS
infected_by   = [None] * N_AGENTS   # tracks who infected each agent

# Seed infections: agents 0 through N_INITIAL_INFECTED-1
for i in range(N_INITIAL_INFECTED):
    health_state[i]  = "Infected"
    days_infected[i] = 1

indiv_rows   = []
pop_rows     = []
contact_rows = []
cum_infections = N_INITIAL_INFECTED

for step in range(N_STEPS):
    # -- Snapshot at start of step ---------------------------------------------
    n_S = health_state.count("Susceptible")
    n_I = health_state.count("Infected")
    n_R = health_state.count("Recovered")
    rate_pct = (n_I / N_AGENTS) * 100.0

    # -- Location decisions ----------------------------------------------------
    locations = []    # "home" or building key
    responses = []
    for i in range(N_AGENTS):
        h = health_state[i]
        d = days_infected[i]
        if h == "Infected":
            p_stay = 0.2 + (d / TIME_TO_HEAL) * 0.6
        elif h == "Recovered":
            p_stay = 0.0
        else:
            if rate_pct < 5:
                p_stay = 0.02
            elif rate_pct < 15:
                p_stay = 0.12
            elif rate_pct < 30:
                p_stay = 0.35
            else:
                p_stay = 0.55
        stay = random.random() < p_stay
        if stay:
            locations.append("home")
            responses.append("Yes")
        else:
            # Assign to a building (round-robin with shuffle for variety)
            bldg = BUILDING_KEYS[(i + step * 7) % len(BUILDING_KEYS)]
            locations.append(bldg)
            responses.append("No")

    # -- Contact generation (building-aware) -----------------------------------
    building_agents = {}
    for i in range(N_AGENTS):
        loc = locations[i]
        if loc != "home":
            building_agents.setdefault(loc, []).append(i)

    step_pairs  = set()
    step_pairs_list = []  # ordered list for deterministic round assignment
    contact_adj = {i: [] for i in range(N_AGENTS)}

    for bldg_key, occupants in building_agents.items():
        shuffled = occupants[:]
        random.shuffle(shuffled)
        for i_ag in shuffled:
            slots_left = CONTACT_RATE - len(contact_adj[i_ag])
            if slots_left <= 0:
                continue
            possible = [
                j for j in occupants
                if j != i_ag and j not in contact_adj[i_ag]
            ]
            n_pick = min(slots_left, len(possible))
            if n_pick <= 0:
                continue
            picks = random.sample(possible, n_pick)
            for j_ag in picks:
                pair = (min(i_ag, j_ag), max(i_ag, j_ag))
                if pair not in step_pairs:
                    step_pairs.add(pair)
                    step_pairs_list.append(pair)
                    contact_adj[i_ag].append(j_ag)
                    contact_adj[j_ag].append(i_ag)

    # -- Round assignment (greedy edge-coloring) ----------------------------------
    counts = {}
    for a, b in step_pairs_list:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    max_c = max(counts.values()) if counts else 0

    pair_assigned = [False] * len(step_pairs_list)
    rounds = []
    for r in range(max_c):
        busy = set()
        round_pairs = []
        for i, (a, b) in enumerate(step_pairs_list):
            if pair_assigned[i]:
                continue
            if a not in busy and b not in busy:
                round_pairs.append((a, b))
                busy.add(a)
                busy.add(b)
                pair_assigned[i] = True
        rounds.append(round_pairs)

    # -- Location assignment per round (constraint-based) -------------------------
    MAX_PAIRS_PER_BUILDING = 5
    agent_visited = {}  # agent -> set of buildings visited this step
    for r, round_pairs in enumerate(rounds):
        bldg_load = {}
        for a, b in round_pairs:
            if a not in agent_visited:
                agent_visited[a] = set()
            if b not in agent_visited:
                agent_visited[b] = set()
            excluded = agent_visited[a] | agent_visited[b]
            cands = [k for k in BUILDING_KEYS
                     if k not in excluded
                     and bldg_load.get(k, 0) < MAX_PAIRS_PER_BUILDING]
            if not cands:
                cands = sorted(BUILDING_KEYS, key=lambda k: bldg_load.get(k, 0))
            loc = cands[(a * 31 + b * 17 + step * 7 + r * 53) % len(cands)]
            bldg_load[loc] = bldg_load.get(loc, 0) + 1
            agent_visited[a].add(loc)
            agent_visited[b].add(loc)
            contact_rows.append({
                "timestep":   step,
                "round":      r,
                "agent_a_id": a,
                "agent_b_id": b,
                "location":   loc,
            })

    # -- Infection transmission ------------------------------------------------
    newly_infected = {}   # agent_id -> infector_id
    for (a, b) in step_pairs:
        if health_state[a] == "Infected" and health_state[b] == "Susceptible":
            if random.random() < INFECTION_RATE:
                if b not in newly_infected:
                    newly_infected[b] = a
        elif health_state[b] == "Infected" and health_state[a] == "Susceptible":
            if random.random() < INFECTION_RATE:
                if a not in newly_infected:
                    newly_infected[a] = b

    # -- Build individual rows for this step -----------------------------------
    for i in range(N_AGENTS):
        ag = agents[i]
        h  = health_state[i]
        d  = days_infected[i]
        loc = locations[i]
        reasoning = make_reasoning(ag, h, rate_pct, responses[i], d)
        indiv_rows.append({
            "agent_id":         i,
            "name":             ag["name"],
            "age":              ag["age"],
            "traits":           ag["traits"],
            "timestep":         step,
            "health_condition": h,
            "health_string":    make_health_string(ag, h, d),
            "reasoning":        reasoning,
            "response":         responses[i],
            "location":         loc,
            "building":         BUILDING_NAMES.get(loc, "home"),
            "infected_by":      infected_by[i] if infected_by[i] is not None else "",
            "format_valid":     True,
            "input_tokens":     random.randint(290, 380),
            "output_tokens":    random.randint(40, 130),
            "reasoning_tokens": 0,
        })

    # -- Population row --------------------------------------------------------
    new_inf_count    = len(newly_infected)
    cum_infections  += new_inf_count
    total_contact    = len(step_pairs) * 2
    n_home           = sum(1 for loc in locations if loc == "home")
    n_grid           = N_AGENTS - n_home
    pop_rows.append({
        "Step":                   step,
        "Susceptible":            n_S,
        "Infected":               n_I,
        "Recovered":              n_R,
        "# Home":                 n_home,
        "# Grid":                 n_grid,
        "New Infections":         new_inf_count,
        "Cumulative Infections":  cum_infections,
        "Total Contact":          total_contact,
        "Daily New Cases Day 4":  0,
    })

    # -- Apply state transitions -----------------------------------------------
    for victim, infector in newly_infected.items():
        health_state[victim]  = "Infected"
        days_infected[victim] = 1
        infected_by[victim]   = infector

    for i in range(N_AGENTS):
        if health_state[i] == "Infected":
            days_infected[i] += 1
            if days_infected[i] > TIME_TO_HEAL:
                health_state[i]  = "Recovered"
                days_infected[i] = 0

# -- Write files ---------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# Individual data
write_csv(
    OUT_DIR / f"{SLUG}_indiv_data{RUN_SUFFIX}.csv",
    indiv_rows,
    ["agent_id","name","age","traits","timestep","health_condition","health_string",
     "reasoning","response","location","building","infected_by",
     "format_valid","input_tokens","output_tokens","reasoning_tokens"],
)

# Population data
write_csv(
    OUT_DIR / f"{SLUG}_population_data{RUN_SUFFIX}.csv",
    pop_rows,
    ["Step","Susceptible","Infected","Recovered","# Home","# Grid",
     "New Infections","Cumulative Infections","Total Contact","Daily New Cases Day 4"],
)

# Contacts data
write_csv(
    OUT_DIR / f"{SLUG}_contacts{RUN_SUFFIX}.csv",
    contact_rows,
    ["timestep","round","agent_a_id","agent_b_id","location"],
)

# Metadata
metadata = {
    "provider":      "synthetic",
    "model":         "deterministic",
    "reasoning":     "none",
    "provider_slug": SLUG,
    "run_number":    1,
    "parameters": {
        "name":           "synthetic_test",
        "contact_rate":   CONTACT_RATE,
        "infection_rate": INFECTION_RATE,
        "no_init_healthy": N_AGENTS - N_INITIAL_INFECTED,
        "no_init_infect":  N_INITIAL_INFECTED,
        "no_days":         N_STEPS,
        "time_to_heal":    TIME_TO_HEAL,
    },
    "final_counts": {
        "Susceptible": health_state.count("Susceptible"),
        "Infected":    health_state.count("Infected"),
        "Recovered":   health_state.count("Recovered"),
        "cum_infections": cum_infections,
    },
}
with open(OUT_DIR / f"{SLUG}_metadata{RUN_SUFFIX}.json", "w") as f:
    json.dump(metadata, f, indent=2)

# -- Summary -------------------------------------------------------------------
print(f"Output: {OUT_DIR}")
print(f"Agents:        {N_AGENTS}")
print(f"Steps:         {N_STEPS}")
print(f"Rows (indiv):  {len(indiv_rows)}")
print(f"Contact pairs: {len(contact_rows)}")
print(f"Final state:   S={metadata['final_counts']['Susceptible']}  "
      f"I={metadata['final_counts']['Infected']}  "
      f"R={metadata['final_counts']['Recovered']}")
print(f"Total infected over run: {cum_infections}")
print("Done.")
