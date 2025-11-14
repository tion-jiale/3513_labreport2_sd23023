import numpy as np
import streamlit as st
import pandas as pd


POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_FITNESS = 80
GENERATIONS = 50
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN
ELITISM = 2


def fitness(ind):
    ones = int(ind.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)  # max=80 when ones=50


def init_population():
    return np.random.randint(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)

def tournament_selection(fits):
    idx = np.random.randint(0, POP_SIZE, size=TOURNAMENT_K)
    return idx[np.argmax(fits[idx])]

def one_point_crossover(a, b):
    if np.random.rand() > CROSSOVER_RATE:
        return a.copy(), b.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def mutate(ind):
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    out = ind.copy()
    out[mask] = 1 - out[mask]
    return out


def run_ga():
    pop = init_population()
    history = {"Best": [], "Avg": [], "Worst": []}
    best_overall = None
    best_fitness_ever = -999

    for gen in range(GENERATIONS):
        fits = np.array([fitness(ind) for ind in pop])

        # Record fitness progress
        history["Best"].append(np.max(fits))
        history["Avg"].append(np.mean(fits))
        history["Worst"].append(np.min(fits))

        # Track global best
        if np.max(fits) > best_fitness_ever:
            best_fitness_ever = np.max(fits)
            best_overall = pop[np.argmax(fits)].copy()

        # Elitism
        elite_idx = np.argpartition(fits, -ELITISM)[-ELITISM:]
        elites = pop[elite_idx]

        # Generate new population
        new_pop = []
        while len(new_pop) < POP_SIZE - ELITISM:
            p1 = pop[tournament_selection(fits)]
            p2 = pop[tournament_selection(fits)]
            c1, c2 = one_point_crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE - ELITISM:
                new_pop.append(mutate(c2))

        pop = np.vstack([new_pop, elites])

    # Final evaluation
    final_fits = np.array([fitness(ind) for ind in pop])
    pop_snapshot = pd.DataFrame({
        "Chromosome": ["".join(map(str, row)) for row in pop],
        "Fitness": final_fits
    })

    return best_overall, best_fitness_ever, pd.DataFrame(history), final_fits, pop_snapshot


st.set_page_config(page_title="GA — Bit Pattern Solver", page_icon="🧬", layout="wide")
st.title(" Genetic Algorithm — Bit Pattern (80 bits, Target = 50 ones)")

seed = st.number_input("Random seed", min_value=0, value=42)
run = st.button("Run GA", type="primary")

if run:
    np.random.seed(seed)

    best_ind, best_fit, hist, final_fits, snapshot = run_ga()

    ones = int(best_ind.sum())
    zeros = CHROM_LEN - ones
    bitstring = "".join(map(str, best_ind.tolist()))

   
    st.subheader(" Convergence Graph (Best / Avg / Worst)")
    st.line_chart(hist)

    
    st.subheader(" Fitness Progress ")
    st.line_chart({"Fitness": hist["Best"]})


    st.subheader(" Best Result")
    st.write(f"**Best Fitness:** {best_fit}")
    st.write(f"**Number of Ones:** {ones}")
    st.write(f"**Number of Zeros:** {zeros}")
    st.code(bitstring)


    
    st.subheader(" Population Snapshot Table (Final Generation)")
    st.dataframe(snapshot, height=400)

    if ones == TARGET_ONES:
        st.success(" Perfect solution! Exactly 50 ones achieved.")
    else:
        st.info("Good solution. Try different seeds for better results.")
