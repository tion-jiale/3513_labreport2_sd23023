import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definition --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # "bit"
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


# -------------------- Final Population Storage --------------------
if "_final_pop" not in st.session_state:
    st.session_state["_final_pop"] = None
    st.session_state["_final_fit"] = None

def _store_final(pop, fit):
    st.session_state["_final_pop"] = pop
    st.session_state["_final_fit"] = fit


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    return (
        np.concatenate([a[:point], b[point:]]),
        np.concatenate([b[:point], a[point:]])
    )


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


# -------------------- GA Execution Function --------------------
def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int | None,
    stream_live: bool,
):
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best = []
    history_avg = []
    history_worst = []

    for gen in range(generations):

        # Logging
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if stream_live:
            df = pd.DataFrame({
                "Best": history_best,
                "Average": history_avg,
                "Worst": history_worst,
            })
            chart_area.line_chart(df)
            best_area.markdown(
                f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.6f}**"
            )

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))
        elites_fit = fit[elite_idx].copy() if E > 0 else np.empty((0,))

        # Create next generation
        next_pop = []
        while len(next_pop) < pop_size - E:
            p1_idx = tournament_selection(fit, tournament_k, rng)
            p2_idx = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[p1_idx], pop[p2_idx]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Combine
        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, problem)

    # Final best
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    # Store for population table
    _store_final(pop, fit)

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": df,
        "final_population": pop,
        "final_fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (Bitstring GA)")
st.caption("Maximization GA — Only bitstrings are supported.")

# Sidebar
with st.sidebar:
    st.header("Bitstring GA Settings")

    dim = st.number_input(
        "Chromosome length (bits)",
        min_value=8,
        max_value=4096,
        value=80,
        step=8
    )

    # Custom fitness function (your requirement: peak at 50 ones returns 80)
    def custom_fitness(x):
        ones = np.sum(x)
        return 80 - abs(50 - ones)

    problem = GAProblem(
        name=f"Bit GA ({dim} bits)",
        chromosome_type="bit",
        dim=int(dim),
        bounds=None,
        fitness_fn=custom_fitness,
    )

    st.header("GA Parameters")
    pop_size = st.number_input("Population size", min_value=10, max_value=5000, value=300, step=10)
    generations = st.number_input("Generations", min_value=1, max_value=10000, value=50, step=10)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation rate (per gene)", 0.0, 1.0, 0.01, 0.005)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 100, 2)
    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
    live = st.checkbox("Live chart during run", value=True)

left, right = st.columns([1, 1])

# Run GA
with left:
    if st.button("Run GA", type="primary"):
        result = run_ga(
            problem=problem,
            pop_size=int(pop_size),
            generations=int(generations),
            crossover_rate=float(crossover_rate),
            mutation_rate=float(mutation_rate),
            tournament_k=int(tournament_k),
            elitism=int(elitism),
            seed=int(seed),
            stream_live=bool(live),
        )

        st.subheader("Fitness Over Generations")
        st.line_chart(result["history"])

        st.subheader("Best Solution")
        st.write(f"Best fitness: {result['best_fitness']:.6f}")

        bitstring = ''.join(map(str, result["best"].astype(int).tolist()))
        st.code(bitstring, language="text")
        st.write(f"Number of ones: {int(np.sum(result['best']))} / {problem.dim}")

# Show final population automatically
st.subheader("Final Population (Last Generation)")

if st.session_state["_final_pop"] is not None:
    df = pd.DataFrame(st.session_state["_final_pop"])
    st.dataframe(df, use_container_width=True)
else:
    st.info("Run the algorithm to generate the final population.")
