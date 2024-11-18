import matplotlib.pyplot as plt
import numpy as np

# Set Helvetica as the default font for all text in the plots
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = "Helvetica"  # Use Helvetica as sans-serif option

# Increase font sizes for various plot elements
plt.rcParams["font.size"] = 14  # Default font size for all elements
plt.rcParams["axes.titlesize"] = 20  # Font size for subplot titles
plt.rcParams["axes.labelsize"] = 14  # Font size for x and y labels
plt.rcParams["xtick.labelsize"] = 18  # Font size for x tick labels
plt.rcParams["ytick.labelsize"] = 20  # Font size for y tick labels
plt.rcParams["figure.titlesize"] = 20  # Font size for the main figure title
plt.rcParams["legend.fontsize"] = 16  # Font size for legend text (if used)

# Sample data provided
results = {
    "llm": {
        "nle": {
            "gpt-4o-mini": (0.0, 0.0),
            "gpt-4o": (0.3694, 0.36939999999999995),
            "gemini-1.5-flash": (0.0, 0.0),
            "gemini-1.5-pro": (0.30779999999999996, 0.3077999999999999),
            "llama-3.1-70B-it": (0.3508, 0.3508),
            "llama-3.2-11B-it": (0.0, 0.0),
            "llama-3.2-90B-it": (0.0, 0.0),
            "o1-mini": (0.35940000000000005, 0.23968572942269406),
            "o1-preview": (1.568, 0.3956564166041036),
        },
        "minihack": {
            "gpt-4o-mini": (0.0, 0.0),
            "gpt-4o": (5.71, 3.92208362990898),
            "gemini-1.5-flash": (0.0, 0.0),
            "gemini-1.5-pro": (5.71, 3.92208362990898),
            "llama-3.1-70B-it": (0.0, 0.0),
            "llama-3.2-11B-it": (0.0, 0.0),
            "llama-3.2-90B-it": (0.0, 0.0),
        },
        "crafter": {
            "gpt-4o-mini": (12.724, 1.1335442940911777),
            "gpt-4o": (33.095000000000006, 2.3235252192400333),
            "gemini-1.5-flash": (19.996000000000002, 0.7413788954823751),
            "gemini-1.5-pro": (30.205000000000002, 2.862100336000353),
            "llama-3.1-70B-it": (31.312, 2.680171138814336),
            "llama-3.2-11B-it": (26.198, 3.2968634252034836),
            "llama-3.2-90B-it": (31.688000000000006, 1.3576996886073311),
        },
        "babyai": {
            "gpt-4o-mini": (50.4, 4.471992844359213),
            "gpt-4o": (77.6, 3.7290642257810473),
            "gemini-1.5-flash": (25.6, 3.9034753745860877),
            "gemini-1.5-pro": (58.4, 4.408573465419398),
            "llama-3.1-70B-it": (73.2, 3.96157544418884),
            "llama-3.2-11B-it": (32.8, 4.199199923795008),
            "llama-3.2-90B-it": (55.2, 4.447884890596878),
        },
        "babaisai": {
            "gpt-4o-mini": (15.6, 2.5342916458729574),
            "gpt-4o": (33.66, 3.300409730661316),
            "gemini-1.5-flash": (12.8, 2.333384436141903),
            "gemini-1.5-pro": (32.02, 3.2585494303821605),
            "llama-3.1-70B-it": (40.0, 3.4215956910732066),
            "llama-3.2-11B-it": (15.6, 2.5),
            "llama-3.2-90B-it": (43.9, 3.4660654575610024),
        },
        "textworld": {
            "gpt-4o-mini": (12.254901960784309, 3.5489975744272685),
            "gpt-4o": (39.313725490196066, 5.236180272847558),
            "gemini-1.5-flash": (0.0, 0.0),
            "gemini-1.5-pro": (0.0, 0.0),
            "llama-3.1-70B-it": (15.0, 4.6097722286464435),
            "llama-3.2-11B-it": (6.666666666666667, 2.1681627914131262),
            "llama-3.2-90B-it": (11.176, 2.9780000000000006),
        },
    },
    "vlm": {
        "nle": {
            "gpt-4o-mini": (0.0, 0.0),
            "gpt-4o": (0.3694, 0.36939999999999995),
            "gemini-1.5-flash": (0.0, 0.0),
            "gemini-1.5-pro": (0.484, 0.48399999999999993),
            "llama-3.2-11B-it": (0.0, 0.0),
            "llama-3.2-90B-it": (0.0, 0.0),
        },
        "minihack": {
            "gpt-4o-mini": (0.0, 0.0),
            "gpt-4o": (5.71, 3.92208362990898),
            "gemini-1.5-flash": (0.0, 0.0),
            "gemini-1.5-pro": (5.71, 3.92208362990898),
            "llama-3.2-11B-it": (0.0, 0.0),
            "llama-3.2-90B-it": (0.0, 0.0),
        },
        "crafter": {
            "gpt-4o-mini": (19.906, 3.1342318569839938),
            "gpt-4o": (26.813, 3.738469398623512),
            "gemini-1.5-flash": (20.702222222222222, 4.425283034820667),
            "gemini-1.5-pro": (33.504, 2.0684300004270546),
            "llama-3.2-11B-it": (23.634000000000004, 1.4843907841266057),
            "llama-3.2-90B-it": (9.996, 1.1335442940911777),
        },
        "babyai": {
            "gpt-4o-mini": (38.0, 4.3414283363888435),
            "gpt-4o": (62.0, 4.3414283363888435),
            "gemini-1.5-flash": (43.2, 4.430584611538301),
            "gemini-1.5-pro": (58.4, 4.408573465419398),
            "llama-3.2-11B-it": (10.4, 2.7303333129857976),
            "llama-3.2-90B-it": (28.199999999999996, 4.024683838514523),
        },
        "babaisai": {
            "gpt-4o-mini": (16.41, 2.5867503862908015),
            "gpt-4o": (18.62, 2.7187655017780656),
            "gemini-1.5-flash": (8.3, 1.9268450138288429),
            "gemini-1.5-pro": (31.4, 3.2415292988220776),
            "llama-3.2-11B-it": (5.76, 1.63),
            "llama-3.2-90B-it": (21.9, 2.888488506755486),
        },
    },
}

# Define a consistent color mapping for models
model_colors = {
    "gpt-4o-mini": "#45e150",
    "gpt-4o": "#45e19e",
    "gemini-1.5-flash": "#45d6e1",
    "gemini-1.5-pro": "#4588e1",
    "llama-3.1-70B-it": "#5045e1",
    "llama-3.2-11B-it": "#9e45e1",
    "llama-3.2-90B-it": "#eb3840",
    "o1-mini": "#eb8938",
    "o1-preview": "#ebe338",
}

game_ordering = {
    "babyai": "BabyAI",
    "crafter": "Crafter",
    "textworld": "TextWorld",
    "babaisai": "BabaIsAI",
    "minihack": "MiniHack",
    "nle": "NetHack",
}


# Function to plot combined LLM and VLM results with consistent model ordering
def plot_combined_results(results, title, filename, game_ordering):
    # Filter tasks based on the ordering and data availability
    tasks = [
        task
        for task in game_ordering.keys()
        if task in results["llm"] or task in results["vlm"]
    ]

    # Adjusted subplot arrangement
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 15), sharey=True)
    fig.suptitle(title, fontsize=22)
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        models_llm = results["llm"].get(task, {})
        models_vlm = results["vlm"].get(task, {})

        # Get the list of unique models present in this task
        models_in_task = set(models_llm.keys()) | set(models_vlm.keys())
        # Get the desired model order
        desired_model_order = list(model_colors.keys())
        # Order model_names according to desired_model_order, including only models present in this task
        model_names = [
            model for model in desired_model_order if model in models_in_task
        ]
        num_models = len(model_names)
        x = np.arange(num_models)
        bar_width = 0.35  # Adjust as needed

        # Extract llm and vlm values and errors
        llm_values = []
        llm_errors = []
        vlm_values = []
        vlm_errors = []
        colors = []

        for model in model_names:
            colors.append(model_colors.get(model, "#000000"))
            # Get llm data
            if model in models_llm:
                llm_values.append(models_llm[model][0])
                llm_errors.append(models_llm[model][1])
            else:
                llm_values.append(0)
                llm_errors.append(0)
            # Get vlm data
            if model in models_vlm:
                vlm_values.append(models_vlm[model][0])
                vlm_errors.append(models_vlm[model][1])
            else:
                vlm_values.append(0)
                vlm_errors.append(0)

        # Plot llm bars
        ax.bar(
            x - bar_width / 2,
            llm_values,
            width=bar_width,
            yerr=llm_errors,
            capsize=5,
            color=colors,
            label="LLM" if ax == axes[0] else "",
        )
        # Plot vlm bars with hatching
        ax.bar(
            x + bar_width / 2,
            vlm_values,
            width=bar_width,
            yerr=vlm_errors,
            capsize=5,
            color=colors,
            hatch="//",
            label="VLM" if ax == axes[0] else "",
        )

        # Adjust x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title(game_ordering[task])  # Use display name from game_ordering
        ax.set_ylabel("Average Progress (%)")
        ax.set_ylim(0, 100)  # Set y-axis limits to 0-100
        ax.grid(axis="y", linestyle="--", linewidth=0.7)
        if ax == axes[0]:
            ax.legend()

    # Hide any unused subplots
    for i in range(len(tasks), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# Plot and save the combined results
plot_combined_results(
    results, "LLM and VLM Results", "llm_vlm_results.png", game_ordering
)


import math


def compute_average_performance(results, model_type, exclude_tasks=[]):
    model_scores = {}  # model name -> list of scores
    model_errors = {}  # model name -> list of errors
    for task, models in results[model_type].items():
        if task in exclude_tasks:
            continue
        for model, (value, error) in models.items():
            if model not in model_scores:
                model_scores[model] = []
                model_errors[model] = []
            model_scores[model].append(value)
            model_errors[model].append(error)
    # Compute average and standard error per model
    model_stats = {}
    for model in model_scores:
        scores = np.array(model_scores[model])
        errors = np.array(model_errors[model])
        n = len(scores)
        if n > 0:
            average_score = np.mean(scores)
            # Aggregate standard errors using the square root of the sum of squares divided by n
            aggregated_error = np.sqrt(np.sum(errors**2)) / n
            model_stats[model] = (average_score, aggregated_error)
    # Sort models by average performance
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1][0], reverse=True)
    return sorted_models


# Compute LLM performance excluding 'textworld'
llm_performance = compute_average_performance(
    results,
    "llm",
)

# Compute VLM performance
vlm_performance = compute_average_performance(results, "vlm")


# Generate LaTeX tables
def generate_latex_table(performance, title):
    latex_table = f"\\begin{{table}}[h]\n\\centering\n\\caption{{{title}}}\n"
    latex_table += "\\begin{tabular}{lc}\n\\hline\n"
    latex_table += "Model & Average Progress (\%) \\\\\n\\hline\n"
    for model, (avg_score, std_error) in performance:
        latex_table += f"{model} & {avg_score:.2f} $\\pm$ {std_error:.2f} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex_table


# Generate and print LaTeX tables
llm_table = generate_latex_table(llm_performance, "LLM Performance")
vlm_table = generate_latex_table(vlm_performance, "VLM Performance")

print("LaTeX Table for LLM Performance:")
print(llm_table)

print("\nLaTeX Table for VLM Performance:")
print(vlm_table)


# Generate side-by-side LaTeX tables
def generate_side_by_side_latex_tables(
    llm_performance, vlm_performance, llm_title, vlm_title
):
    latex_code = "\\begin{table}[h]\n\\centering\n"
    latex_code += "\\begin{minipage}[t]{0.48\\linewidth}\n\\centering\n"
    latex_code += f"\\caption{{{llm_title}}}\n"
    latex_code += "\\begin{tabular}{lc}\n\\hline\n"
    latex_code += "Model & Average Progress (\\%) \\\\\n\\hline\n"
    for model, (avg_score, std_error) in llm_performance:
        latex_code += f"{model} & {avg_score:.2f} $\\pm$ {std_error:.2f} \\\\\n"
    latex_code += "\\hline\n\\end{tabular}\n\\end{minipage}\n"
    latex_code += "\\hfill\n"
    latex_code += "\\begin{minipage}[t]{0.48\\linewidth}\n\\centering\n"
    latex_code += f"\\caption{{{vlm_title}}}\n"
    latex_code += "\\begin{tabular}{lc}\n\\hline\n"
    latex_code += "Model & Average Progress (\\%) \\\\\n\\hline\n"
    for model, (avg_score, std_error) in vlm_performance:
        latex_code += f"{model} & {avg_score:.2f} $\\pm$ {std_error:.2f} \\\\\n"
    latex_code += "\\hline\n\\end{tabular}\n\\end{minipage}\n"
    latex_code += "\\end{table}\n"
    return latex_code


# Generate and print LaTeX tables
latex_tables = generate_side_by_side_latex_tables(
    llm_performance,
    vlm_performance,
    "LLM Performance (excluding 'textworld')",
    "VLM Performance",
)

print("LaTeX Code for Side-by-Side Tables:")
print(latex_tables)
