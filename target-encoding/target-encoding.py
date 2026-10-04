def target_encoding(categories: list, targets: list) -> list:
    stats = {}

    # Phase 1: Track sum and count for each category
    for category, target in zip(categories, targets):
        if category in stats:
            stats[category][0] += target  # Add to sum
            stats[category][1] += 1       # Add to count
        else:
            stats[category] = [target, 1] # [initial_sum, initial_count]

    # Phase 2: Calculate the mean for each category
    means = {}
    for category, values in stats.items():
        total_sum = values[0]
        count = values[1]
        means[category] = total_sum / count

    # Phase 3: Swap original categories with their computed means
    result = []
    for category in categories:
        result.append(means[category])

    return result
