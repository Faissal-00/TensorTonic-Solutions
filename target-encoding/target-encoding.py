def target_encoding(categories: list, targets: list) -> list:
    
    stats = {}
    for category, target in zip(categories, targets) :
        if category in stats :
            stats[category][0] += target
            stats[category][1] += 1
        else :
            stats[category] = [target, 1]

    means = {}
    for category, values in stats.items() :
        total_sum = values[0]
        count = values[1]
        means[category] = total_sum / count

    results = []
    for category in categories:
        results.append(means[category])

    return results