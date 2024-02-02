from collections import defaultdict
import numpy as np


def generate_positive_sample_lookup(location_index_map, d=1):
    lookup = defaultdict()
    suitable_set = set()
    candidate_iterator = iter(location_index_map.items())
    next_suitable = next(candidate_iterator)


    for anchor_index, anchor_pos in location_index_map.items():
        while (np.linalg.norm(location_index_map[next_suitable[0]] - anchor_pos) < d):
            # suitable_set.update(((next_suitable[0], None)))
            suitable_set.add(next_suitable[0])
            try:
                next_suitable = next(candidate_iterator)
            except StopIteration:
                break

        lookup[anchor_index] = suitable_set.values() #set(suitable_set.values())
        # lookup[anchor_index].remove(anchor_index)
        lookup[anchor_index] = list(lookup[anchor_index])

    return lookup


def generate_triplets(csi_data, location_index_map, num_triplets=1000, d=1):
    positive_sample_lookup = generate_positive_sample_lookup(location_index_map, d)

    triplet_indices = []
    anchor_indices = list(positive_sample_lookup.keys())
    
    while len(triplet_indices) < num_triplets:
        rd_idx = np.random.randint(len(anchor_indices))
        anchor = anchor_indices[rd_idx]

        if len(positive_sample_lookup[anchor]) < 1:
            continue

        positive = positive_sample_lookup[anchor][np.random.randint(len(positive_sample_lookup[anchor]))]
        negative = anchor_indices[np.random.randint(len(anchor_indices))]

        triplet_indices.append((anchor, positive, negative))


    # Load CSI data
    csi_to_load = defaultdict()

    for target, indices in enumerate(triplet_indices):
        for sample in range(3):
            if indices[sample] not in csi_to_load:
                csi_to_load[indices[sample]] = []

                csi_to_load[indices[sample]].append((target, sample))

    csi_to_load = dict(sorted(csi_to_load.items()))

    anchors = [None] * num_triplets
    positives = [None] * num_triplets
    negatives = [None] * num_triplets

    print(f'loading batch of CSI triplets')
    for index, data in enumerate(csi_data):
        if index in csi_to_load:
            for target in csi_to_load[index]:
                # target: triplet_id, sample type (0-a, 1-p, 2-n)
                if target[1] == 0:
                    anchors[target[0]] = data[0]
                elif target[1] == 1:
                    positives[target[0]] = data[0]
                elif target[1] == 2:
                    negatives[target[0]] = data[0]
    print(f'Finished loading tirpelts')

    return ...