def encode_sequence(seq, max_len=60):
    mapping = {
        'A': [1,0,0,0],
        'T': [0,1,0,0],
        'G': [0,0,1,0],
        'C': [0,0,0,1],
        'N': [0,0,0,0], 
        'D': [0,0,0,0],
        'R': [0,0,0,0],
        'S': [0,0,0,0]# unknown base
    }

    seq = seq.upper().strip()

    encoded = []

    for char in seq:
        if char not in mapping:
            print("Unknown char:", char)  # debug
        encoded.extend(mapping.get(char, [0,0,0,0]))

    # padding
    while len(encoded) < max_len * 4:
        encoded.extend([0,0,0,0])

    return encoded[:max_len * 4]