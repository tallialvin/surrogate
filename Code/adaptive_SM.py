import random

def predictive_mo(seq_to_process):
    # Simulate 50% chance
    return random.choice([True, False])

def check_seq(seq, fes_seq, all_true=True):
    if not seq:
        return fes_seq, all_true

    i = len(seq) - 1
    while i >= 0:
        obj_lid = seq[i]
        obstacle = seq[:i] + seq[i+1:]
        seq_to_process = [obstacle, obj_lid]
        fes = predictive_mo(seq_to_process)
        print(f"Checking target: {obj_lid}, obstacle: {obstacle}, result: {fes}")
        if fes:
            fes_seq.append(obj_lid)
            # Recurse with the new obstacle and updated fes_seq
            return check_seq(obstacle, fes_seq, all_true)
        i -= 1

    # If all targets failed, remove the last element and set all_true to False
    forced_remove = seq[-1]
    print(f"All targets failed. Forcibly removing {forced_remove}")
    fes_seq.append(forced_remove)
    obstacle = seq[:-1]
    return check_seq(obstacle, fes_seq, False)

# Example usage
seq = [1, 3, 4, 7, 5, 2, 6, 8, 9]
fes_seq = []
result, status = check_seq(seq, fes_seq)
print("Final fes_seq:", result)
print("All True Status:", status)
