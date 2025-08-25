input_path = "train_fine_tune.txt"      
output_path = "fine_tune_data.txt"        

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    block = []
    for line in infile:
        line = line.strip()
        if not line:
            continue
        block.append(line)

        if line.startswith("Solution TR:"):
            
            error_line = next((l for l in block if l.startswith("Error:")), None)
            sol_en_line = next((l for l in block if l.startswith("Solution EN:")), None)
            sol_tr_line = next((l for l in block if l.startswith("Solution TR:")), None)

            if error_line and sol_en_line and sol_tr_line:
                outfile.write(f"ERROR: {error_line[len('Error: '):]}\n")
                outfile.write(f"SOLUTION_EN: {sol_en_line[len('Solution EN: '):]}\n")
                outfile.write(f"SOLUTION_TR: {sol_tr_line[len('Solution TR: '):]}\n\n")

            block = []
